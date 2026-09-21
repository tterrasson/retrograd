//! Capability reporting and automatic LoRA-target selection on the local
//! reference GGUF. These tests skip when that optional model is unavailable.

mod common;

use retrograd::{Device, KvDtype, LoraConfig, MIN_BASE_STEP_ULPS, TargetSet, TrainConfig, Trainer};

fn cpu_config() -> TrainConfig {
    TrainConfig {
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

#[test]
fn f16_training_kv_request_falls_back_explicitly_on_cpu() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let trainer = Trainer::new(
        model,
        TrainConfig {
            kv_dtype: KvDtype::F16,
            ..cpu_config()
        },
    )
    .expect("load CPU model with requested F16 KV");
    let report = trainer.backend_report().expect("backend report");
    assert!(
        report.contains("training_kv_requested_dtype: F16"),
        "{report}"
    );
    assert!(report.contains("training_kv_dtype: F32"), "{report}");
    assert!(report.contains("training_kv_f16: fallback_f32"), "{report}");
}

/// The half-precision update matrix as one report line: which optimizers this
/// device runs a rounded store for, and on which grids. A frontend reads it to
/// explain why a precision was turned down.
#[test]
fn the_capability_report_names_each_optimizers_half_precision_storages() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let trainer = Trainer::new(model, cpu_config()).expect("load CPU model");
    let report = trainer.backend_report().expect("backend report");
    let line = report
        .lines()
        .find_map(|line| line.trim().strip_prefix("cap_opt_step: "))
        .expect("the report names the half-precision update matrix");
    // The CPU carries both storages under both optimizers whose step rounds.
    assert_eq!(line.trim(), "adamw{f16, bf16}, sgd{f16, bf16}", "{report}");
    // The F32-only optimizers are absent, not named with an empty brace pair.
    assert!(!line.contains("muon"), "{report}");
    assert!(!line.contains("gefen"), "{report}");
}

/// The floor a half-precision base run is refused under, as one report line.
/// The runtime keeps its own copy of the constant; this checks it agrees with
/// the Rust table.
#[test]
fn the_capability_report_carries_the_half_precision_step_floor() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let trainer = Trainer::new(model, cpu_config()).expect("load CPU model");
    let report = trainer.backend_report().expect("backend report");
    let floor = report
        .lines()
        .find_map(|line| line.trim().strip_prefix("base_step_min_ulps: "))
        .expect("the report names the half-precision step floor")
        .trim()
        .parse::<f32>()
        .expect("the floor is a number");
    assert!(
        (floor - MIN_BASE_STEP_ULPS).abs() < f32::EPSILON,
        "the runtime refuses under {floor} ulp and the table declares \
         {MIN_BASE_STEP_ULPS}:\n{report}"
    );
    // A LoRA run steps no base tensor, so there is no measured step to print.
    let measured = report
        .lines()
        .find_map(|line| line.trim().strip_prefix("base_step_ulps: "))
        .expect("the report names this run's own step")
        .trim()
        .to_string();
    assert_eq!(measured, "n/a", "{report}");
}

#[test]
fn train_preflight_reports_per_device_training_support() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(&model, cpu_config()).expect("load model");

    let pending = trainer.capability_report().expect("capability report");
    assert!(pending.contains("architecture: lfm2"), "{pending}");
    assert!(
        pending.contains("automatic_target_profile: qv-attention"),
        "{pending}"
    );
    assert!(
        pending.contains("automatic_target_patterns: [blk.*.attn_q.weight, blk.*.attn_v.weight]"),
        "{pending}"
    );
    assert!(pending.contains("lora_candidate_patterns:"), "{pending}");
    assert!(pending.contains("model_tensor_types:"), "{pending}");
    assert!(pending.contains("training_graph:"), "{pending}");
    assert!(pending.contains("preflight: pending"), "{pending}");

    // The preflight needs trainable parameters, so it refuses to run first.
    assert!(trainer.train_preflight().is_err());

    trainer
        .create_lora(&LoraConfig::auto(2, 4.0))
        .expect("create LoRA");
    let report = trainer.train_preflight().expect("train preflight");

    assert!(report.contains("training preflight"), "{report}");
    assert!(report.contains("architecture: lfm2"), "{report}");
    assert!(report.contains("missing_gradient_rules: 0"), "{report}");
    // A separate count from the one above, and separate on purpose: that line
    // is an op with no gradient rule at all, this one is a parameter that was
    // marked and still ends the backward with no gradient. A rule that exists
    // and declines the operand it was given is invisible to the first and
    // caught by the second, and the symptom - a marked tensor that trains as a
    // silent no-op - is the same either way.
    assert!(
        report.contains("parameters_without_gradient: 0"),
        "{report}"
    );
    assert!(report.contains("CPU: training graph ready"), "{report}");

    // The preflight result is cached and surfaced by the capability report.
    let capabilities = trainer.capability_report().expect("capability report");
    assert!(capabilities.contains("preflight: passed"), "{capabilities}");
    let automatic_description = trainer.describe_lora().expect("describe auto LoRA");
    assert!(automatic_description.contains("blk.*.attn_q.weight"));
    assert!(automatic_description.contains("blk.*.attn_v.weight"));
}

#[test]
fn explicit_target_overrides_are_supported() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let _guard = common::serialize_models();
    let mut explicit = Trainer::new(&model, cpu_config()).expect("load model");
    let mut config = LoraConfig::auto(2, 4.0);
    config.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    explicit.create_lora(&config).expect("create explicit LoRA");
    let explicit_description = explicit.describe_lora().expect("describe explicit LoRA");
    assert!(explicit_description.contains("blk.2.attn_q.weight"));
    assert!(!explicit_description.contains("blk.*.attn_v.weight"));
}

#[test]
fn real_chat_template_prepares_sft_rows_and_renders_tools_prefix_stably() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let trainer = Trainer::new(&model, cpu_config()).expect("load model");
    let n_ctx = trainer.context_size().expect("context size");
    let path = std::env::temp_dir().join(format!(
        "retrograd-chat-template-{}.jsonl",
        std::process::id()
    ));
    std::fs::write(
        &path,
        "{\"messages\":[{\"role\":\"system\",\"content\":\"Be concise.\"},{\"role\":\"user\",\"content\":\"Say hello.\"},{\"role\":\"assistant\",\"content\":\"Hello!\"}]}\n",
    )
    .unwrap();
    let prepared = retrograd::dataset::prepare(
        &trainer,
        &path,
        retrograd::dataset::DataFormat::ChatJsonl,
        n_ctx,
    )
    .expect("prepare chat JSONL through the model template");
    std::fs::remove_file(path).unwrap();

    assert_eq!(prepared.examples, 1);
    assert_eq!(prepared.tokens.len(), n_ctx);
    assert_eq!(prepared.labels.len(), n_ctx);
    assert!(prepared.supervised_tokens > 0);
    assert!(prepared.supervised_tokens < n_ctx);
    assert_eq!(
        prepared
            .labels
            .iter()
            .filter(|&&label| label != retrograd::dataset::IGNORE_LABEL)
            .count(),
        prepared.supervised_tokens
    );

    // Native tool rendering, end to end on the fixture's real Jinja template.
    assert!(
        trainer
            .chat_template_supports_tools()
            .expect("probe the template"),
        "the reference model's template declares tools; the probe must see them"
    );

    let tools = r#"[{"type":"function","function":{"name":"retro_echo",
        "description":"echoes its input","parameters":{"type":"object",
        "properties":{"text":{"type":"string"}}}}}]"#;
    let opening = r#"[{"role":"system","content":"Be concise."},
        {"role":"user","content":"Echo hi."}]"#;
    let prompt = trainer
        .format_chat_messages(opening, Some(tools), true)
        .expect("render the opening prompt with tools");
    assert!(
        prompt.contains("retro_echo"),
        "the catalog must reach the rendered text: {prompt}"
    );
    assert!(
        !trainer
            .format_chat_messages(opening, None, true)
            .expect("render without tools")
            .contains("retro_echo"),
        "and it must come from the tools we passed, not from the messages"
    );

    // The sampled assistant turn is withheld from the template - a sentinel
    // stands in for it - and the observation comes back framed by the template's
    // own tool role.
    let sentinel = "retro_span_0_9d41c7";
    let with_turn = format!(
        concat!(
            r#"[{{"role":"system","content":"Be concise."}},"#,
            r#"{{"role":"user","content":"Echo hi."}},"#,
            r#"{{"role":"assistant","content":"{sentinel}"}},"#,
            r#"{{"role":"tool","content":"hi","tool_call_id":"call-1"}}]"#,
        ),
        sentinel = sentinel
    );
    let next = trainer
        .format_chat_messages(&with_turn, Some(tools), true)
        .expect("render the tool turn");
    assert!(
        next.contains("<tool_response>")
            || (next.contains("<|im_start|>tool") && next.contains("hi<|im_end|>")),
        "the observation must be framed by the template's own tool role: {next}"
    );

    // The invariant the rollout engine enforces, at the level it enforces it:
    // the framing the template puts *before* the sampled turn must come back
    // unchanged once the tool result is appended, so the tokens generated
    // against it are still the tokens the model saw. The sampled turn itself is
    // never re-rendered and never re-tokenized.
    let (before, after) = next
        .split_once(sentinel)
        .expect("the template must render the assistant turn it was handed");
    assert_eq!(
        trainer.tokenize_text(before).expect("tokenize the opening"),
        trainer.tokenize_text(&prompt).expect("tokenize the prompt"),
        "the opening framing must survive a tool turn being appended"
    );
    assert!(
        !after.is_empty(),
        "the observation must add framing after the sampled turn"
    );

    // Fragments continue a stream someone else opened, so they add no BOS of
    // their own: concatenating them is what assembles a multi-turn prompt.
    let head = trainer.tokenize_text("Echo hi.").expect("tokenize head");
    let fragment = trainer
        .tokenize_fragment("Echo hi.")
        .expect("tokenize fragment");
    assert_eq!(
        head[head.len() - fragment.len()..],
        fragment[..],
        "a fragment is the same tokenization minus whatever BOS the model prepends"
    );
    assert!(trainer.tokenize_fragment("").expect("empty").is_empty());
}

/// JSON variables reach the fixture's Jinja context through the trainer and FFI.
#[test]
fn chat_template_variables_reach_the_template_and_refuse_the_renderers_own_keys() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(&model, cpu_config()).expect("load model");

    // The assistant turn sits before the last user turn, so the template's other condition -
    // `loop.index0 > ns.last_user_index` - is false and `preserve_thinking` is the only thing
    // that can keep the block.
    let messages = r#"[{"role":"user","content":"First."},
        {"role":"assistant","content":"Answer.","thinking":"the reasoning"},
        {"role":"user","content":"Second."}]"#;

    let without = trainer
        .format_chat_messages(messages, None, true)
        .expect("render without variables");
    assert!(
        !without.contains("the reasoning"),
        "the template drops a thinking block by default: {without}"
    );

    trainer
        .set_chat_template_variables(Some(r#"{"preserve_thinking":true}"#))
        .expect("set a template variable");
    let with = trainer
        .format_chat_messages(messages, None, true)
        .expect("render with variables");
    assert!(
        with.contains("the reasoning"),
        "preserve_thinking must reach the template: {with}"
    );

    // Clearing restores the default rather than leaving the last set in place.
    trainer
        .set_chat_template_variables(None)
        .expect("clear the variables");
    assert_eq!(
        trainer
            .format_chat_messages(messages, None, true)
            .expect("render after clearing"),
        without
    );

    // Variables cannot replace the renderer's conversation inputs.
    for reserved in [
        r#"{"messages":[]}"#,
        r#"{"tools":[]}"#,
        r#"{"bos_token":"x"}"#,
        r#"{"eos_token":"x"}"#,
        r#"{"add_generation_prompt":false}"#,
    ] {
        let error = trainer
            .set_chat_template_variables(Some(reserved))
            .expect_err("a reserved key must be refused");
        assert!(
            error.to_string().contains("cannot be set"),
            "unexpected error for {reserved}: {error}"
        );
    }

    // And so is anything that is not a JSON object.
    for bad in [r#"{"unclosed": "#, "[1,2]", r#""a string""#] {
        trainer
            .set_chat_template_variables(Some(bad))
            .expect_err("malformed variables must be refused");
    }
    // A refused set leaves the previous state alone.
    assert_eq!(
        trainer
            .format_chat_messages(messages, None, true)
            .expect("render after a refused set"),
        without
    );
}

#[test]
fn the_derived_parser_reads_the_call_format_this_template_teaches() {
    // The bug of, on the model that produced it: LFM2's
    // template renders its own catalog, so the rollout renders natively and the
    // model answers in `<|tool_call_start|>[…]<|tool_call_end|>` - which the
    // `<tool_call>` parser reads as prose, silently, for a whole run. The
    // parser derived from the same template is what closes that, and only a
    // real model proves the derivation reaches the template it claims to.
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let trainer = Trainer::new(&model, cpu_config()).expect("load model");

    let tools = r#"[{"type":"function","function":{"name":"move",
        "description":"move the agent","parameters":{"type":"object",
        "properties":{"direction":{"type":"string"}},"required":["direction"]}}}]"#;
    let parser = trainer
        .tool_call_parser(Some(tools))
        .expect("derive a parser")
        .expect("this template yields one");

    let document = retrograd::parse_assistant_output(
        &parser,
        "<|tool_call_start|>[move(direction='left')]<|tool_call_end|>",
    )
    .expect("parse an LFM2-formatted call");
    assert!(document.contains("\"move\""), "{document}");
    assert!(document.contains("left"), "{document}");

    // And prose stays prose: a parser that read every answer as a call would
    // end each trajectory on a tool nobody asked for.
    let plain = retrograd::parse_assistant_output(&parser, "the exit is to the left")
        .expect("parse a plain answer");
    assert!(plain.contains("the exit is to the left"), "{plain}");
    assert!(plain.contains("\"tool_calls\":[]"), "{plain}");
}

#[test]
fn detokenizing_for_the_tool_call_parser_keeps_the_models_own_call_delimiters() {
    // The other half of the bug the test above closes: even a template whose
    // parser is derived correctly reads nothing if the text handed to it has
    // already lost the delimiters. `<|tool_call_start|>` and
    // `<|tool_call_end|>` are control tokens in LFM2's vocabulary, same family
    // as `<|im_start|>`/`<|im_end|>`, and `Trainer::detokenize`'s
    // `unparse_special` is the switch between rendering them and dropping
    // them silently to an empty string (llama-vocab.cpp's `token_to_piece`).
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let trainer = Trainer::new(&model, cpu_config()).expect("load model");

    let text = "<|tool_call_start|>[move(direction='left')]<|tool_call_end|>";
    let tokens = trainer.tokenize_text(text).expect("tokenize a native call");

    let for_parser = trainer
        .detokenize(&tokens, true)
        .expect("detokenize for the parser");
    assert!(
        for_parser.contains("<|tool_call_start|>") && for_parser.contains("<|tool_call_end|>"),
        "parser-facing text lost the call delimiters: {for_parser:?}"
    );

    let for_display = trainer
        .detokenize(&tokens, false)
        .expect("detokenize for display");
    assert!(
        !for_display.contains("<|tool_call_start|>") && !for_display.contains("<|tool_call_end|>"),
        "human-facing text unexpectedly kept control tokens: {for_display:?}"
    );
}

/// Enough text for one short optimizer epoch on the fixture's 32-token context.
const DUTY_CYCLE_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";

#[test]
fn a_cpu_backend_accepts_a_duty_cycle_and_says_it_is_not_honouring_it() {
    // `model.device = "auto"` can resolve to CPU, so the document cannot be
    // rejected at parse time. What must not happen is a silent promise: the run
    // keeps the requested value, reports that the limiter did not engage, and
    // never sleeps.
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(
        model,
        TrainConfig {
            n_ctx: 32,
            n_batch: 32,
            n_ubatch: 16,
            max_gpu_duty_cycle: Some(0.5),
            ..cpu_config()
        },
    )
    .expect("load CPU model with a requested duty cycle");

    let report = trainer.backend_report().expect("backend report");
    assert!(report.contains("gpu_duty_cycle_requested: 0.5"), "{report}");
    assert!(report.contains("gpu_duty_cycle_active: false"), "{report}");
    assert!(
        report.contains("gpu_duty_cycle_reason: cpu_backend"),
        "{report}"
    );

    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    trainer.create_lora(&lora).expect("create lora");
    let tokens = trainer
        .tokenize_text(&DUTY_CYCLE_TEXT.repeat(64))
        .expect("tokenize");
    trainer.train_tokens(&tokens).expect("train");

    // A whole epoch of optimizer callbacks and not one of them slept: on the
    // inactive path there is no window to account and no debt to repay.
    let stats = trainer.duty_cycle_stats().expect("duty cycle stats");
    assert_eq!(stats.requested_fraction, 0.5);
    assert!(!stats.active);
    assert_eq!(stats.compute_seconds, 0.0);
    assert_eq!(stats.idle_seconds, 0.0);
    assert_eq!(stats.wall_seconds, 0.0);
    assert_eq!(stats.observed(), None);
    assert_eq!(stats.wall_share(), None);
}

#[test]
fn the_duty_cycle_setter_validates_and_clears_between_operations() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(model, cpu_config()).expect("load CPU model");
    assert_eq!(
        trainer
            .duty_cycle_stats()
            .expect("stats")
            .requested_fraction,
        1.0,
        "an unconfigured trainer requests no limit"
    );

    trainer
        .set_max_gpu_duty_cycle(0.25)
        .expect("a fraction in (0, 1] is accepted");
    assert_eq!(
        trainer
            .duty_cycle_stats()
            .expect("stats")
            .requested_fraction,
        0.25
    );

    // `1.0` is how the C ABI spells "no limit", and it clears the state rather
    // than leaving a limiter installed that happens never to sleep.
    trainer.set_max_gpu_duty_cycle(1.0).expect("1.0 disables");
    let cleared = trainer.duty_cycle_stats().expect("stats");
    assert_eq!(cleared.requested_fraction, 1.0);
    assert!(!cleared.active);

    // Zero is refused rather than read as "pause": run control already owns
    // pausing a live run, and a limiter that never repays is a stall.
    for fraction in [0.0_f32, -0.5, 1.5, f32::NAN, f32::INFINITY] {
        let error = trainer
            .set_max_gpu_duty_cycle(fraction)
            .expect_err("out-of-range fraction accepted");
        assert!(
            error.to_string().contains("max_gpu_duty_cycle"),
            "{fraction}: {error}"
        );
    }
    // A refused setting leaves the previous one in place.
    assert_eq!(
        trainer
            .duty_cycle_stats()
            .expect("stats")
            .requested_fraction,
        1.0
    );
}
