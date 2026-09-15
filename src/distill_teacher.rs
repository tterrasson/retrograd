//! `retrograd distill-teacher`: writes the top-k sidecar an offline
//! distillation run reads.
//!
//! **In the repository rather than in Python, and that is the design.** A
//! producer written outside the crate would have to retokenize the corpus with
//! its own idea of the chat template, and a one-token shift between the sidecar
//! and the training batch is *silent*: the run trains, the loss falls, and the
//! student is fitted to the teacher's opinion about the wrong positions. Here
//! the corpus goes through `retrograd-dataset` and the teacher scores the very
//! rows the trainer will walk, so the alignment is a property of the code path
//! and not something a test has to keep rediscovering.
//!
//! It loads one model - the teacher - and never creates an adapter, so it costs
//! weights plus KV and nothing else, the same invariant `Teacher` is held under
//! on the on-policy path.

use std::path::PathBuf;
use std::time::Instant;

use indicatif::HumanDuration;
use retrograd::config::{self, Algorithm};
use retrograd::dataset::topk::{
    DEFAULT_K, MAX_K, TopKHeader, TopKSidecar, VERSION, corpus_fingerprint, tokenizer_fingerprint,
};
use retrograd::dataset::{self, DataFormat};
use retrograd::training::distill::offline::{score_corpus, witness_ids};
use retrograd::{Device, Error, Result, TrainConfig, Trainer};

use retrograd_cli_ui::CliUi;

pub(crate) const FLAGS: &[&str] = &["--out", "--k", "--data", "--model", "--device", "--ctx"];

#[derive(Debug)]
struct Args {
    config: PathBuf,
    out: Option<PathBuf>,
    k: Option<usize>,
    data: Option<PathBuf>,
    model: Option<PathBuf>,
    device: Option<Device>,
    ctx: Option<u32>,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut parsed = Args {
        config: PathBuf::new(),
        out: None,
        k: None,
        data: None,
        model: None,
        device: None,
        ctx: None,
    };
    let mut positional = None;
    // `Args` names this command's own result, so the shared reader is not
    // imported under that name.
    let mut line = crate::args::Args::new("distill-teacher", args);
    while let Some(argument) = line.next_arg() {
        match argument {
            "--out" => parsed.out = Some(PathBuf::from(line.value(argument)?)),
            "--k" => parsed.k = Some(line.parse(argument, "an integer")?),
            "--data" => parsed.data = Some(PathBuf::from(line.value(argument)?)),
            "--model" => parsed.model = Some(PathBuf::from(line.value(argument)?)),
            "--device" => parsed.device = Some(line.value(argument)?.parse()?),
            "--ctx" => parsed.ctx = Some(line.parse(argument, "an integer")?),
            other if other.starts_with("--") && !FLAGS.contains(&other) => {
                return Err(line.unknown(other));
            }
            other if other.starts_with("--") => {
                // In FLAGS but not matched above: the two lists have drifted.
                return Err(Error::invalid(format!(
                    "distill-teacher flag {other} is declared but not handled"
                )));
            }
            other => {
                if positional.replace(PathBuf::from(other)).is_some() {
                    return Err(Error::invalid(
                        "distill-teacher takes one configuration file",
                    ));
                }
            }
        }
    }
    parsed.config = positional
        .ok_or_else(|| Error::invalid("distill-teacher requires a configuration file"))?;
    if let Some(k) = parsed.k
        && (k == 0 || k > MAX_K)
    {
        return Err(Error::invalid(format!(
            "--k must be in 1..={MAX_K}, got {k}"
        )));
    }
    Ok(parsed)
}

pub(crate) fn distill_teacher(args: Vec<String>) -> Result<()> {
    let args = parse(&args)?;
    let run_config = config::load_with(
        &args.config,
        config::ModelOverride {
            // `--model` names the *student* in every other subcommand, and it
            // does here too: it is what the document's `[model]` stands for, and
            // the sidecar is checked against the student's vocabulary at load.
            // The teacher comes from `[distill].teacher_path`, never from a flag,
            // because a sidecar produced by a model the document does not name is
            // a file nothing can later attribute.
            path: args.model.clone(),
            device: args.device,
        },
    )?;
    let Algorithm::Distill(distill) = &run_config.algorithm else {
        return Err(Error::invalid(
            "distill-teacher needs a [distill] configuration: it is the section that names the \
             teacher whose distribution the sidecar records",
        ));
    };
    let corpus = args
        .data
        .clone()
        .or_else(|| distill.mode.offline().map(|offline| offline.data.clone()))
        .ok_or_else(|| {
            Error::invalid(
                "distill-teacher requires distill.data (with distill.mode = \"topk_offline\") or \
                 an explicit --data path",
            )
        })?;
    let out = args
        .out
        .clone()
        .or_else(|| {
            distill
                .mode
                .offline()
                .map(|offline| offline.sidecar.clone())
        })
        .ok_or_else(|| {
            Error::invalid(
                "distill-teacher requires distill.sidecar (with distill.mode = \"topk_offline\") \
                 or an explicit --out path",
            )
        })?;
    let k = args.k.unwrap_or(DEFAULT_K);

    let mut training = run_config.training.clone();
    if let Some(ctx) = args.ctx {
        training.n_ctx = ctx;
    }

    let ui = CliUi::new();
    ui.section("distill-teacher");
    ui.info(format!("config: {}", args.config.display()));
    ui.info(format!("student: {}", run_config.model.display()));
    ui.info(format!("teacher: {}", distill.teacher_path.display()));
    ui.info(format!("corpus: {}", corpus.display()));
    ui.info(format!("sidecar: {} (k = {k})", out.display()));

    // The student is opened first and only to prepare the corpus. Two models
    // are never resident at once here: the student is dropped before the
    // teacher is loaded, which is what lets this command run on a machine that
    // could not hold the pair.
    let started = Instant::now();
    let prepare = ui.spinner("preparing the corpus");
    let (prepared, vocab_size, witnesses) = {
        let student = Trainer::new(&run_config.model, forward_only(&training))
            .inspect_err(|_| ui.fail_spinner(prepare.clone(), "student loading failed"))?;
        let format = DataFormat::infer(&corpus)?;
        let n_ctx = student.context_size()?;
        let prepared = dataset::prepare(&student, &corpus, format, n_ctx)?;
        let vocab_size = u32::try_from(student.vocab_size()?)
            .map_err(|_| Error::overflow("the student's vocabulary size does not fit in u32"))?;
        let witnesses = witness_ids(&student)?;
        (prepared, vocab_size, witnesses)
    };
    if prepared.is_empty() || prepared.supervised_tokens == 0 {
        return Err(Error::invalid(format!(
            "{}: no supervised token to score",
            corpus.display()
        )));
    }
    ui.finish_spinner(
        prepare,
        format!(
            "{} examples, {} supervised positions of {} in {}",
            prepared.examples,
            prepared.supervised_tokens,
            prepared.tokens.len(),
            HumanDuration(started.elapsed())
        ),
    );

    let load = ui.spinner("loading the teacher");
    let mut teacher = Trainer::new(&distill.teacher_path, forward_only(&training))
        .inspect_err(|_| ui.fail_spinner(load.clone(), "teacher loading failed"))?;
    // The same gate as `Teacher::compatibility`, before a single row is scored:
    // a teacher with another vocabulary produces a sidecar whose ids name other
    // tokens, and every later check would pass because they compare the file to
    // itself.
    let teacher_vocab = u32::try_from(teacher.vocab_size()?)
        .map_err(|_| Error::overflow("the teacher's vocabulary size does not fit in u32"))?;
    if teacher_vocab != vocab_size {
        return Err(Error::config(format!(
            "teacher {} has vocabulary size {teacher_vocab}, the student has {vocab_size}: the \
             two models do not share one tokenizer",
            distill.teacher_path.display()
        )));
    }
    let teacher_witnesses = witness_ids(&teacher)?;
    if teacher_witnesses != witnesses {
        return Err(Error::config(format!(
            "teacher {} tokenizes the witness sentences differently from the student: the two \
             models do not share one tokenizer",
            distill.teacher_path.display()
        )));
    }
    ui.finish_spinner(load, "teacher loaded");

    let scoring = ui.spinner("scoring the corpus");
    let (ids, logprobs) = score_corpus(&mut teacher, &prepared, k)
        .inspect_err(|_| ui.fail_spinner(scoring.clone(), "scoring failed"))?;
    let header = TopKHeader {
        version: VERSION,
        k: k as u32,
        n_rows: prepared.tokens.len() as u64,
        vocab_size,
        tokenizer_hash: tokenizer_fingerprint(vocab_size, &witnesses),
        source_hash: corpus_fingerprint(&prepared),
    };
    let sidecar = TopKSidecar::new(header, ids, logprobs)?;
    sidecar.write(&out)?;
    ui.finish_spinner(
        scoring,
        format!(
            "wrote {} in {}",
            out.display(),
            HumanDuration(started.elapsed())
        ),
    );
    Ok(())
}

/// The teacher's geometry: where and how wide the forward pass runs, and
/// nothing else. No epochs, no learning rate, no optimizer - and no adapter is
/// ever created, which is what keeps the backward graph out of existence.
fn forward_only(training: &TrainConfig) -> TrainConfig {
    TrainConfig {
        device: training.device,
        n_ctx: training.n_ctx,
        n_batch: training.n_batch,
        n_ubatch: training.n_ubatch,
        n_seq_max: training.n_seq_max,
        threads: training.threads,
        kv_dtype: training.kv_dtype,
        ..TrainConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_config_file_is_required_and_flags_take_values() {
        assert!(parse(&[]).is_err());
        let args = parse(&[
            "run.toml".into(),
            "--k".into(),
            "8".into(),
            "--out".into(),
            "corpus.topk".into(),
        ])
        .expect("parse");
        assert_eq!(args.config, PathBuf::from("run.toml"));
        assert_eq!(args.k, Some(8));
        assert_eq!(args.out, Some(PathBuf::from("corpus.topk")));
        assert!(parse(&["run.toml".into(), "--k".into()]).is_err());
        assert!(parse(&["run.toml".into(), "--nope".into()]).is_err());
        assert!(parse(&["a.toml".into(), "b.toml".into()]).is_err());
    }

    /// `k` past the operator's ceiling would write a file no run could train
    /// on, so it is refused where it is cheap to refuse.
    #[test]
    fn k_is_bounded_by_what_the_runtime_operator_accepts() {
        assert!(parse(&["run.toml".into(), "--k".into(), "0".into()]).is_err());
        let over = (MAX_K + 1).to_string();
        assert!(parse(&["run.toml".into(), "--k".into(), over]).is_err());
        assert!(parse(&["run.toml".into(), "--k".into(), MAX_K.to_string()]).is_ok());
    }
}
