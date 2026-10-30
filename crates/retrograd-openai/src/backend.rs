//! The serving backend: a [`DecodeBackend`] that can also render prompts,
//! driven by the session's scheduler on the one thread that owns the weights.
//!
//! [`Engine`](retrograd_engine::Engine) is the production implementation -
//! inference only, no optimizer, no training state - so a `Target::Cold`
//! request is served by the same step-wise engine the rollouts run on.
//! [`EngineLoader`] builds it from a [`WeightsSpec`]; the crate's tests use
//! [`crate::testing::FakeLoader`]'s fake instead.

use retrograd_core::{Result as CoreResult, TrainConfig};
use retrograd_engine::{Engine, generation_batch};
use retrograd_infer::{DecodeBackend, SchedulerOptions};

use crate::error::OpenAiError;
use crate::render::PromptModel;
use crate::request::ToolConstraint;
use crate::session::WeightsSpec;

/// A decode backend that also answers the read-only prompt-rendering queries:
/// what the session drives through the scheduler.
pub trait ServeBackend: DecodeBackend + PromptModel {
    /// Sets the engine-level reasoning budget (tokens; `-1`: the model
    /// decides). The runtime builds each request's budget machine from the
    /// engine's plan. The serving session drains requests before changing it.
    fn set_reasoning_budget(&mut self, budget: i32) -> CoreResult<()>;
    /// Whether the loaded model declares a pooling mode, i.e. can produce a
    /// sequence embedding at all. A causal chat model without one gets no
    /// embedding invented by averaging.
    fn supports_embeddings(&self) -> CoreResult<bool>;
    /// The pooled embedding of one token sequence. Prefill-only: the runtime
    /// clears the KV before and after, so it never shares KV with a
    /// generation whose attention mode differs.
    fn embed(&mut self, tokens: &[i32]) -> CoreResult<Vec<f32>>;
}

impl ServeBackend for Engine {
    fn set_reasoning_budget(&mut self, budget: i32) -> CoreResult<()> {
        Ok(Engine::set_reasoning_budget(self, budget)?)
    }

    fn supports_embeddings(&self) -> CoreResult<bool> {
        Ok(Engine::supports_embeddings(self)?)
    }

    fn embed(&mut self, tokens: &[i32]) -> CoreResult<Vec<f32>> {
        Ok(Engine::embed(self, tokens)?)
    }
}

impl PromptModel for Engine {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        Ok(Engine::chat_template_supports_tools(self)?)
    }

    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>> {
        Ok(Engine::tool_call_parser(self, Some(tools_json))?)
    }

    fn tool_call_grammar(
        &self,
        tools_json: &str,
        constraint: &ToolConstraint,
    ) -> CoreResult<Option<String>> {
        let (mode, forced) = match constraint {
            ToolConstraint::Required => (0, None),
            ToolConstraint::Forced(name) => (1, Some(name.as_str())),
            ToolConstraint::AtMostOne => (2, None),
        };
        Ok(Engine::tool_call_grammar(self, tools_json, mode, forced)?)
    }

    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String> {
        Ok(Engine::format_chat_messages(
            self,
            messages_json,
            tools_json,
            true,
        )?)
    }

    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>> {
        Ok(Engine::tokenize_text(self, text)?)
    }

    fn context_size(&self) -> CoreResult<usize> {
        // Widening u32 -> usize.
        Ok(Engine::context_size(self)? as usize)
    }
}

/// A loaded serving backend plus the scheduler options its geometry implies.
pub struct ServingBackend {
    pub backend: Box<dyn ServeBackend>,
    pub scheduler: SchedulerOptions,
}

/// Builds the serving backend from a weights spec, on the thread that will own
/// it (the backend never crosses a thread boundary, so it need not be `Send`
/// beyond the box).
pub trait Loader: Send + Sync + 'static {
    fn load(
        &self,
        spec: &WeightsSpec,
        queue: usize,
        admission: retrograd_infer::AdmissionMode,
    ) -> Result<ServingBackend, OpenAiError>;
}

/// The name a spec's adapter is registered under: its path, which is what the
/// model ids of a frontend name it by. Two specs naming the same adapter file
/// share the registered adapter - the base is not reloaded for a checkpoint
/// change.
pub fn adapter_name(adapter: &std::path::Path) -> String {
    adapter.to_string_lossy().into_owned()
}

/// Loads with a [`retrograd_engine::Engine`], configured for inference the way
/// `retrograd chat` configures it: the geometry of a run's dedicated generation
/// context, so serving is bit-identical to what a trainer would generate.
pub struct EngineLoader;

impl Loader for EngineLoader {
    fn load(
        &self,
        spec: &WeightsSpec,
        queue: usize,
        admission: retrograd_infer::AdmissionMode,
    ) -> Result<ServingBackend, OpenAiError> {
        let defaults = TrainConfig::default();
        // Generation decodes the prompt in `n_batch` chunks inside one context,
        // so the batch only has to stay within the window.
        let n_batch = defaults.n_batch.min(spec.n_ctx);
        let training = TrainConfig {
            n_ctx: spec.n_ctx,
            n_batch,
            n_ubatch: defaults.n_ubatch.min(n_batch),
            device: spec.device,
            generation_concurrency: spec.n_seq_max.unwrap_or(defaults.generation_concurrency),
            ..defaults
        };
        let mut engine = Engine::from_train_config(&spec.model, &training)?;
        if let Some(adapter) = &spec.adapter {
            // Registered under its name, not applied: the scheduler activates
            // the adapter each request asks for, so several checkpoints of this
            // base can be resident and switched without a reload.
            engine.load_lora_named(&adapter_name(adapter), adapter)?;
        }
        if let Some(variables) = &spec.chat_template_variables {
            engine.set_chat_template_variables(variables)?;
        }
        let n_vocab = engine.vocab_size()?;
        let scheduler = SchedulerOptions {
            mode: admission,
            n_batch: generation_batch(&training, n_vocab) as usize,
            queue_capacity: queue.max(1),
            ..SchedulerOptions::default()
        };
        Ok(ServingBackend {
            backend: Box::new(engine),
            scheduler,
        })
    }
}
