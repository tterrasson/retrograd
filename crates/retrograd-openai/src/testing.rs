//! A model that needs no GGUF, for the tests of every crate that serves this
//! contract.
//!
//! Byte-level: a token is a byte, the reply is fixed, and end of generation is
//! the token `-1`. That is enough to drive every branch between the request and
//! the response - rendering, context bound, stop, finish reason, parsing -
//! without a device.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use retrograd_core::{Result as CoreResult, SamplingParams};

use crate::error::OpenAiError;
use crate::render::ChatModel;
use crate::session::{Loader, WeightsSpec};

/// The end-of-generation token of [`FakeModel`].
pub const EOG: i32 = -1;

/// Answers every prompt with the same reply, byte for byte.
pub struct FakeModel {
    reply: String,
    context: usize,
    samples: usize,
    /// What the last `format_chat` was handed, so a test can read the prompt
    /// the model was actually given.
    last: Option<(String, Option<String>)>,
}

impl FakeModel {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            context: 1 << 16,
            samples: 0,
            last: None,
        }
    }

    pub fn with_context(mut self, context: usize) -> Self {
        self.context = context;
        self
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn last_messages(&self) -> Option<String> {
        self.last.as_ref().map(|(messages, _)| messages.clone())
    }

    pub fn last_tools(&self) -> Option<String> {
        self.last.as_ref().and_then(|(_, tools)| tools.clone())
    }
}

impl ChatModel for FakeModel {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        Ok(false)
    }

    fn tool_call_parser(&self, _tools_json: &str) -> CoreResult<Option<String>> {
        Ok(None)
    }

    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String> {
        // `&self`, like the trainer's: the prompt comes back to `sample` as
        // tokens, and that is where it is recorded.
        Ok(format!(
            "{messages_json}\u{1}{}",
            tools_json.unwrap_or_default()
        ))
    }

    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>> {
        Ok(text.bytes().map(i32::from).collect())
    }

    fn context_size(&self) -> CoreResult<usize> {
        Ok(self.context)
    }

    fn sample(
        &mut self,
        prompt: &[i32],
        sampling: &SamplingParams,
        base: bool,
    ) -> CoreResult<Vec<i32>> {
        self.samples += 1;
        let rendered = prompt
            .iter()
            .map(|&token| u8::try_from(token).expect("the fake tokenizer emits bytes"))
            .collect::<Vec<_>>();
        let rendered = String::from_utf8(rendered).expect("the fake tokenizer emits UTF-8");
        let (messages, tools) = rendered
            .split_once('\u{1}')
            .expect("the fake template separates messages and tools");
        self.last = Some((
            messages.to_owned(),
            (!tools.is_empty()).then(|| tools.to_owned()),
        ));
        let reply = match base {
            true => format!("base: {}", self.reply),
            false => self.reply.clone(),
        };
        let budget = sampling.max_new_tokens as usize;
        let mut tokens = reply
            .bytes()
            .map(i32::from)
            .take(budget)
            .collect::<Vec<_>>();
        if tokens.len() < budget {
            tokens.push(EOG);
        }
        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[i32], _unparse_special: bool) -> CoreResult<String> {
        let bytes = tokens
            .iter()
            .filter(|&&token| token != EOG)
            .map(|&token| u8::try_from(token).expect("the fake tokenizer emits bytes"))
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn is_eog(&self, token: i32) -> CoreResult<bool> {
        Ok(token == EOG)
    }
}

/// Loads a [`FakeModel`] for any spec, and remembers what it was asked to load.
#[derive(Clone)]
pub struct FakeLoader {
    reply: String,
    loads: Arc<AtomicUsize>,
    specs: Arc<Mutex<Vec<WeightsSpec>>>,
}

impl FakeLoader {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            loads: Arc::default(),
            specs: Arc::default(),
        }
    }

    pub fn loads(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }

    pub fn specs(&self) -> Vec<WeightsSpec> {
        self.specs.lock().expect("fake loader lock").clone()
    }
}

impl Default for FakeLoader {
    fn default() -> Self {
        Self::new("hello from retrograd")
    }
}

impl Loader for FakeLoader {
    fn load(&self, spec: &WeightsSpec) -> Result<Box<dyn ChatModel>, OpenAiError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.specs
            .lock()
            .expect("fake loader lock")
            .push(spec.clone());
        Ok(Box::new(FakeModel::new(self.reply.clone())))
    }
}
