use super::*;

impl Trainer {
    /// Renders tokens back to text.
    ///
    /// `unparse_special` is the choice between two readers: `false` drops
    /// control tokens (`<|im_start|>` and the like) for text a person reads,
    /// `true` keeps them for text a tool-call parser reads - a model's own
    /// call-format delimiters, such as LFM2's `<|tool_call_start|>`, are
    /// control tokens too, and a parser derived from the model's template
    /// needs them in the string or it never matches anything.
    ///
    /// Sampling can stop mid-character - a rollout cut at `max_tokens` in the
    /// middle of an emoji leaves a truncated multi-byte sequence - so the
    /// bytes are decoded leniently instead of failing the whole update.
    pub fn detokenize(&self, tokens: &[i32], unparse_special: bool) -> Result<String> {
        // Detokenization runs once per rollout and its size probe would render
        // every token piece a second time, so guess generously and retry only
        // when the runtime reports the exact size.
        let size_hint = tokens.len().saturating_mul(16).max(64);
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        read_string_sized_lossy(size_hint, |buffer, n_buffer, out| unsafe {
            ffi::retro_trainer_detokenize(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                unparse_special,
                buffer,
                n_buffer,
                out,
            )
        })
    }

    /// Tokenizes a complete text, prepending BOS. Compare
    /// [`Trainer::tokenize_fragment`] for a piece meant to be appended to an
    /// existing stream.
    pub fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenize(text, false)
    }

    /// Tokenizes a fragment that continues an existing token stream: no BOS is
    /// prepended, so concatenating fragments yields the stream the whole text
    /// would have produced from the second token on.
    ///
    /// An empty fragment tokenizes to nothing rather than erroring, because a
    /// chat template legitimately puts nothing between two turns.
    pub fn tokenize_fragment(&self, text: &str) -> Result<Vec<i32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.tokenize(text, true)
    }

    fn tokenize(&self, text: &str, fragment: bool) -> Result<Vec<i32>> {
        let tokenize = if fragment {
            ffi::retro_trainer_tokenize_fragment
        } else {
            ffi::retro_trainer_tokenize_text
        };
        let text = CString::new(text).map_err(nul_error)?;
        let mut needed = 0_usize;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        let code = unsafe {
            tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        if code != 0 {
            return Err(runtime_error());
        }
        if needed == 0 {
            return Ok(Vec::new());
        }

        ffi_out_vec(needed, |tokens_out| {
            let mut written = 0_usize;
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            self.check(unsafe {
                tokenize(
                    self.raw.as_ptr(),
                    text.as_ptr(),
                    tokens_out,
                    needed,
                    &mut written,
                )
            })?;
            Ok(written)
        })
    }

    pub fn eos_token(&self) -> Result<i32> {
        let mut token = 0;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_eos_token(self.raw.as_ptr(), &mut token) })?;
        Ok(token)
    }

    /// Number of token ids accepted by the loaded model vocabulary.
    pub fn vocab_size(&self) -> Result<usize> {
        let mut size = 0;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_vocab_size(self.raw.as_ptr(), &mut size) })?;
        Ok(size as usize)
    }

    /// Whether the token ends generation. Generation stops on any
    /// end-of-generation token, so a caller cannot infer a natural stop from
    /// `eos_token` alone - templates like ChatML end turns on their own marker.
    pub fn is_eog_token(&self, token: i32) -> Result<bool> {
        let mut is_eog = false;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_is_eog_token(self.raw.as_ptr(), token, &mut is_eog)
        })?;
        Ok(is_eog)
    }

    /// Effective context allocated by llama.cpp (which may round the request).
    pub fn context_size(&self) -> Result<usize> {
        let mut n_ctx = 0;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_context_size(self.raw.as_ptr(), &mut n_ctx) })?;
        Ok(n_ctx as usize)
    }

    /// Formats messages with the GGUF model's declared chat template.
    pub fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String> {
        if messages.is_empty() && !add_assistant {
            return Err(Error::invalid("chat messages must not be empty"));
        }
        let roles = messages
            .iter()
            .map(|(role, _)| CString::new(*role).map_err(nul_error))
            .collect::<Result<Vec<_>>>()?;
        let contents = messages
            .iter()
            .map(|(_, content)| CString::new(*content).map_err(nul_error))
            .collect::<Result<Vec<_>>>()?;
        let role_ptrs = roles.iter().map(|value| value.as_ptr()).collect::<Vec<_>>();
        let content_ptrs = contents
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        read_string(|buffer, n_buffer, out| unsafe {
            ffi::retro_trainer_format_chat(
                self.raw.as_ptr(),
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len(),
                add_assistant,
                buffer,
                n_buffer,
                out,
            )
        })
    }

    /// Formats messages given as a JSON array, so a message can carry the
    /// structured fields its chat template expects - `tool_call_id`, `name`,
    /// `tool_calls` - instead of only `(role, content)`, and hands the tool
    /// catalog to the template as `tools` (OpenAI function shape).
    ///
    /// For a model whose template renders tools natively this is the difference
    /// between training on the format it was pre-trained with and training on a
    /// hand-written approximation of it. Check
    /// [`Trainer::chat_template_supports_tools`] before passing a catalog: a
    /// template that ignores `tools` renders the same text either way, and the
    /// caller still has to describe the tools itself.
    pub fn format_chat_messages(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        add_assistant: bool,
    ) -> Result<String> {
        let messages = CString::new(messages_json).map_err(nul_error)?;
        let tools = tools_json
            .map(|tools| CString::new(tools).map_err(nul_error))
            .transpose()?;
        let tools_ptr = tools
            .as_ref()
            .map_or(std::ptr::null(), |tools| tools.as_ptr());
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        read_string(|buffer, n_buffer, out| unsafe {
            ffi::retro_trainer_format_chat_messages(
                self.raw.as_ptr(),
                messages.as_ptr(),
                tools_ptr,
                add_assistant,
                buffer,
                n_buffer,
                out,
            )
        })
    }

    /// Whether the model's own chat template renders a tool catalog. Probed once
    /// by the runtime with a sentinel tool, so a template that mentions `tools`
    /// and then drops it answers `false` - the question is only ever whether the
    /// catalog reaches the rendered text.
    pub fn chat_template_supports_tools(&self) -> Result<bool> {
        let mut supports = false;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_chat_template_supports_tools(self.raw.as_ptr(), &mut supports)
        })?;
        Ok(supports)
    }

    /// Derives, from the model's own chat template, the parser for the tool-call
    /// format that template teaches the model - the reading half of
    /// [`Trainer::format_chat_messages`].
    ///
    /// The result is a serialized parser that carries no reference to the model,
    /// which is what lets [`parse_assistant_output`] run it off the trainer's
    /// thread. It is only valid for `tools_json`: the generated grammar may name
    /// the functions.
    ///
    /// `Ok(None)` when the template yields no parser. That is a nominal outcome
    /// for an exotic template, not a broken model, and the caller answers it by
    /// describing the tools in the prompt and reading the answer by convention.
    pub fn tool_call_parser(&self, tools_json: Option<&str>) -> Result<Option<String>> {
        let tools = tools_json
            .map(|tools| CString::new(tools).map_err(nul_error))
            .transpose()?;
        let tools_ptr = tools
            .as_ref()
            .map_or(std::ptr::null(), |tools| tools.as_ptr());
        let parser = read_optional_string(
            ffi::RETRO_CHAT_PARSER_UNAVAILABLE,
            |buffer, n_buffer, out| {
                // SAFETY: the `Trainer` invariant holds and all borrowed arguments
                // live through this synchronous call.
                unsafe {
                    ffi::retro_trainer_tool_call_parser(
                        self.raw.as_ptr(),
                        tools_ptr,
                        buffer,
                        n_buffer,
                        out,
                    )
                }
            },
        )?;
        if parser.is_none() {
            tracing::warn!(
                "the chat template yields no tool-call parser; falling back to prompt rendering"
            );
        }
        Ok(parser)
    }
}

/// Runs a two-call string entry whose one documented status means "no value".
/// Every other non-zero status remains an error; collapsing those together is
/// so malformed catalogs and runtime failures remain errors rather than
/// becoming a fallback.
fn read_optional_string<F>(unavailable: i32, mut call: F) -> Result<Option<String>>
where
    F: FnMut(*mut std::ffi::c_char, usize, *mut usize) -> i32,
{
    let mut needed = 0_usize;
    match call(std::ptr::null_mut(), 0, &mut needed) {
        code if code == unavailable => Ok(None),
        0 => Ok(Some(string_from_runtime(read_bytes_exact(
            needed, &mut call,
        )?)?)),
        _ => Err(runtime_error()),
    }
}

/// Renders messages through a chat template given as Jinja source rather than
/// read from a model.
///
/// Same engine as [`Trainer::format_chat_messages`], minus the vocabulary:
/// `{{ bos_token }}` and `{{ eos_token }}` render empty. It exists so the
/// agreement between what a template writes and what
/// [`tool_call_parser_from_source`] reads back can be checked on template
/// fixtures, without a GGUF and therefore inside the fast lane.
pub fn render_chat_template_source(
    template_src: &str,
    messages_json: &str,
    tools_json: Option<&str>,
    add_assistant: bool,
) -> Result<String> {
    let template = CString::new(template_src).map_err(nul_error)?;
    let messages = CString::new(messages_json).map_err(nul_error)?;
    let tools = tools_json
        .map(|tools| CString::new(tools).map_err(nul_error))
        .transpose()?;
    let tools_ptr = tools
        .as_ref()
        .map_or(std::ptr::null(), |tools| tools.as_ptr());
    // SAFETY: every borrowed buffer lives through this synchronous call, and
    // the entry opens no model.
    read_string(|buffer, n_buffer, out| unsafe {
        ffi::retro_chat_template_render(
            template.as_ptr(),
            messages.as_ptr(),
            tools_ptr,
            add_assistant,
            buffer,
            n_buffer,
            out,
        )
    })
}

/// [`Trainer::tool_call_parser`] over a template given as Jinja source. Unlike
/// the method, a template that yields no parser is an `Err` here: a fixture that
/// stopped being parseable is a result to read, not a fallback to take.
pub fn tool_call_parser_from_source(
    template_src: &str,
    tools_json: Option<&str>,
) -> Result<String> {
    let template = CString::new(template_src).map_err(nul_error)?;
    let tools = tools_json
        .map(|tools| CString::new(tools).map_err(nul_error))
        .transpose()?;
    let tools_ptr = tools
        .as_ref()
        .map_or(std::ptr::null(), |tools| tools.as_ptr());
    // SAFETY: every borrowed buffer lives through this synchronous call, and
    // the entry opens no model.
    read_string(|buffer, n_buffer, out| unsafe {
        ffi::retro_chat_template_tool_call_parser(
            template.as_ptr(),
            tools_ptr,
            buffer,
            n_buffer,
            out,
        )
    })
}

/// Runs a parser from [`Trainer::tool_call_parser`] over one assistant output.
///
/// A free function rather than a method because it is one: the parser blob is
/// self-sufficient, so this needs no trainer, no model and no context, and may
/// be called concurrently with generation.
///
/// Returns the JSON document the runtime produces:
/// `{"content", "reasoning_content", "tool_calls": [{"id", "name", "arguments"}]}`,
/// with `arguments` left as the string llama.cpp emitted.
pub fn parse_assistant_output(parser: &str, text: &str) -> Result<String> {
    // Sized rather than probed: the size probe would run the PEG parser a
    // second time over the same output, and this is called once per turn per
    // rollout member. The JSON adds the field names and the escaping around a
    // document whose bulk is the text itself.
    let size_hint = text.len().saturating_mul(2).saturating_add(256);
    let text = CString::new(text).map_err(nul_error)?;
    // SAFETY: both borrowed buffers live through this synchronous call, and the
    // entry takes no trainer to invalidate.
    read_string_sized_lossy(size_hint, |buffer, n_buffer, out| unsafe {
        ffi::retro_chat_parse_assistant(
            parser.as_ptr().cast(),
            parser.len(),
            text.as_ptr(),
            buffer,
            n_buffer,
            out,
        )
    })
}

#[cfg(test)]
mod optional_string_tests {
    use super::*;

    #[test]
    fn only_the_unavailable_status_becomes_none() {
        assert!(
            read_optional_string(-3, |_buffer, _capacity, _needed| -3)
                .expect("unavailable is nominal")
                .is_none()
        );
        assert!(read_optional_string(-3, |_buffer, _capacity, _needed| -1).is_err());
    }
}
