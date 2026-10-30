//! The engine's model-backed chat-template operations: the same contracts as
//! the trainer's counterparts ([`Trainer::format_chat`](crate::Trainer::format_chat) and friends), over the
//! engine's own FFI entry points. The engine's chat state is built from the
//! same GGUF metadata, so a model renders and parses identically through either
//! handle.

use std::ffi::CString;

use super::*;
use crate::{nul_error, read_string, trainer::read_optional_string};

impl Engine {
    /// Formats messages with the GGUF model's declared chat template; see
    /// [`Trainer::format_chat`](crate::Trainer::format_chat) for the contract.
    pub fn format_chat(
        &self,
        messages: &[(&str, &str)],
        add_assistant: bool,
    ) -> std::result::Result<String, EngineError> {
        if messages.is_empty() && !add_assistant {
            return Err(EngineError::invalid("chat messages must not be empty"));
        }
        let roles = messages
            .iter()
            .map(|(role, _)| CString::new(*role).map_err(nul_error))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let contents = messages
            .iter()
            .map(|(_, content)| CString::new(*content).map_err(nul_error))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let role_ptrs = roles.iter().map(|value| value.as_ptr()).collect::<Vec<_>>();
        let content_ptrs = contents
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        Ok(unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_engine_format_chat(
                    self.raw.as_ptr(),
                    role_ptrs.as_ptr(),
                    content_ptrs.as_ptr(),
                    messages.len(),
                    add_assistant,
                    buffer,
                    n_buffer,
                    out,
                )
            })?
        })
    }

    /// Formats messages given as a JSON array with an optional tool catalog;
    /// see [`Trainer::format_chat_messages`](crate::Trainer::format_chat_messages) for the contract.
    pub fn format_chat_messages(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        add_assistant: bool,
    ) -> std::result::Result<String, EngineError> {
        let messages = CString::new(messages_json).map_err(nul_error)?;
        let tools = tools_json
            .map(|tools| CString::new(tools).map_err(nul_error))
            .transpose()?;
        let tools_ptr = tools
            .as_ref()
            .map_or(std::ptr::null(), |tools| tools.as_ptr());
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        Ok(unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_engine_format_chat_messages(
                    self.raw.as_ptr(),
                    messages.as_ptr(),
                    tools_ptr,
                    add_assistant,
                    buffer,
                    n_buffer,
                    out,
                )
            })?
        })
    }

    /// Whether the model's own chat template renders a tool catalog; see
    /// [`Trainer::chat_template_supports_tools`](crate::Trainer::chat_template_supports_tools).
    pub fn chat_template_supports_tools(&self) -> std::result::Result<bool, EngineError> {
        let mut supports = false;
        // SAFETY: the `Engine` invariant holds and `supports` is a valid out-pointer.
        self.check(unsafe {
            ffi::retro_engine_chat_template_supports_tools(self.raw.as_ptr(), &mut supports)
        })?;
        Ok(supports)
    }

    /// Derives the tool-call parser from the model's own chat template; see
    /// [`Trainer::tool_call_parser`](crate::Trainer::tool_call_parser) for the contract, including the
    /// `Ok(None)` outcome when the template yields no parser.
    pub fn tool_call_parser(
        &self,
        tools_json: Option<&str>,
    ) -> std::result::Result<Option<String>, EngineError> {
        let tools = tools_json
            .map(|tools| CString::new(tools).map_err(nul_error))
            .transpose()?;
        let tools_ptr = tools
            .as_ref()
            .map_or(std::ptr::null(), |tools| tools.as_ptr());
        // SAFETY: the `Engine` invariant holds, all borrowed arguments live
        // through this synchronous call, and the entry point writes the byte
        // count it reports.
        let parser = unsafe {
            read_optional_string(
                ffi::RETRO_CHAT_PARSER_UNAVAILABLE,
                |buffer, n_buffer, out| {
                    ffi::retro_engine_tool_call_parser(
                        self.raw.as_ptr(),
                        tools_ptr,
                        buffer,
                        n_buffer,
                        out,
                    )
                },
            )?
        };
        if parser.is_none() {
            tracing::warn!(
                "the chat template yields no tool-call parser; falling back to prompt rendering"
            );
        }
        Ok(parser)
    }

    /// Derives, from the model's own chat template, the GBNF grammar that
    /// constrains decoding to the requested tool-calling mode: at least one
    /// call (`mode` 0, `tool_choice: "required"`), exactly the forced function
    /// named `forced_name` (`mode` 1), or at most one call (`mode` 2,
    /// `parallel_tool_calls: false`). The grammar names the catalog's
    /// functions, so it is keyed by `tools_json` exactly like the parser.
    ///
    /// `Ok(None)` when the template cannot express the mode - the caller
    /// refuses the request rather than sampling unconstrained.
    pub fn tool_call_grammar(
        &self,
        tools_json: &str,
        mode: i32,
        forced_name: Option<&str>,
    ) -> std::result::Result<Option<String>, EngineError> {
        let tools = CString::new(tools_json).map_err(nul_error)?;
        let forced = forced_name
            .map(|name| CString::new(name).map_err(nul_error))
            .transpose()?;
        let forced_ptr = forced
            .as_ref()
            .map_or(std::ptr::null(), |name| name.as_ptr());
        // SAFETY: the `Engine` invariant holds, all borrowed arguments live
        // through this synchronous call, and the entry point writes the byte
        // count it reports.
        Ok(unsafe {
            read_optional_string(
                ffi::RETRO_TOOL_GRAMMAR_UNAVAILABLE,
                |buffer, n_buffer, out| {
                    ffi::retro_engine_tool_call_grammar(
                        self.raw.as_ptr(),
                        tools.as_ptr(),
                        mode,
                        forced_ptr,
                        buffer,
                        n_buffer,
                        out,
                    )
                },
            )?
        })
    }
}
