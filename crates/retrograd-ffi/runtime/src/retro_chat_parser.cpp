// Reads an assistant turn with a parser derived from the model's chat template.
// llama.cpp supports specialized and differential derivation; both produce a
// serialized PEG parser that `common_chat_parse` can run without a model or
// context. The model-backed and model-free entry points therefore share the
// same parser format.
#include "retro_runtime.hpp"

#include "chat.h"

namespace retro {

namespace {

using json = common_json;

// The envelope carries what `common_chat_parse` needs beside the arena itself:
// the format selects the AST mapper, and `generation_prompt` is prepended to
// the input before parsing because generated parsers anchor on it (LFM2's, for
// one, opens on `p.literal("<|im_start|>assistant\n")`). Serializing the arena
// alone would produce a parser that rejects every input.
std::string encode_parser_blob(const common_chat_params & params) {
    json envelope = json::object();
    envelope["format"] = static_cast<int>(params.format);
    envelope["generation_prompt"] = params.generation_prompt;
    envelope["parser"] = params.parser;
    return envelope.dump();
}

bool decode_parser_blob(const std::string & blob, common_chat_parser_params & out) {
    json envelope;
    try {
        envelope = json::parse(blob);
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse tool-call parser blob: ") + err.what());
        return false;
    }
    if (!envelope.is_object() || !envelope.contains("parser")
            || !envelope.at("parser").is_string()) {
        set_error("tool-call parser blob must be an object carrying a `parser` string");
        return false;
    }
    if (envelope.contains("format") && envelope.at("format").is_number_integer()) {
        out.format = static_cast<common_chat_format>(envelope.at("format").get<int>());
    }
    if (envelope.contains("generation_prompt") && envelope.at("generation_prompt").is_string()) {
        out.generation_prompt = envelope.at("generation_prompt").get<std::string>();
    }
    try {
        out.parser.load(envelope.at("parser").get<std::string>());
    } catch (const std::exception & err) {
        set_error(std::string("failed to load tool-call PEG parser: ") + err.what());
        return false;
    }
    return true;
}

// These parser entry points return JSON through the same two-call buffer
// contract as the chat renderers.
int copy_json_out(const std::string & text, const char * error, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return render_string_out(
            buffer, n_buffer, out_n_bytes, error,
            [&](char * destination, int32_t capacity) -> int32_t {
                const auto n = static_cast<int32_t>(std::min<size_t>(text.size(), static_cast<size_t>(INT32_MAX)));
                if (destination && capacity > 0) {
                    const auto to_copy = static_cast<size_t>(std::min<int32_t>(n, capacity));
                    std::memcpy(destination, text.data(), to_copy);
                }
                return n;
            });
}

// Derives a parser from an initialized template. The template may come from a
// loaded model or from `.jinja` source supplied by a test.
enum class parser_derivation {
    ok,
    unavailable,
    error,
};

parser_derivation derive_parser_blob(
        const common_chat_templates * templates,
        const char * tools_json,
        std::string & blob) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.parallel_tool_calls = true;
    // The sampled text is what gets trained, so nothing may be moved out of it
    // into a separate `reasoning_content` field: a template's own thinking tags
    // stay part of the content the trajectory replays.
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    // One user turn is the minimum the generation-prompt probe needs: it
    // renders the conversation with and without the assistant prefix and keeps
    // the difference.
    common_chat_msg user;
    user.role = "user";
    user.content = "probe";
    inputs.messages.push_back(user);

    if (tools_json && tools_json[0] != '\0') {
        json tools;
        try {
            tools = json::parse(tools_json);
        } catch (const std::exception & err) {
            set_error(std::string("failed to parse tool catalog: ") + err.what());
            return parser_derivation::error;
        }
        if (!tools.is_array()) {
            set_error("tool catalog must be a JSON array");
            return parser_derivation::error;
        }
        if (!tools.empty()) {
            inputs.tools = common_chat_tools_parse_oaicompat(tools);
        }
    }

    common_chat_params params;
    try {
        params = common_chat_templates_apply(templates, inputs);
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::exception & err) {
        // An unparsable template is an explicit unsupported result; callers may
        // use the prompt-described convention. Other runtime failures remain
        // ordinary errors.
        set_error(std::string("failed to derive a tool-call parser from the chat template: ") + err.what());
        return parser_derivation::unavailable;
    }
    if (params.parser.empty()) {
        set_error("chat template yielded no tool-call parser");
        return parser_derivation::unavailable;
    }
    blob = encode_parser_blob(params);
    return parser_derivation::ok;
}

}  // namespace

int build_tool_call_parser(
        trainer_state & state,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    // Key the cache by the tool catalog because the generated grammar may name
    // functions. This avoids re-deriving the parser for every frontend call.
    const std::string tools_key = tools_json ? std::string(tools_json) : std::string();
    if (!state.tool_call_parser_cache.empty() && state.tool_call_parser_tools == tools_key) {
        return copy_json_out(
                state.tool_call_parser_cache, "failed to build tool-call parser",
                buffer, n_buffer, out_n_bytes);
    }

    common_chat_templates_ptr templates;
    try {
        templates = common_chat_templates_init(state.model.get(), "");
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::exception & err) {
        set_error(std::string("failed to initialize the model chat template: ") + err.what());
        return -1;
    }
    if (!templates) {
        set_error("model does not define a chat template usable for tool-call parsing");
        return RETRO_CHAT_PARSER_UNAVAILABLE;
    }
    std::string blob;
    switch (derive_parser_blob(templates.get(), tools_json, blob)) {
        case parser_derivation::ok:
            break;
        case parser_derivation::unavailable:
            return RETRO_CHAT_PARSER_UNAVAILABLE;
        case parser_derivation::error:
            return -1;
    }

    state.tool_call_parser_cache = std::move(blob);
    state.tool_call_parser_tools = tools_key;
    return copy_json_out(
            state.tool_call_parser_cache, "failed to build tool-call parser",
            buffer, n_buffer, out_n_bytes);
}

int build_tool_call_parser_from_source(
        const char * template_src,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    common_chat_templates_ptr templates;
    try {
        // A null model is legal precisely when the source is given explicitly;
        // bos/eos then render as empty, which is what a fixture wants.
        templates = common_chat_templates_init(nullptr, template_src);
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::exception & err) {
        set_error(std::string("failed to initialize the chat template: ") + err.what());
        return -1;
    }
    if (!templates) {
        set_error("failed to initialize the chat template");
        return -1;
    }
    std::string blob;
    switch (derive_parser_blob(templates.get(), tools_json, blob)) {
        case parser_derivation::ok:
            break;
        case parser_derivation::unavailable:
            return RETRO_CHAT_PARSER_UNAVAILABLE;
        case parser_derivation::error:
            return -1;
    }
    return copy_json_out(blob, "failed to build tool-call parser", buffer, n_buffer, out_n_bytes);
}

int parse_assistant_output(
        const char * parser_blob,
        size_t n_parser_blob,
        const char * text,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    common_chat_parser_params params;
    if (!decode_parser_blob(std::string(parser_blob, n_parser_blob), params)) {
        return -1;
    }
    params.reasoning_format = COMMON_REASONING_FORMAT_NONE;

    common_chat_msg message;
    try {
        message = common_chat_parse(text, /*is_partial=*/false, params);
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse assistant output: ") + err.what());
        return -1;
    }

    json out = json::object();
    out["content"] = message.content;
    out["reasoning_content"] = message.reasoning_content;
    json calls = json::array();
    for (const auto & call : message.tool_calls) {
        // `arguments` stays the string llama.cpp produced. Turning it into a
        // value here would hide a malformed one behind a C++ exception, where
        // the Rust side can hand it back to the policy as an observation.
        calls.push_back(json{
            { "id", call.id },
            { "name", call.name },
            { "arguments", call.arguments },
        });
    }
    out["tool_calls"] = calls;

    return copy_json_out(out.dump(), "failed to render parsed assistant output", buffer, n_buffer, out_n_bytes);
}

}  // namespace retro
