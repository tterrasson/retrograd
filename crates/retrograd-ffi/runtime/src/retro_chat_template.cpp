// Renders chat messages by executing the model's own Jinja
// tokenizer.chat_template instead of matching llama.cpp's fixed list of
// hand-coded template families. Any model with a compatible HF-style template
// can use this path.
#include "retro_runtime.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include "json.h"

namespace retro {

struct chat_template_state {
    jinja::program prog;
    std::string src;
    std::string bos_token;
    std::string eos_token;
    // -1 until probed: rendering twice per rollout turn to rediscover a
    // property of the template itself would be pure waste.
    int supports_tools = -1;
};

void chat_template_state_deleter::operator()(chat_template_state * state) const {
    delete state;
}

namespace {

// llama_detokenize's unparse_special=true mirrors what common_token_to_piece
// gives HF-style templates for {{ bos_token }} / {{ eos_token }}.
std::string token_to_text(const llama_vocab * vocab, llama_token token) {
    if (token == LLAMA_TOKEN_NULL) {
        return std::string();
    }
    char stack_buf[128];
    int32_t n = llama_detokenize(vocab, &token, 1, stack_buf, sizeof(stack_buf), false, true);
    if (n >= 0) {
        return std::string(stack_buf, static_cast<size_t>(n));
    }
    std::vector<char> heap_buf(static_cast<size_t>(-n));
    n = llama_detokenize(vocab, &token, 1, heap_buf.data(), static_cast<int32_t>(heap_buf.size()), false, true);
    if (n < 0) {
        return std::string();
    }
    return std::string(heap_buf.data(), static_cast<size_t>(n));
}

// Lexes and parses one template source. Keeping this independent of the model
// lets `.jinja` fixtures exercise rendering and parser derivation without a
// GGUF.
std::unique_ptr<chat_template_state> parse_chat_template(const char * source) {
    auto owned = std::unique_ptr<chat_template_state>(new chat_template_state());
    try {
        jinja::lexer lexer;
        const auto lexer_res = lexer.tokenize(source);
        owned->prog = jinja::parse_from_tokens(lexer_res);
        owned->src = lexer_res.source;
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse model chat template: ") + err.what());
        return nullptr;
    }
    return owned;
}

chat_template_state * ensure_chat_template(trainer_state & state) {
    if (state.chat_template_cache) {
        return state.chat_template_cache.get();
    }
    const char * tmpl = llama_model_chat_template(state.model.get(), nullptr);
    if (!tmpl || tmpl[0] == '\0') {
        set_error("model does not define tokenizer.chat_template");
        return nullptr;
    }
    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());

    auto owned = parse_chat_template(tmpl);
    if (!owned) {
        return nullptr;
    }
    owned->bos_token = token_to_text(vocab, llama_vocab_bos(vocab));
    owned->eos_token = token_to_text(vocab, llama_vocab_eos(vocab));

    state.chat_template_cache = chat_template_state_ptr(owned.release());
    return state.chat_template_cache.get();
}

using json = common_json;

// Executes the template over an already-built message array. `tools` is passed
// through when non-null, so a template that knows about tools sees the catalog
// where it expects it rather than in a hand-written system prompt.
bool apply_template(
        const chat_template_state & tmpl,
        const json & messages,
        const json * tools,
        bool add_assistant,
        const std::string & variables,
        std::string & rendered) {
    json inputs = json::object();
    // The configured variables go in first and the renderer's own keys are written over them.
    if (!variables.empty()) {
        try {
            const json extra = json::parse(variables);
            for (auto it = extra.begin(); it != extra.end(); ++it) {
                inputs[it.key()] = it.value();
            }
        } catch (const std::exception & err) {
            set_error(std::string("failed to parse chat template variables: ") + err.what());
            return false;
        }
    }
    inputs["messages"] = messages;
    inputs["bos_token"] = tmpl.bos_token;
    inputs["eos_token"] = tmpl.eos_token;
    inputs["add_generation_prompt"] = add_assistant;
    if (tools) {
        inputs["tools"] = *tools;
    }

    try {
        jinja::context ctx(tmpl.src);
        jinja::global_from_json(ctx, inputs, /*mark_input=*/false);
        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(tmpl.prog);
        const auto parts = jinja::runtime::gather_string_parts(results);
        rendered = parts->as_string().str();
        return true;
    } catch (const std::exception & err) {
        set_error(std::string("failed to apply model chat template: ") + err.what());
        return false;
    }
}

int copy_rendered_out(const std::string & rendered, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return render_string_out(
            buffer, n_buffer, out_n_bytes, "failed to apply model chat template",
            [&](char * destination, int32_t capacity) -> int32_t {
                const auto n = static_cast<int32_t>(std::min<size_t>(rendered.size(), static_cast<size_t>(INT32_MAX)));
                if (destination && capacity > 0) {
                    const auto to_copy = static_cast<size_t>(std::min<int32_t>(n, capacity));
                    std::memcpy(destination, rendered.data(), to_copy);
                }
                return n;
            });
}

}  // namespace

bool validate_template_variables(const char * variables_json, std::string & out_validated) {
    out_validated.clear();
    if (!variables_json || *variables_json == '\0') {
        return true;
    }
    json variables;
    try {
        variables = json::parse(variables_json);
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse chat template variables: ") + err.what());
        return false;
    }
    if (!variables.is_object()) {
        set_error("chat template variables must be a JSON object");
        return false;
    }
    // Refused, not overwritten.
    static const char * const reserved[] = {
        "messages", "tools", "bos_token", "eos_token", "add_generation_prompt",
    };
    for (const char * name : reserved) {
        if (variables.contains(name)) {
            set_error(std::string("chat template variable '") + name
                      + "' is built from the conversation and cannot be set");
            return false;
        }
    }
    // llama.cpp reads it as a flag when it derives the parser; a string would
    // be truthy in the template and ignored by the derivation.
    if (variables.contains("enable_thinking") && !variables.at("enable_thinking").is_boolean()) {
        set_error("chat template variable 'enable_thinking' must be a boolean");
        return false;
    }
    out_validated = variables.empty() ? std::string() : variables.dump();
    return true;
}

namespace {

// What the template appends for an assistant to start: one user turn rendered
// with and without the generation prompt, and the difference.
bool generation_prompt(const chat_template_state & tmpl, const json & variables, std::string & out) {
    const json messages = json::array({ json{ { "role", "user" }, { "content", "probe" } } });
    const std::string dumped = variables.empty() ? std::string() : variables.dump();
    std::string without;
    std::string with;
    if (!apply_template(tmpl, messages, nullptr, false, dumped, without)
            || !apply_template(tmpl, messages, nullptr, true, dumped, with)) {
        return false;
    }
    size_t common = 0;
    while (common < without.size() && common < with.size() && without[common] == with[common]) {
        ++common;
    }
    out = with.substr(common);
    return true;
}

bool derivation_variables(const chat_template_state & tmpl, const std::string & validated, std::string & out) {
    json variables = validated.empty() ? json::object() : json::parse(validated);
    if (!variables.contains("enable_thinking")) {
        std::string unset;
        if (!generation_prompt(tmpl, variables, unset)) {
            return false;
        }
        bool found = false;
        // True first: it is llama.cpp's default, and the answer for a template
        // that never reads the variable.
        for (const bool candidate : { true, false }) {
            json candidate_variables = variables;
            candidate_variables["enable_thinking"] = candidate;
            std::string rendered;
            if (!generation_prompt(tmpl, candidate_variables, rendered)) {
                return false;
            }
            if (rendered == unset) {
                variables["enable_thinking"] = candidate;
                found = true;
                break;
            }
        }
        if (!found) {
            set_error("with `enable_thinking` unset the chat template renders a generation prompt "
                      "that neither true nor false reproduces; set it in the template variables");
            return false;
        }
    }
    out = variables.dump();
    return true;
}

}  // namespace

bool resolve_derivation_variables(trainer_state & state, std::string & out_variables) {
    const chat_template_state * tmpl = ensure_chat_template(state);
    return tmpl && derivation_variables(*tmpl, state.chat_template_variables, out_variables);
}

bool resolve_derivation_variables_source(
        const char * template_src,
        const char * variables_json,
        std::string & out_variables) {
    std::string validated;
    if (!validate_template_variables(variables_json, validated)) {
        return false;
    }
    const auto tmpl = parse_chat_template(template_src);
    return tmpl && derivation_variables(*tmpl, validated, out_variables);
}

int set_chat_template_variables(trainer_state & state, const char * variables_json) {
    std::string validated;
    if (!validate_template_variables(variables_json, validated)) {
        return -1;
    }
    state.chat_template_variables = std::move(validated);
    // Everything derived from the variables: the tool-support probe, the
    // parser (whose generation prompt they decide) and the reasoning budget's
    // tags.
    if (state.chat_template_cache) {
        state.chat_template_cache->supports_tools = -1;
    }
    state.tool_call_parser_cache.clear();
    state.tool_call_parser_tools.clear();
    state.reasoning_plan = reasoning_budget_plan{};
    return 0;
}

int render_chat_template(
        trainer_state & state,
        const char * const * roles,
        const char * const * contents,
        size_t n_messages,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    chat_template_state * tmpl = ensure_chat_template(state);
    if (!tmpl) {
        return -1;
    }

    json messages = json::array();
    for (size_t i = 0; i < n_messages; ++i) {
        messages.push_back(json{
            { "role", roles[i] },
            { "content", contents[i] },
        });
    }

    std::string rendered;
    if (!apply_template(*tmpl, messages, nullptr, add_assistant, state.chat_template_variables, rendered)) {
        return -1;
    }
    return copy_rendered_out(rendered, buffer, n_buffer, out_n_bytes);
}

int render_chat_messages(
        trainer_state & state,
        const char * messages_json,
        const char * tools_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    chat_template_state * tmpl = ensure_chat_template(state);
    if (!tmpl) {
        return -1;
    }

    json messages;
    json tools;
    try {
        messages = json::parse(messages_json);
        if (!messages.is_array()) {
            set_error("chat messages must be a JSON array");
            return -1;
        }
        if (tools_json) {
            tools = json::parse(tools_json);
            if (!tools.is_array()) {
                set_error("chat tools must be a JSON array");
                return -1;
            }
        }
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse chat messages: ") + err.what());
        return -1;
    }

    std::string rendered;
    if (!apply_template(*tmpl, messages, tools_json ? &tools : nullptr, add_assistant,
                        state.chat_template_variables, rendered)) {
        return -1;
    }
    return copy_rendered_out(rendered, buffer, n_buffer, out_n_bytes);
}

int render_chat_template_source(
        const char * template_src,
        const char * messages_json,
        const char * tools_json,
        const char * variables_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    std::string variables;
    if (!validate_template_variables(variables_json, variables)) {
        return -1;
    }
    auto tmpl = parse_chat_template(template_src);
    if (!tmpl) {
        return -1;
    }
    // No vocabulary to draw them from: a fixture renders `{{ bos_token }}` as
    // the empty string, which is also what llama.cpp's own model-less
    // template initialization does.
    json messages;
    json tools;
    try {
        messages = json::parse(messages_json);
        if (!messages.is_array()) {
            set_error("chat messages must be a JSON array");
            return -1;
        }
        if (tools_json) {
            tools = json::parse(tools_json);
            if (!tools.is_array()) {
                set_error("chat tools must be a JSON array");
                return -1;
            }
        }
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse chat messages: ") + err.what());
        return -1;
    }

    std::string rendered;
    // No trainer here: a fixture states its own variables.
    if (!apply_template(*tmpl, messages, tools_json ? &tools : nullptr, add_assistant, variables, rendered)) {
        return -1;
    }
    return copy_rendered_out(rendered, buffer, n_buffer, out_n_bytes);
}

int chat_template_supports_tools(trainer_state & state, bool * out_supports) {
    chat_template_state * tmpl = ensure_chat_template(state);
    if (!tmpl) {
        return -1;
    }
    if (tmpl->supports_tools < 0) {
        // Probe the complete tool conversation rather than only the `tools`
        // field: templates may accept the catalog but reject the tool result
        // shape used by rollout. The two sentinels are unique to their inputs,
        // so their presence in the output confirms that both were rendered.
        static const char * catalog_sentinel = "retro_probe_tool_9d41c7";
        static const char * result_sentinel = "retro_probe_result_9d41c7";
        json tools = json::array({ json{
            { "type", "function" },
            { "function", json{
                { "name", catalog_sentinel },
                { "description", catalog_sentinel },
                { "parameters", json{ { "type", "object" }, { "properties", json::object() } } },
            } },
        } });
        json messages = json::array({
            json{ { "role", "system" }, { "content", "probe" } },
            json{ { "role", "user" }, { "content", "probe" } },
            json{ { "role", "assistant" }, { "content", "probe" } },
            json{ { "role", "tool" }, { "tool_call_id", "retro_probe_call_0" }, { "content", result_sentinel } },
        });
        std::string rendered;
        // A template that throws on any of that does not support tools; the probe
        // must not turn the refusal into a hard error for the caller.
        const bool rendered_ok = apply_template(*tmpl, messages, &tools, /*add_assistant=*/true,
                                                state.chat_template_variables, rendered);
        const bool renders_tools = rendered_ok
                && rendered.find(catalog_sentinel) != std::string::npos
                && rendered.find(result_sentinel) != std::string::npos;
        tmpl->supports_tools = renders_tools ? 1 : 0;
        if (!rendered_ok) {
            clear_error();
        }
    }
    *out_supports = tmpl->supports_tools == 1;
    return 0;
}

}  // namespace retro
