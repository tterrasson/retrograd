// Reads an assistant turn with a parser derived from the model's chat template.
// llama.cpp supports specialized and differential derivation; both produce a
// PEG parser that `common_chat_parse` can run without a model or context.
// llama.cpp keeps no serialized form of that parser, so what crosses the FFI is
// the recipe it is derived from, and the model-free entry point derives it
// again (once per recipe, see `cached_parser`). The model-backed and
// model-free entry points therefore share the same blob format.
#include "retro_runtime.hpp"

#include "chat.h"

#include <cctype>
#include <mutex>
#include <unordered_map>

namespace retro {

namespace {

using json = common_json;

// Everything the derivation reads: the template source the tool catalog
// selects, the bos/eos text it renders, the catalog itself, since the
// generated grammar may name its functions, and the template variables, since
// they decide the generation prompt the parser starts from. The derivation is
// deterministic, so the same recipe always yields the parser it was validated
// with.
struct parser_recipe {
    std::string template_src;
    std::string bos_token;
    std::string eos_token;
    std::string tools_json;
    std::string variables_json;
};

std::string encode_parser_blob(const parser_recipe & recipe) {
    json envelope = json::object();
    envelope["template"] = recipe.template_src;
    envelope["bos_token"] = recipe.bos_token;
    envelope["eos_token"] = recipe.eos_token;
    envelope["tools"] = recipe.tools_json;
    envelope["variables"] = recipe.variables_json;
    return envelope.dump();
}

bool decode_parser_blob(const std::string & blob, parser_recipe & out) {
    json envelope;
    try {
        envelope = json::parse(blob);
    } catch (const std::exception & err) {
        set_error(std::string("failed to parse tool-call parser blob: ") + err.what());
        return false;
    }
    if (!envelope.is_object()) {
        set_error("tool-call parser blob must be a JSON object");
        return false;
    }
    const auto read = [&](const char * key, std::string & field) {
        if (!envelope.contains(key) || !envelope.at(key).is_string()) {
            set_error(std::string("tool-call parser blob must carry a `") + key + "` string");
            return false;
        }
        field = envelope.at(key).get<std::string>();
        return true;
    };
    return read("template", out.template_src) && read("bos_token", out.bos_token)
        && read("eos_token", out.eos_token) && read("tools", out.tools_json)
        && read("variables", out.variables_json);
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
// loaded model or from `.jinja` source supplied by a test. `template_src` is
// set to the variant the catalog selects, the one a recipe has to carry.
enum class parser_derivation {
    ok,
    unavailable,
    error,
};

// The inputs of a derivation that do not depend on the catalog: one user turn
// and the generation prompt, rendered under the same variables as the prompts
// the parser will read the continuations of. `variables_json` is the resolved
// object, `enable_thinking` included (see resolve_derivation_variables).
bool derivation_inputs(const std::string & variables_json, common_chat_templates_inputs & inputs) {
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.parallel_tool_calls = true;
    // llama.cpp splits the reasoning out, with the tags its own analysis of the
    // template found, so no tag is spelled here. Training never reads this
    // text back: it trains the sampled tokens (see Policy::render_chat_framing).
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    // One user turn is the minimum the generation-prompt probe needs: it
    // renders the conversation with and without the assistant prefix and keeps
    // the difference.
    common_chat_msg user;
    user.role = "user";
    user.content = "probe";
    inputs.messages.push_back(user);

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
    // llama.cpp hands `enable_thinking` to the template from its own flag and
    // then lays the kwargs over it; both are set so neither can disagree, and
    // the template sees the value retrograd's own renders resolved to.
    if (variables.contains("enable_thinking") && variables.at("enable_thinking").is_boolean()) {
        inputs.enable_thinking = variables.at("enable_thinking").get<bool>();
    }
    for (auto it = variables.begin(); it != variables.end(); ++it) {
        inputs.chat_template_kwargs[it.key()] = it.value().dump();
    }
    return true;
}

parser_derivation derive_parser(
        const common_chat_templates * templates,
        const std::string & tools_json,
        const std::string & variables_json,
        common_chat_params & params,
        std::string & template_src) {
    common_chat_templates_inputs inputs;
    if (!derivation_inputs(variables_json, inputs)) {
        return parser_derivation::error;
    }

    if (!tools_json.empty()) {
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
    // The selection common_chat_templates_apply makes: the tool-use variant
    // when there is one and the catalog is not empty.
    template_src = inputs.tools.empty() ? std::string() : common_chat_templates_source(templates, "tool_use");
    if (template_src.empty()) {
        template_src = common_chat_templates_source(templates);
    }
    return parser_derivation::ok;
}

// Derives the parser of a recipe once per process. Rollouts parse every turn
// with the same few recipes; deriving renders the template several times. The
// derivation runs outside the lock: two threads missing the same recipe both
// derive it and the second insert is a no-op. The bound only guards against a
// caller minting recipes without end.
constexpr size_t parser_cache_capacity = 64;

std::shared_ptr<const common_chat_parser_params> cached_parser(const std::string & blob) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<const common_chat_parser_params>> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = cache.find(blob);
        if (it != cache.end()) {
            return it->second;
        }
    }

    parser_recipe recipe;
    if (!decode_parser_blob(blob, recipe)) {
        return nullptr;
    }
    common_chat_templates_ptr templates;
    try {
        templates = common_chat_templates_init(nullptr, recipe.template_src, recipe.bos_token, recipe.eos_token);
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::exception & err) {
        set_error(std::string("failed to initialize the chat template of a tool-call parser: ") + err.what());
        return nullptr;
    }
    if (!templates) {
        set_error("failed to initialize the chat template of a tool-call parser");
        return nullptr;
    }
    common_chat_params params;
    std::string template_src;
    if (derive_parser(templates.get(), recipe.tools_json, recipe.variables_json, params, template_src)
            != parser_derivation::ok) {
        return nullptr;
    }
    auto parser = std::make_shared<common_chat_parser_params>(params);
    // The converting constructor copies the format and the generation prompt,
    // not the arena.
    parser->parser = std::move(params.parser);

    std::lock_guard<std::mutex> lock(mutex);
    if (cache.size() >= parser_cache_capacity) {
        cache.clear();
    }
    return cache.emplace(blob, std::move(parser)).first->second;
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
    parser_recipe recipe;
    if (!resolve_derivation_variables(state, recipe.variables_json)) {
        return -1;
    }
    common_chat_params params;
    switch (derive_parser(templates.get(), tools_key, recipe.variables_json, params,
                          recipe.template_src)) {
        case parser_derivation::ok:
            break;
        case parser_derivation::unavailable:
            return RETRO_CHAT_PARSER_UNAVAILABLE;
        case parser_derivation::error:
            return -1;
    }

    // The same bos/eos text common_chat_templates_init gives the template when
    // it reads them from the vocabulary.
    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
    const auto piece = [&](llama_token token) {
        return token == LLAMA_TOKEN_NULL ? std::string() : common_token_to_piece(vocab, token, true);
    };
    recipe.bos_token = piece(llama_vocab_bos(vocab));
    recipe.eos_token = piece(llama_vocab_eos(vocab));
    recipe.tools_json = tools_key;

    state.tool_call_parser_cache = encode_parser_blob(recipe);
    state.tool_call_parser_tools = tools_key;
    return copy_json_out(
            state.tool_call_parser_cache, "failed to build tool-call parser",
            buffer, n_buffer, out_n_bytes);
}

int build_tool_call_parser_from_source(
        const char * template_src,
        const char * tools_json,
        const char * variables_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    std::string variables;
    if (!resolve_derivation_variables_source(template_src, variables_json, variables)) {
        return -1;
    }
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
    const std::string tools = tools_json ? std::string(tools_json) : std::string();
    parser_recipe recipe;
    common_chat_params params;
    switch (derive_parser(templates.get(), tools, variables, params, recipe.template_src)) {
        case parser_derivation::ok:
            break;
        case parser_derivation::unavailable:
            return RETRO_CHAT_PARSER_UNAVAILABLE;
        case parser_derivation::error:
            return -1;
    }
    recipe.tools_json = tools;
    recipe.variables_json = variables;
    return copy_json_out(
            encode_parser_blob(recipe), "failed to build tool-call parser", buffer, n_buffer, out_n_bytes);
}

int parse_assistant_output(
        const char * parser_blob,
        size_t n_parser_blob,
        const char * text,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    const auto params = cached_parser(std::string(parser_blob, n_parser_blob));
    if (!params) {
        return -1;
    }

    common_chat_msg message;
    try {
        message = common_chat_parse(common_chat_input(text), /*is_partial=*/false, *params);
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

int set_reasoning_budget(trainer_state & state, int32_t budget) {
    if (budget < -1) {
        set_error("reasoning budget must be -1 (the model decides) or a token count");
        return -1;
    }
    state.reasoning_budget = budget;
    state.reasoning_plan = reasoning_budget_plan{};
    return resolve_reasoning_budget(state) ? 0 : -1;
}

// The tags are the ones llama-server would budget with: those the chat parser
// derived for this template under these variables declares. No tag is
// spelled here, which is the point - a template whose parser knows no
// reasoning block is refused, not guessed at.
const reasoning_budget_plan * resolve_reasoning_budget(trainer_state & state) {
    reasoning_budget_plan & plan = state.reasoning_plan;
    if (plan.resolved) {
        return &plan;
    }
    plan = reasoning_budget_plan{};
    plan.budget = state.reasoning_budget;
    if (state.reasoning_budget < 0) {
        plan.resolved = true;
        return &plan;
    }

    common_chat_templates_ptr templates;
    try {
        templates = common_chat_templates_init(state.model.get(), "");
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::exception & err) {
        set_error(std::string("failed to initialize the model chat template: ") + err.what());
        return nullptr;
    }
    if (!templates) {
        set_error("model defines no chat template, so no reasoning block to budget");
        return nullptr;
    }
    std::string variables;
    common_chat_templates_inputs inputs;
    if (!resolve_derivation_variables(state, variables) || !derivation_inputs(variables, inputs)) {
        return nullptr;
    }
    common_chat_params params;
    try {
        params = common_chat_templates_apply(templates.get(), inputs);
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::exception & err) {
        set_error(std::string("failed to read the reasoning tags of the chat template: ") + err.what());
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
    const auto tokens = [&](const std::string & text) {
        return common_tokenize(vocab, text, /*add_special=*/false, /*parse_special=*/true);
    };
    if (!params.thinking_start_tag.empty()) {
        plan.start = tokens(params.thinking_start_tag);
    }
    for (const auto & tag : params.thinking_end_tags) {
        if (!tag.empty()) {
            plan.ends.push_back(tokens(tag));
        }
    }
    if (plan.start.empty() || plan.ends.empty() || plan.ends.front().empty()) {
        set_error("the chat template declares no reasoning block, so a reasoning budget has "
                  "nothing to limit");
        return nullptr;
    }
    plan.forced = plan.ends.front();

    // Counted like common_sampler_init counts its prefill: a tokenizer that
    // puts a space before the first special token adds one token the template
    // did not write.
    const auto prompt = tokens(params.generation_prompt);
    size_t tail = prompt.size();
    if (!prompt.empty() && !params.generation_prompt.empty()) {
        const std::string piece = common_token_to_piece(vocab, prompt.front(), true);
        if (!piece.empty() && std::isspace(static_cast<unsigned char>(piece[0]))
                && !std::isspace(static_cast<unsigned char>(params.generation_prompt[0]))) {
            --tail;
        }
    }
    plan.n_prompt_tail = tail;
    plan.budget = state.reasoning_budget;
    plan.active = true;
    plan.resolved = true;
    return &plan;
}

}  // namespace retro
