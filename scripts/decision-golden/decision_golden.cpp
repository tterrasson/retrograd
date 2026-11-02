// Records the golden decision fixtures: for each request of a request file,
// the rendered prompt of every pass, its tokens before and after the layout,
// where the scores are read, the raw scores and the answer, all produced by
// the reference server code of the vendored fork (`server_decision_context`).
//
//   decision-golden --model M.gguf --requests requests.json --out golden.json
//                   [--device cpu|gpu] [--model-sha256 HEX] [--threads N]
//
// A request entry is `{"name", "body"[, "max_head_tokens"]}`; `body` is a
// `/v1/systemone` request body, and `max_head_tokens` replaces the value read
// from the model for that entry only (a marker readout).
//
// Every readout of the server is recorded: the logits of labels, or the
// largest of each group of labels, at the last token; the embeddings output at
// markers, in the column of the question type; the `[q|k]` pointer product; the
// first rows of a joint head that reads the decision order of each token; the
// pooled output. Each pass is evaluated alone, the whole prompt in one batch,
// as llama-server evaluates it with one slot. Requests with media are refused:
// the fixtures record text only.

// The headers the reference includes come first: the access trick below must
// only reach the declaration of the reference class, which keeps its
// rendering and its labels private. The tool reads them so that the fixture
// records the reference's own text rather than a reimplementation of it.
#include "server-common.h"
#include "server-task.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include "server-decision.h"
#undef private

#include "llama.h"
#include "../../src/llama-ext.h"

namespace {

struct options {
    std::string model;
    std::string requests;
    std::string out;
    std::string device = "cpu";
    std::string model_sha256;
    int threads = 0;
};

options parse_args(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            opts.model = next();
        } else if (arg == "--requests") {
            opts.requests = next();
        } else if (arg == "--out") {
            opts.out = next();
        } else if (arg == "--device") {
            opts.device = next();
        } else if (arg == "--model-sha256") {
            opts.model_sha256 = next();
        } else if (arg == "--threads") {
            opts.threads = std::stoi(next());
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    if (opts.model.empty() || opts.requests.empty() || opts.out.empty()) {
        throw std::invalid_argument("usage: decision-golden --model M.gguf --requests R.json --out G.json [--device cpu|gpu] [--model-sha256 HEX] [--threads N]");
    }
    if (opts.device != "cpu" && opts.device != "gpu") {
        throw std::invalid_argument("--device is cpu or gpu");
    }
    return opts;
}

std::string meta(const llama_model * model, const std::string & key) {
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf);
}

template <typename T>
json int_array(const std::vector<T> & values) {
    json out = json::array();
    for (const T v : values) {
        out.push_back(v);
    }
    return out;
}

// The text pieces of a joint or d1-omni prompt, cut at `sep`, each with the
// tokens of its text once its leading mark is removed: what the reference
// tokenizes one by one.
json pieces_of(const llama_vocab * vocab, const std::string & prompt, const std::string & sep, const std::vector<std::string> & marks) {
    json out = json::array();
    for (std::string piece : string_split(prompt, sep)) {
        for (const auto & mark : marks) {
            if (string_starts_with(piece, mark)) {
                piece = piece.substr(mark.size());
                break;
            }
        }
        out.push_back(json{
            {"text",   piece},
            {"tokens", int_array(common_tokenize(vocab, piece, false, true))},
        });
    }
    return out;
}

json sort_keys(const json & val) {
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(sort_keys(item));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = sort_keys(item);
        }
        json out = json::object();
        for (const auto & [key, item] : sorted) {
            out[key] = item;
        }
        return out;
    }
    return val;
}

json replace_text(const json & val, const std::string & search, const std::string & replace) {
    if (val.is_string()) {
        std::string text = val.get<std::string>();
        string_replace_all(text, search, replace);
        return text;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(replace_text(item, search, replace));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = replace_text(item, search, replace);
        }
        return out;
    }
    return val;
}

const std::string JOINT_MARKER        = "<<clef:";
const std::string JOINT_SEP           = "<<clef:sep>>";
const std::string JOINT_MARK_QUESTION = "<<clef:question>>";
const std::string JOINT_MARK_OPTION   = "<<clef:option>>";
const std::string D1OMNI_SEP          = "<<d1omni:sep>>";

// The joint prompt, rendered as fill_task_joint renders it; the tool checks
// that its pieces give the reference's tokens.
std::string render_joint(const server_decision_context & decision, const json & state, const std::vector<server_decision_question> & questions) {
    static const char * type_names[] = {"choice", "score", "noul"};
    json listed = json::array();
    for (const auto & question : questions) {
        json options = json::array();
        for (const auto & opt : question.options) {
            options.push_back(json{{"key", opt.key}, {"description", opt.description}});
        }
        listed.push_back(json{
            {"id",           question.id},
            {"type",         type_names[question.type]},
            {"instructions", question.instructions},
            {"options",      options},
        });
    }
    json inp = json{{"state", state}, {"questions", listed}};
    inp = replace_text(sort_keys(inp), JOINT_MARKER, "<<clef ");
    inp["images"]        = json::array();
    inp["sep"]           = JOINT_SEP;
    inp["mark_question"] = JOINT_MARK_QUESTION;
    inp["mark_option"]   = JOINT_MARK_OPTION;
    jinja::context ctx(decision.tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(decision.tmpl->prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}

// Evaluates one task alone, every token in one batch, and reads its scores as
// send_decision of llama-server does.
std::vector<float> read_scores(llama_context * ctx, const llama_model * model, const server_task & task) {
    const auto & decision = task.decision;
    const llama_tokens tokens = task.tokens.get_text_tokens();
    const bool all_outputs = task.need_embd();

    if (llama_memory_t memory = llama_get_memory(ctx)) {
        llama_memory_clear(memory, true);
    }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos[4] = {(llama_pos) i, 0, 0, 0};
        if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, pos)) {
            llama_batch_ext_free(batch);
            throw std::runtime_error("the prompt does not fit the batch");
        }
        if (all_outputs || i + 1 == tokens.size()) {
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        if (!decision.order.empty() && decision.order[i] != 0) {
            llama_batch_ext_set_decision_order(batch, idx, (llama_decision_order) decision.order[i]);
        }
    }
    const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    if (ret != 0) {
        throw std::runtime_error("llama_decode failed");
    }

    std::vector<float> scores;
    if (decision.pooled) {
        const float * embd = llama_get_embeddings_seq(ctx, 0);
        if (embd == nullptr) {
            throw std::runtime_error("no pooled embeddings");
        }
        scores.assign(embd, embd + llama_model_n_embd_out(model));
    } else if (!decision.labels.empty()) {
        const float * logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }
        for (const llama_token label : decision.labels) {
            scores.push_back(logits[label]);
        }
        if (!decision.label_groups.empty()) {
            std::vector<float> grouped;
            size_t i = 0;
            for (const int32_t n : decision.label_groups) {
                grouped.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
                i += n;
            }
            scores = std::move(grouped);
        }
    } else {
        for (int32_t i = 0; i < decision.n_scores; i++) {
            const float * embd = llama_get_embeddings_ith(ctx, i);
            if (embd == nullptr) {
                throw std::runtime_error("no embeddings for a joint head");
            }
            scores.push_back(embd[0]);
        }
        const int32_t n_embd_out = llama_model_n_embd_out(model);
        const int32_t n_pointer  = n_embd_out / 2;
        const float * embd_q = decision.pointer >= 0 ? llama_get_embeddings_ith(ctx, decision.pointer) : nullptr;
        for (const int32_t marker : decision.markers) {
            const float * embd = llama_get_embeddings_ith(ctx, marker);
            if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                throw std::runtime_error("no embeddings at a marker");
            }
            if (decision.pointer < 0) {
                scores.push_back(embd[decision.column]);
                continue;
            }
            float dot = 0.0f;
            for (int32_t i = 0; i < n_pointer; i++) {
                dot += embd_q[i] * embd[n_pointer + i];
            }
            scores.push_back(dot / sqrtf((float) n_pointer));
        }
    }
    return scores;
}

// The name of the backend the model runs on, the one the fixture is valid for.
std::string backend_name(const options & opts) {
    if (opts.device == "cpu") {
        return "cpu";
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char) std::tolower(c); });
            return name;
        }
    }
    throw std::runtime_error("--device gpu: no GPU device");
}

int run(const options & opts) {
    llama_backend_init();
    const std::string backend = backend_name(opts);

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    if (opts.device == "cpu") {
        mparams.devices      = no_devices;
        mparams.n_gpu_layers = 0;
    } else {
        mparams.n_gpu_layers = -1;
    }
    llama_model * model = llama_model_load_from_file(opts.model.c_str(), mparams);
    if (model == nullptr) {
        throw std::runtime_error("cannot load " + opts.model);
    }

    server_decision_context decision;
    decision.init(model);
    if (decision.type == COMMON_DECISION_TYPE_NONE) {
        throw std::runtime_error("not a decision model");
    }
    const size_t model_max_head_tokens = decision.max_head_tokens;
    const bool is_marker = decision.type == COMMON_DECISION_TYPE_LAYA;

    std::ifstream in(opts.requests);
    if (!in) {
        throw std::runtime_error("cannot read " + opts.requests);
    }
    const std::string request_text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json request_file = json::parse(request_text);

    // The context llama-server creates (common_init_from_params): the
    // embeddings output for the readouts that read it, one batch for the
    // whole prompt.
    const common_decision_type type = decision.type;
    const bool reads_embeddings = type == COMMON_DECISION_TYPE_LAYA || type == COMMON_DECISION_TYPE_KEV || type == COMMON_DECISION_TYPE_CLEF ||
        type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || type == COMMON_DECISION_TYPE_VLLM_DECISION2 || type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV;
    const uint32_t n_ctx = 8192;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_ctx;
    cparams.n_ubatch   = n_ctx;
    cparams.n_seq_max  = 1;
    cparams.op_offload = opts.device != "cpu";
    if (reads_embeddings) {
        cparams.embeddings    = true;
        cparams.pooling_type  = type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV ? LLAMA_POOLING_TYPE_UNSPECIFIED : LLAMA_POOLING_TYPE_NONE;
        cparams.n_outputs_max = cparams.n_batch;
    }
    if (opts.threads > 0) {
        cparams.n_threads       = opts.threads;
        cparams.n_threads_batch = opts.threads;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        throw std::runtime_error("cannot create a context");
    }

    json requests = json::array();
    for (const auto & entry : request_file.at("requests")) {
        const std::string name = entry.at("name").get<std::string>();
        const json & body      = entry.at("body");

        decision.max_head_tokens = entry.contains("max_head_tokens")
            ? entry.at("max_head_tokens").get<size_t>()
            : model_max_head_tokens;

        // A refused request records the message llama-server answers with.
        json record = json{{"name", name}};
        if (is_marker) {
            record["max_head_tokens"] = decision.max_head_tokens;
        }
        std::vector<server_decision_question> questions;
        json state;
        std::vector<raw_buffer> files;
        try {
            questions = decision.parse_questions(body);
            state     = decision.prepare_state(decision.parse_state(body, files));
        } catch (const std::invalid_argument & e) {
            record["error"] = e.what();
            requests.push_back(record);
            continue;
        }
        if (!files.empty()) {
            throw std::runtime_error(name + ": media are not recorded");
        }

        // The passes of the request, as handle_systemone builds them.
        struct pass {
            std::string id; // empty for the joint prompt of every question
            size_t variant = 0;
            server_task task = server_task(SERVER_TASK_TYPE_DECISION);
        };
        std::vector<pass> passes;
        std::string error;
        try {
            if (decision.is_joint()) {
                passes.push_back(pass{});
                decision.fill_task_joint(state, questions, files, nullptr, mtmd_helper_init_opt{}, passes.back().task);
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        passes.push_back(pass{question.id, variant});
                        decision.fill_task(state, questions, question, variant, files, nullptr, mtmd_helper_init_opt{}, passes.back().task);
                    }
                }
            }
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        if (!error.empty()) {
            record["error"] = error;
            requests.push_back(record);
            continue;
        }

        json traces = json::array();
        std::map<std::string, std::vector<std::vector<float>>> scores_of;
        std::vector<float> joint_scores;
        int32_t n_input_tokens = 0;
        for (auto & p : passes) {
            const server_decision_question * question = nullptr;
            for (const auto & q : questions) {
                if (q.id == p.id) {
                    question = &q;
                }
            }
            json trace = json{{"id", p.id.empty() ? json() : json(p.id)}, {"variant", p.variant}};

            std::string prompt;
            if (decision.type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
                prompt = render_joint(decision, state, { *question });
            } else if (decision.is_joint()) {
                prompt = render_joint(decision, state, questions);
            } else {
                prompt = decision.render(state, questions, *question, p.variant, 0);
            }
            trace["prompt"] = prompt;
            if (decision.type == COMMON_DECISION_TYPE_VLLM_DECISION2 || decision.is_joint()) {
                trace["pieces"] = pieces_of(decision.vocab, prompt, JOINT_SEP, {JOINT_MARK_QUESTION, JOINT_MARK_OPTION});
            } else if (decision.type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
                trace["pieces"] = pieces_of(decision.vocab, prompt, D1OMNI_SEP, {"<<d1omni:state>>", "<<d1omni:question>>", "<<d1omni:option>>"});
            } else {
                trace["rendered_tokens"] = int_array(common_tokenize(decision.vocab, prompt, false, true));
            }

            const llama_tokens tokens = p.task.tokens.get_text_tokens();
            if (tokens.size() > n_ctx) {
                throw std::runtime_error(name + ": prompt longer than the context");
            }
            if (trace.contains("pieces") && decision.type != COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
                // The replayed joint rendering gives the reference's tokens.
                llama_tokens joined;
                for (const auto & piece : trace["pieces"]) {
                    for (const auto & t : piece["tokens"]) {
                        joined.push_back(t.get<llama_token>());
                    }
                }
                if (joined != tokens) {
                    throw std::runtime_error(name + ": the joint prompt is not the reference's");
                }
            }
            const std::vector<float> scores = read_scores(ctx, model, p.task);
            n_input_tokens = decision.is_joint() ? (int32_t) tokens.size() : n_input_tokens + (int32_t) tokens.size();

            const auto & d = p.task.decision;
            trace["tokens"]       = int_array(tokens);
            trace["markers"]      = int_array(d.markers);
            trace["column"]       = d.column;
            trace["pointer"]      = d.pointer;
            trace["labels"]       = int_array(d.labels);
            trace["label_groups"] = int_array(d.label_groups);
            trace["order"]        = int_array(d.order);
            trace["n_scores"]     = d.n_scores;
            trace["pooled"]       = d.pooled;
            trace["scores"]       = scores;
            traces.push_back(trace);

            if (decision.is_joint()) {
                joint_scores = scores;
            } else {
                scores_of[p.id].push_back(scores);
            }
        }

        json answers = json::object();
        size_t i_score = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                scores.emplace_back(joint_scores.begin() + i_score, joint_scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
            } else {
                scores = scores_of[question.id];
            }
            answers[question.id] = decision.format_answer(question, scores);
        }

        record["questions"] = traces;
        record["response"]  = json{
            {"answers", answers},
            {"usage",   {{"input_tokens", n_input_tokens}, {"output_tokens", 0}}},
        };
        requests.push_back(record);
    }

    const std::string arch = meta(model, "general.architecture");
    json model_info = {
        {"file",          opts.model.substr(opts.model.find_last_of('/') + 1)},
        {"sha256",        opts.model_sha256},
        {"architecture",  arch},
        {"decision_type", meta(model, arch + ".decision.type")},
        {"n_embd_out",    llama_model_n_embd_out(model)},
    };
    if (is_marker) {
        model_info["max_head_tokens"] = model_max_head_tokens;
    }
    // What the readout read from the model, so that a replay without the
    // model can rebuild it.
    model_info["readout"] = json{
        {"labels",           int_array(decision.labels)},
        {"label_texts",      decision.label_texts},
        {"marker",           decision.token_marker},
        {"sep",              decision.token_sep},
        {"marker_text",      decision.text_marker},
        {"quoted_spellings", decision.quoted_spellings},
        {"n_options_max",    decision.n_options_max},
    };
    json temperatures = json::object();
    for (const auto & [key, value] : decision.temperatures) {
        temperatures[key] = value;
    }
    json score_bias = json::object();
    for (const auto & [n_levels, bias] : decision.score_bias) {
        score_bias[std::to_string(n_levels)] = bias;
    }
    model_info["readout"]["temperatures"] = temperatures;
    model_info["readout"]["score_bias"]   = score_bias;
    json golden = {
        {"model",    model_info},
        {"backend",  backend},
        {"requests", requests},
    };

    std::ofstream out(opts.out);
    out << golden.dump(1) << "\n";
    if (!out) {
        throw std::runtime_error("cannot write " + opts.out);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "decision-golden: %s\n", e.what());
        return 1;
    }
}
