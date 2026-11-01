// Records the golden decision fixtures: for each request of a request file,
// the rendered prompt, the tokens before and after the layout, the positions
// the scores are read at, the raw scores and the answer, all produced by the
// reference server code of the vendored fork (`server_decision_context`).
//
//   decision-golden --model M.gguf --requests requests.json --out golden.json
//                   [--device cpu|gpu] [--model-sha256 HEX] [--threads N]
//
// A request entry is `{"name", "body"[, "max_head_tokens"]}`; `body` is a
// `/v1/systemone` request body, and `max_head_tokens` replaces the value read
// from the model for that entry only.
//
// Only the readouts that read one sequence per question at marker positions
// (laya) are recorded; the tool refuses the other ones rather than writing a
// fixture that does not say what it measured.

// The rendering and the head budget are private to the reference class; this
// tool reads them so that the fixture records the reference's own text rather
// than a reimplementation of it.
#define private public
#include "server-decision.h"
#undef private

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

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

json token_array(const llama_tokens & tokens) {
    json out = json::array();
    for (const llama_token t : tokens) {
        out.push_back(t);
    }
    return out;
}

// One sequence, outputs at the markers only, the column of the question type.
std::vector<float> read_scores(llama_context * ctx, const llama_tokens & tokens, const std::vector<int32_t> & markers, int32_t column, int32_t n_embd_out) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    for (const int32_t m : markers) {
        batch.logits[m] = 1;
    }
    batch.n_tokens = (int32_t) tokens.size();

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }
    if (column < 0 || column >= n_embd_out) {
        llama_batch_free(batch);
        throw std::runtime_error("score column out of range");
    }

    std::vector<float> scores;
    for (const int32_t m : markers) {
        const float * embd = llama_get_embeddings_ith(ctx, m);
        if (embd == nullptr) {
            llama_batch_free(batch);
            throw std::runtime_error("no embeddings at a marker");
        }
        scores.push_back(embd[column]);
    }
    llama_batch_free(batch);
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
    if (decision.type != COMMON_DECISION_TYPE_LAYA) {
        throw std::runtime_error("only the laya readout is recorded by this tool");
    }
    const size_t model_max_head_tokens = decision.max_head_tokens;

    std::ifstream in(opts.requests);
    if (!in) {
        throw std::runtime_error("cannot read " + opts.requests);
    }
    const std::string request_text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json request_file = json::parse(request_text);

    // a non-causal sequence must fit in one ubatch, size both after the longest prompt
    const uint32_t n_ctx = 8192;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = n_ctx;
    cparams.n_batch      = n_ctx;
    cparams.n_ubatch     = n_ctx;
    cparams.n_seq_max    = 1;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.op_offload   = opts.device != "cpu";
    if (opts.threads > 0) {
        cparams.n_threads       = opts.threads;
        cparams.n_threads_batch = opts.threads;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        throw std::runtime_error("cannot create a context");
    }
    const int32_t n_embd_out = llama_model_n_embd_out(model);

    json requests = json::array();
    for (const auto & entry : request_file.at("requests")) {
        const std::string name = entry.at("name").get<std::string>();
        const json & body      = entry.at("body");

        decision.max_head_tokens = entry.contains("max_head_tokens")
            ? entry.at("max_head_tokens").get<size_t>()
            : model_max_head_tokens;

        const auto questions = decision.parse_questions(body);
        std::vector<raw_buffer> files;
        const json state = decision.prepare_state(decision.parse_state(body, files));
        if (!files.empty()) {
            throw std::runtime_error(name + ": images are not recorded");
        }

        json answers         = json::object();
        json question_traces = json::array();
        int32_t n_input_tokens = 0;
        for (const auto & question : questions) {
            if (decision.n_variants(question) != 1) {
                throw std::runtime_error(name + ": a question with several variants");
            }
            const std::string prompt = decision.render(state, questions, question, 0, 0);
            const llama_tokens rendered = common_tokenize(decision.vocab, prompt, false, true);

            server_task task(SERVER_TASK_TYPE_DECISION);
            decision.fill_task(state, questions, question, 0, files, nullptr, mtmd_helper_init_opt{}, task);
            const llama_tokens tokens = task.tokens.get_text_tokens();
            if (tokens.size() > n_ctx) {
                throw std::runtime_error(name + ": prompt longer than the context");
            }

            const std::vector<float> scores = read_scores(ctx, tokens, task.decision.markers, task.decision.column, n_embd_out);
            n_input_tokens += (int32_t) tokens.size();

            answers[question.id] = decision.format_answer(question, { scores });

            json markers = json::array();
            for (const int32_t m : task.decision.markers) {
                markers.push_back(m);
            }
            question_traces.push_back(json{
                {"id",              question.id},
                {"prompt",          prompt},
                {"rendered_tokens", token_array(rendered)},
                {"tokens",          token_array(tokens)},
                {"markers",         markers},
                {"column",          task.decision.column},
                {"scores",          scores},
            });
        }

        json record = {
            {"name",            name},
            {"max_head_tokens", decision.max_head_tokens},
            {"questions",       question_traces},
            {"response",        json{
                {"answers", answers},
                {"usage",   {{"input_tokens", n_input_tokens}, {"output_tokens", 0}}},
            }},
        };
        requests.push_back(record);
    }

    const std::string arch = meta(model, "general.architecture");
    json golden = {
        {"model", {
            {"file",            opts.model.substr(opts.model.find_last_of('/') + 1)},
            {"sha256",          opts.model_sha256},
            {"architecture",    arch},
            {"decision_type",   meta(model, arch + ".decision.type")},
            {"max_head_tokens", model_max_head_tokens},
            {"n_embd_out",      n_embd_out},
        }},
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
