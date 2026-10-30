// Grammar handles for the v2 step API: an owned GBNF string. The JSON-schema
// conversion reuses the already-compiled common/json-schema-to-grammar.cpp; the
// built-in any-JSON grammar is the vendored grammars/json.gbnf embedded at
// build time (build.rs generates retro_json_gbnf_embed.cpp for the shipped
// build, CMake configure_file for the IDE target) as retro_embedded_json_gbnf.
#include "retro_runtime.hpp"

#include "json-schema-to-grammar.h"
#include "json.h"

#include <exception>
#include <new>
#include <string>
#include <utility>

// Defined by the build-time-generated translation unit (raw string literal
// over the pinned grammars/json.gbnf).
extern const char retro_embedded_json_gbnf[];

struct retro_grammar {
    std::string gbnf;
};

namespace {

retro_grammar * make_grammar(std::string gbnf) {
    if (gbnf.empty()) {
        retro::set_error("the grammar text is empty");
        return nullptr;
    }
    return new retro_grammar{std::move(gbnf)};
}

} // namespace

extern "C" retro_grammar * retro_grammar_from_json_schema(const char * json_schema) {
    retro_grammar * result = nullptr;
    retro::boundary([&]() -> int {
        if (!json_schema || json_schema[0] == '\0') {
            retro::set_error("json_schema is required and must not be empty");
            return -1;
        }
        result = make_grammar(json_schema_to_grammar(common_json::parse(json_schema)));
        return result ? 0 : -1;
    });
    return result;
}

extern "C" retro_grammar * retro_grammar_from_gbnf(const char * gbnf) {
    retro_grammar * result = nullptr;
    retro::boundary([&]() -> int {
        if (!gbnf || gbnf[0] == '\0') {
            retro::set_error("gbnf is required and must not be empty");
            return -1;
        }
        result = make_grammar(std::string(gbnf));
        return result ? 0 : -1;
    });
    return result;
}

extern "C" retro_grammar * retro_grammar_json_object(void) {
    return retro_grammar_from_gbnf(retro_embedded_json_gbnf);
}

extern "C" const char * retro_grammar_gbnf(const retro_grammar * grammar) {
    if (!grammar) {
        retro::set_error("grammar is null");
        return nullptr;
    }
    return grammar->gbnf.c_str();
}

extern "C" void retro_grammar_free(retro_grammar * grammar) {
    delete grammar;
}
