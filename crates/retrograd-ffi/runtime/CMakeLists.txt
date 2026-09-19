cmake_minimum_required(VERSION 3.16)
project(retro_lora_train LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Mirrors the LLAMA_CPP_DIR override honoured by build.rs.
if(DEFINED ENV{LLAMA_CPP_DIR})
    set(LLAMA_CPP_DIR $ENV{LLAMA_CPP_DIR})
else()
    set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp)
endif()

# Keep this list in sync with `runtime_sources` in build.rs (the build that
# actually ships); this CMake target only serves IDE/tooling integration.
add_library(retro_lora_train STATIC
    src/retro_runtime.cpp
    src/retro_duty_cycle.cpp
    src/retro_backend.cpp
    src/retro_inventory.cpp
    src/retro_lora.cpp
    src/retro_training.cpp
    src/retro_preflight.cpp
    src/retro_rollout.cpp
    src/retro_probe.cpp
    src/retro_fused_ce.cpp
    src/retro_lora_train.cpp
    src/retro_checkpoint.cpp
    src/retro_chat_template.cpp
    src/retro_chat_parser.cpp
    ${LLAMA_CPP_DIR}/common/unicode.cpp
    ${LLAMA_CPP_DIR}/common/json.cpp
    ${LLAMA_CPP_DIR}/common/jinja/lexer.cpp
    ${LLAMA_CPP_DIR}/common/jinja/parser.cpp
    ${LLAMA_CPP_DIR}/common/jinja/runtime.cpp
    ${LLAMA_CPP_DIR}/common/jinja/value.cpp
    ${LLAMA_CPP_DIR}/common/jinja/string.cpp
    ${LLAMA_CPP_DIR}/common/jinja/caps.cpp
    # The chat chain retro_chat_parser.cpp derives its PEG parser from; see the
    # `llama_common_sources` comment in build.rs for why the closure reaches
    # common.cpp and what it deliberately stops short of.
    ${LLAMA_CPP_DIR}/common/chat.cpp
    ${LLAMA_CPP_DIR}/common/chat-peg-parser.cpp
    ${LLAMA_CPP_DIR}/common/chat-auto-parser-generator.cpp
    ${LLAMA_CPP_DIR}/common/chat-auto-parser-helpers.cpp
    ${LLAMA_CPP_DIR}/common/chat-diff-analyzer.cpp
    ${LLAMA_CPP_DIR}/common/json-schema.cpp
    ${LLAMA_CPP_DIR}/common/parsers/parsers.cpp
    ${LLAMA_CPP_DIR}/common/parsers/cohere2moe.cpp
    ${LLAMA_CPP_DIR}/common/parsers/deepseek.cpp
    ${LLAMA_CPP_DIR}/common/parsers/functionary-v3-2.cpp
    ${LLAMA_CPP_DIR}/common/parsers/gemma4.cpp
    ${LLAMA_CPP_DIR}/common/parsers/gigachat-v3.cpp
    ${LLAMA_CPP_DIR}/common/parsers/gpt-oss.cpp
    ${LLAMA_CPP_DIR}/common/parsers/kimi-k2.cpp
    ${LLAMA_CPP_DIR}/common/parsers/kimi-k3.cpp
    ${LLAMA_CPP_DIR}/common/parsers/lfm2.cpp
    ${LLAMA_CPP_DIR}/common/parsers/minicpm5.cpp
    ${LLAMA_CPP_DIR}/common/parsers/minimax-m3.cpp
    ${LLAMA_CPP_DIR}/common/parsers/ministral3.cpp
    ${LLAMA_CPP_DIR}/common/parsers/muse-glimmer.cpp
    ${LLAMA_CPP_DIR}/common/parsers/qwen3-coder.cpp
    ${LLAMA_CPP_DIR}/common/peg-parser.cpp
    ${LLAMA_CPP_DIR}/common/json-schema-to-grammar.cpp
    ${LLAMA_CPP_DIR}/common/log.cpp
    ${LLAMA_CPP_DIR}/common/trie.cpp
    ${LLAMA_CPP_DIR}/common/common.cpp
    ${LLAMA_CPP_DIR}/common/sampling.cpp
    ${LLAMA_CPP_DIR}/common/speculative.cpp
    ${LLAMA_CPP_DIR}/common/fit.cpp
    ${LLAMA_CPP_DIR}/common/ngram-cache.cpp
    ${LLAMA_CPP_DIR}/common/ngram-map.cpp
    ${LLAMA_CPP_DIR}/common/ngram-mod.cpp
    ${LLAMA_CPP_DIR}/common/reasoning-budget.cpp
)

target_include_directories(retro_lora_train
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/src
        ${LLAMA_CPP_DIR}/ggml/include
        # retro delta: for ggml-retro-quant.h (GGML_RETRO_DEQUANT_TYPES), which
        # retro_dequant_types() expands so tests enumerate the table.
        ${LLAMA_CPP_DIR}/ggml/src
        ${LLAMA_CPP_DIR}/common
        ${LLAMA_CPP_DIR}/vendor
)

# build.rs injects the real commit hashes; placeholders keep tooling happy.
target_compile_definitions(retro_lora_train PRIVATE
    RETRO_LLAMA_CPP_COMMIT="ide-tooling"
    RETRO_LLAMA_CPP_UPSTREAM_COMMIT="ide-tooling"
)

target_compile_features(retro_lora_train PUBLIC cxx_std_17)

if(APPLE)
    target_link_libraries(retro_lora_train PUBLIC "-framework Accelerate")
endif()
