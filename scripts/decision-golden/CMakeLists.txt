# Out-of-tree build of the reference decision server and of the capture tool
# that records the golden fixtures under tests/fixtures/decision/.
#
# The workspace build of the fork turns `tools` off; this project turns it back
# on, in a build directory of its own, so that the reference is the server code
# of the vendored fork itself and nothing in the workspace build changes.
# Driven by scripts/decision-golden.sh.
cmake_minimum_required(VERSION 3.14)
project(retrograd-decision-golden CXX C)

set(FORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../crates/retrograd-ffi/runtime/vendor/llama.cpp")

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)

add_subdirectory("${FORK_DIR}" llama.cpp)

add_executable(decision-golden decision_golden.cpp)
target_compile_features(decision-golden PRIVATE cxx_std_17)
target_include_directories(decision-golden PRIVATE "${FORK_DIR}/tools/server" "${FORK_DIR}/tools/mtmd" "${FORK_DIR}")
target_link_libraries(decision-golden PRIVATE server-context)
set_target_properties(decision-golden PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
