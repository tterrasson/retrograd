# Security policy

## Supported versions

Retrograd is a 0.x project. Security fixes land on `main` and in
the next release; older releases are not patched.

## Reporting a vulnerability

Please do not open a public issue. Report the problem privately through
[GitHub's private vulnerability reporting](https://github.com/tterrasson/retrograd/security/advisories/new)
(**Security** tab, then **Report a vulnerability**), with:

- the Retrograd version or commit, the platform and the backend (CPU, Metal,
  Vulkan, CUDA);
- the steps or the input that reproduce the problem;
- what an attacker gains.

## Scope

What Retrograd treats as trusted, so that a report can be weighed against it:

- **Run configurations are trusted input.** A configuration can start reward
  commands, judges, tools and MCP servers; running a configuration from an
  untrusted source is equivalent to running its commands yourself.
- **The HTTP server** (`retrograd-server`) has one authentication model: a
  single shared bearer token, or none. It refuses to listen on a non-loopback
  address without a token. A way to reach a route without the token, or to
  bind beyond loopback without one, is in scope.
- **GGUF files** are parsed by the vendored
  [llama.cpp fork](https://github.com/tterrasson/llama.cpp-retrograd). A defect
  that also exists in upstream llama.cpp belongs to
  [its security policy](https://github.com/ggml-org/llama.cpp/security/policy);
  a defect in the fork's own changes belongs here.
