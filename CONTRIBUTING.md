# Contributing to Retrograd

The rules live in the documentation; this page points to them.

## Getting the sources

```sh
git clone --recurse-submodules https://github.com/tterrasson/retrograd.git
```

The `llama.cpp` fork is a submodule at
`crates/retrograd-ffi/runtime/vendor/llama.cpp`. `scripts/setup-llama-cpp.sh`
initializes it in an existing clone and adds the `upstream` remote you need to
work on the fork itself.

## Before opening a pull request

- Read [the contribution principles](docs/engineering/contributing.md): scope,
  architecture rules, errors and numeric conversions.
- Run the fast lanes on every change, `scripts/test-fast-rust.sh` and
  `scripts/test-fast-python.sh`. [Tests and validation](docs/engineering/tests/notice.md)
  says which other lane a change needs. CI runs the CPU lanes only: the Metal,
  Vulkan and CUDA lanes stay a manual step on real hardware, so say in the pull
  request which ones you ran.
- A change to `ggml` or `llama.cpp` itself goes to the fork,
  [tterrasson/llama.cpp-retrograd](https://github.com/tterrasson/llama.cpp-retrograd),
  following [the fork workflow](docs/engineering/LLAMA_CPP_FORK_WORKFLOW.md).

## Security issues

Do not open a public issue; see [SECURITY.md](SECURITY.md).

## License

Retrograd is licensed under the [MIT License](LICENSE). By contributing, you
agree that your contribution is licensed under the same terms.
