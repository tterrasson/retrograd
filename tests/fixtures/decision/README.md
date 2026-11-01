# Decision fixtures

Reference numbers for the decision models served on `/v1/systemone`, recorded
from the decision server of the vendored llama.cpp fork by
`scripts/decision-golden.sh`. They are what a decision implementation in this
repository has to reproduce.

| File | What it is |
|---|---|
| `requests.json` | the frozen request set, `/v1/systemone` bodies |
| `golden-<model>.<backend>.json` | the reference for one model on one backend |
| `TINYLAYA_FIXTURE.toml` | the manifest of the small test model, fetched by `scripts/fetch-cpu-fixture.sh decision/TINYLAYA_FIXTURE` |

## The requests

| Name | What it pins |
|---|---|
| `reference_example` | the request of the upstream pull request: an object state, a 4-option choice, a noul with and without criteria, a 4-level score, a 10-option choice with `null` descriptions |
| `text_state_null_descriptions` | a text state, a choice mixing a description and `null` ones |
| `marker_text_in_inputs` | the marker text (`<mask>`) in the state, the instructions, a description and an option key: it is replaced before rendering |
| `twenty_long_options` | 20 long options: every option is cut, then all are shrunk evenly to the head budget |
| `score_ten_levels` | a 10-level score, an array state |
| `long_question_model_budget` | a question and options over the head budget, with the budget of the model |
| `long_question_budget_512` | the same request with the budget at 512, the override the configuration may set |

`llama-server` reads the budget from the model only, so the last entry is
recorded by the capture tool and not replayed against the server.

## What a golden file holds

For each request: the head budget used, and per question the rendered prompt
(`prompt`), its tokens (`rendered_tokens`), the tokens after the layout
(`tokens`), the positions the scores are read at (`markers`), the score column
(`column`, the question type), the raw scores, and the server response
(`answers`, `usage`). The model file is identified by name and SHA-256; the
files themselves are never committed.

| Model | Readout | Head budget in the file | Opt-in variable |
|---|---|---|---|
| `tinylaya` (Q8_0, 97 MB, Apache-2.0) | `laya` | 192 | always, `RETRO_TINYLAYA_FIXTURE` to relocate it |
| `julia-1` (`Julia-1-Q8_0.gguf`, Apache-2.0) | `laya` | 256 | `RETRO_DECISION_JULIA` |
| `laya` (`Laya-Q8_0.gguf`, Apache-2.0) | `laya` | 192 | `RETRO_DECISION_LAYA` |

## Tolerance

The CPU references are built as the workspace builds its CPU backend: no BLAS,
no Accelerate, no host-specific instructions (`scripts/decision-golden/CMakeLists.txt`).
llama.cpp's default macOS build multiplies matrices through Accelerate, which
changes the arithmetic; a reference the workspace cannot run the kernels of is
not one it can match. The Metal references do not depend on these options.

Measured on the recorded files (Apple M-series, CPU and Metal backends):

| Comparison | Largest probability difference |
|---|---|
| capture tool against `llama-server` built the same way, same backend | 0 (same code, same kernels) |
| two runs, same backend | 0 |
| CPU against Metal, `tinylaya` Q8_0 | 1.2e-3 |
| CPU against Metal, `laya` Q8_0 | 1.1e-1 (`twenty_long_options`, two near-tied options) |
| CPU against Metal, `julia-1` Q8_0 | 3.1e-1, and one choice changes |
| this CPU build against llama.cpp's default (Accelerate) build | `tinylaya` 3.8e-4, `laya` 4.9e-2, `julia-1` 3.6e-1 (two choices change) |
| CPU against Metal, `julia-1` BF16, default build | 2.7e-2 |
| `julia-1` Q8_0 against BF16, CPU, default build | 2.6e-1 |

What follows from it:

- **Same weights, same backend: 1e-3**, and a parity test compares against the
  golden file of its own backend. Tokens, markers and usage are exact on any
  backend: they do not depend on arithmetic.
- **Across backends there is no single tolerance.** The CPU quantizes the
  activations of a Q8_0 matrix product to Q8_0, Metal keeps them in F32, and
  Julia-1's head amplifies the difference far beyond what Laya's does. The
  upstream pull request reports the same fragility (Julia-1 is its worst model
  against the PyTorch reference, 2.2e-2, measured on a higher-precision file).
- Laya on Metal reproduces the values the upstream pull request reports for
  `reference_example` within 1.4e-3, and its token count (414) exactly.
- The head budget changes answers: on Metal (and on the default CPU build),
  Julia-1 picks `legal` on `long_question_model_budget` (256) and `support` with
  512. The value in the model file is the default for that reason.
