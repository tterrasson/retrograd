# Datasets and paths

Retrograd reads files from the machine running the CLI. A relative path is
resolved against the TOML file's directory, not against the current shell
directory.

## Plain text

Set `data_format = "text"` for an SFT text file. The file is tokenized as a
stream and split into overlapping context windows. Text mode does not identify
prompts and answers, so every token is a training target.

```toml
[sft]
data = "data/train.txt"
data_format = "text"
```

For SFT, `.txt` and `.md` paths are inferred as text when `data_format` is
omitted.

## Chat JSONL

Each non-empty line is one JSON object with a `messages` array. Every message
has a `role` and `content`. The accepted roles are `system`, `user`, and
`assistant`.

```jsonl
{"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
{"messages":[{"role":"user","content":"Name a primary color."},{"role":"assistant","content":"Blue."}]}
```

Chat JSONL is used for SFT data, PPO prompts, GRPO prompts, and evaluation
files. Its role determines the final-message requirement:

- `messages` is not empty.
- Content is not empty.
- Roles are only `system`, `user`, or `assistant`.
- User and assistant turns alternate after optional system messages.
- An SFT record must contain at least one assistant message.
- A PPO or GRPO prompt record must end in a non-empty user message.

During SFT, only assistant content becomes a training target. System and user
content provides context and is masked from the loss. During PPO and GRPO, the
assistant turn is generated after the final user message.

For SFT, `.jsonl` and `.json` paths are inferred as chat JSONL. For another
extension, set `data_format = "jsonl"` explicitly; if no format is given, the
loader can also recognize a file whose first non-empty line starts with `{`.
PPO and GRPO prompts are always chat JSONL.

## Training and evaluation files

The optional `[evaluation]` section points to a held-out file. Keep it separate
from training data. For SFT, evaluation measures loss on assistant tokens. For
PPO and GRPO, evaluation generates responses and sends them through the same
reward path or evaluation configuration used by the run.

```toml
[evaluation]
data = "data/eval.jsonl"
every_iterations = 1
max_examples = 100
patience = 3
min_delta = 0.0
```

The loader reports the file and line number for malformed JSONL records. Fix
the source record rather than deleting the error from the dataset.
