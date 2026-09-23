# Datasets

Paths in a configuration resolve against the directory of the TOML file, not
the shell's working directory. Paths passed on the command line (`--model`,
`--data`, …) resolve against the working directory.

## Chat JSONL

One JSON object per line, with a `messages` array. Roles are `system`, `user`
and `assistant`:

```jsonl
{"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
{"messages":[{"role":"user","content":"Name a primary color."},{"role":"assistant","content":"Blue."}]}
```

Rules:

- No empty `messages` array and no empty `content`.
- After optional system messages, user and assistant turns alternate.
- **SFT** data needs at least one assistant message. Only assistant content is
  trained; system and user content is context.
- **PPO, GRPO and distillation** prompts must end with a user message. The
  model generates the assistant reply.

Conversations are rendered with the model's own GGUF chat template. A malformed
line is reported with its file and line number.

## Plain text

SFT also accepts plain text. The file is tokenized as one stream and split into
context windows; every token is trained.

```toml
[sft]
data = "data/train.txt"
```

## Format detection

For `[sft].data`, `.jsonl` and `.json` are read as chat JSONL, `.txt` and `.md`
as text. For any other extension, a file whose first non-empty line starts with
`{` is read as JSONL; set `data_format = "jsonl"` or `"text"` to be explicit.
Prompt files for PPO, GRPO and distillation are always chat JSONL.

## Evaluation data

`[evaluation]` points to a held-out file in the same format as the training
data. Keep it separate from the training set.

```toml
[evaluation]
data = "data/eval.jsonl"
every_iterations = 1   # every SFT epoch, or every rollout update
max_examples = 100     # rollout algorithms: cap on generated examples
patience = 3           # stop after 3 evaluations without improvement
```

SFT measures loss on assistant tokens. PPO and GRPO generate one answer per
held-out prompt and report the mean reward. See
[Checkpoints](../operations/checkpoints#keep-the-best-evaluation) to keep the
best result.
