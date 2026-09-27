# Datasets

Paths in a configuration resolve against the directory of the TOML file, not
the shell's working directory. Paths passed on the command line (`--model`,
`--data`, …) resolve against the working directory.

## Chat JSONL

One JSON object per line, with a `messages` array. Roles are `system`, `user`
and `assistant` - and `tool`, in a [tool conversation](#tool-conversations):

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
  model generates the assistant reply. They cannot be tool records: an agentic
  run reads [scenarios](../training/agent) instead.

Conversations are rendered with the model's own GGUF chat template. A malformed
line is reported with its file and line number.

A GRPO or distillation prompt may carry a `rubric`: criteria a judge reads
when grading the answer to that prompt. SFT ignores it.

## Tool conversations

A record may also declare a tool catalog, make calls from an assistant turn,
and answer them with `tool` messages. Tool records and plain records can share
one file.

```json
{"tools": [{"type": "function",
            "function": {"name": "run", "description": "Run a shell command",
                         "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}],
 "messages": [
   {"role": "system", "content": "You fix bugs."},
   {"role": "user", "content": "test_parse_empty fails."},
   {"role": "assistant", "content": "Let me look.",
    "tool_calls": [{"id": "call_0", "type": "function",
                    "function": {"name": "run", "arguments": {"cmd": "pytest -q"}}}]},
   {"role": "tool", "tool_call_id": "call_0", "content": "1 failed"},
   {"role": "assistant", "content": "Fixed: the empty case now returns []."}
 ],
 "metadata": {"scenario_id": "fix-parser-3", "reward": 1.0}}
```

- `tools` uses the OpenAI function shape; the flat `{name, description,
  parameters}` form is read too.
- `arguments` is an object, or a string holding one, as the OpenAI API sends it.
- A `tool` message answers the call named by `tool_call_id`, or the next call
  still waiting when it has none. A call without an `id` is `call_0`, `call_1`,
  … in turn order. `name`, when present, must be the called tool's.
- `is_error: true` marks a failed call. The observation is shown to the model
  with the same `ERROR: ` prefix a rollout gives it.
- `metadata` is free-form and never read by training.

An assistant turn takes one of two forms:

- **Structured** (the default): `content` is the prose, without call markup,
  and `tool_calls` holds the calls. The model's own chat template writes the
  turn, so a record written by hand or by another model trains any student.
- **Raw** (`"raw": true`): `content` is the turn exactly as generated, call
  markup included, and is trained as is. `tool_calls` still lists the calls,
  and the student's parser must read back exactly those from `content` - which
  a model of another family's format will not.

Rules for tool records, on top of the ones above:

- Every call is answered before the next message that is not a `tool` message,
  and only a call's own turn is answered.
- A call names a tool the record declares; a record with calls declares tools.
- Two `user` messages in a row are allowed: it is how an environment's opening
  text reaches the model. Two `assistant` messages in a row are not.
- The record ends on an assistant turn, or on the results of its calls.
- `content` may be empty on an assistant turn with calls and on a `tool`
  message.

A tool record is rendered the way an agentic rollout renders its conversation,
with the same catalog, the same observation framing and the same parser. Each
turn is read back and must yield exactly its declared calls: a record that
would teach a call the rollout cannot parse is refused with its line number.
The prepared tokens match a rollout of the same conversation, up to one
limit: sampling may split the same text into tokens differently from the
tokenizer, and a record can only hold the text.

`retrograd collect` writes this format from successful rollouts; see
[Tools](../training/tools#warm-start-sft).

## Preference pairs

A [preference run](../training/preference) reads pairs: a `prompt`, then the
`chosen` and the `rejected` answer to it, each a list of messages in the chat
schema above.

```jsonl
{"prompt":[{"role":"user","content":"What is 2 + 2?"}],"chosen":[{"role":"assistant","content":"4"}],"rejected":[{"role":"assistant","content":"5"}]}
```

The prompt ends on a user turn; both answers start with an assistant turn and
must differ. A preference file is never inferred from its content: on the
server, upload it with `format=preference-jsonl`.

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
