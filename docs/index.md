---
layout: home

hero:
  name: Retrograd
  text: Run and train GGUF models
  tagline: Serve a model and its LoRA adapters through an OpenAI-compatible API, and train adapters or the model's own weights with SFT, preference optimization, PPO, GRPO, agentic GRPO or distillation. No conversion step, no Python stack.
  image:
    src: /logo.png
    alt: Retrograd
  actions:
    - theme: brand
      text: Quickstart
      link: /getting-started/quickstart
    - theme: alt
      text: Run a model
      link: /running/

features:
  - title: OpenAI-compatible server
    details: Chat Completions, Responses, Completions and Embeddings over a model and its adapters, with batched requests and a shared prefix cache.
    link: /running/
  - title: Training control plane
    details: Start, follow and control runs over HTTP or in the web interface, and serve their weights while they train.
    link: /running/server
  - title: SFT
    details: Learn from example answers in chat JSONL or plain text.
    link: /training/sft
  - title: Preference optimization
    details: Learn from pairs of answers, one preferred over the other, with DPO, IPO, SimPO or ORPO.
    link: /training/preference
  - title: PPO and GRPO
    details: Learn from a reward program that scores the model's own answers, with GSPO's sequence-level ratio as an option.
    link: /training/grpo
  - title: Agentic GRPO
    details: Multi-turn training with tool calls, MCP servers and sandboxed environments.
    link: /training/agent
  - title: Distillation
    details: Make a small model behave like a larger one, with no reward needed.
    link: /training/distill
---

## How it works

Retrograd loads a GGUF model as is, with no conversion, on CPU, Metal, Vulkan
or CUDA. Training and serving share one step-wise generation engine: rollouts
and HTTP requests go through the same scheduler.

Retrograd trains a LoRA adapter by default, written as a standard GGUF that
llama.cpp loads with `--lora`. It can also train some or all of the model's
weights. Several adapters of one base can be served from the same process.

```text
retrograd inspect --model base.gguf    # check the model and pick LoRA targets
retrograd train run.toml               # train
retrograd bench run.toml --adapter adapter.gguf   # compare base and adapter
retrograd chat run.toml --compare      # try it interactively
retrograd serve run.toml               # serve it through the OpenAI API
retrograd-infer serve.toml             # serve a model and its adapters
retrograd-server server.toml           # run the training and serving server
```

Start with the [quickstart](/getting-started/quickstart), or go to
[Running models](/running/) to serve a GGUF you already have.
