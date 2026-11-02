#!/usr/bin/env python3
"""Generate the tiny decision model: a ModernBERT encoder with a `laya` head.

The shape of a real decision model (an encoder, then a decision head of plain
pre-norm blocks evaluated once per question type, then a scorer), at a size a
unit test can afford: 2 encoder blocks and 1 head block, n_embd 64, a
byte-fallback vocabulary. Every tensor is F32 and every weight is drawn from
the explicit xorshift of `gen-tiny-fixture.py`, so the bytes are a property of
these two files and a finite-difference gradient is not polluted by storage
rounding.

What the graph exercises, and why each is here:

- encoder blocks: LayerNorm without bias (GGML_OP_NORM), fused QKV, RoPE,
  symmetric sliding-window attention on block 1 (window 4, shorter than the
  test sequences), the fused GEGLU with the erf GELU (GGML_GLU_OP_GEGLU_ERF);
- the decision head: one block per question type over the same input plus a
  row of `token_types`, LayerNorm with bias, full attention without RoPE, a
  ReLU FFN with biases, and the scorer LayerNorm -> Linear -> GELU (erf) ->
  Linear, the three scores concatenated;
- the metadata a decision server reads: `decision.type = laya`,
  `decision.max_head_tokens`, the mask and sep tokens, and a `systemone`
  template.

With `--type NAME`, the file is a tiny model of another decision type of
llama.cpp's decision server, with a `systemone` template written for it and
the metadata the type reads (see `TYPES`): the same encoder with the d1-omni
layout, a causal trunk whose logits are read at labels, a qwen3 trunk with
the head of `vllm-decision2`, a qwen35 trunk (gated delta net and gated
attention) with the joint head of `clef` or the `[q|k]` projection of `kev`.
Their answers mean nothing; what they pin is the readout of each type end to
end at a size a unit test can afford. Every tensor is F32.

Usage: scripts/gen-tiny-decision-fixture.py [--type NAME] <destination.gguf>
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gen_tiny_fixture", Path(__file__).with_name("gen-tiny-fixture.py")
)
assert _spec is not None and _spec.loader is not None
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

ARCHITECTURE = "modern-bert"
MODEL_NAME = "retrograd-tiny-decision"

N_LAYER_ENC = 2
N_LAYER_HEAD = 1
N_LAYER = N_LAYER_ENC + N_LAYER_HEAD
N_EMBD = 64
N_HEAD = 4
N_FF_ENC = 96  # the encoder FFN up projection is 2*N_FF_ENC wide (GEGLU)
N_FF_HEAD = 128
N_CTX_TRAIN = 512
NORM_EPS = 1.0e-5
ROPE_FREQ_BASE = 10000.0
SLIDING_WINDOW = 4
SWA_PATTERN = 2  # block 0 full, block 1 windowed; head blocks are always full
N_TOKEN_TYPES = 3  # choice, score, noul
MAX_HEAD_TOKENS = 64

PAD, EOS, BOS, UNK, MASK = 0, 1, 2, 3, 4

SYSTEMONE = (
    "<bos>{{ type }} question: "
    "{{ instructions if instructions is string else instructions | tojson }}<eos>"
    "{% for o in options %}<mask> {% if o.description %}"
    "{{ o.description if o.description is string else o.description | tojson }}"
    "{% else %}{{ o.key }}{% endif %}{% endfor %}"
    "<eos>{{ state if state is string else state | tojson }}<eos>"
)


def vocabulary() -> tuple[list[str], list[int]]:
    """Five specials in the order of the real decision vocabularies, then bytes."""
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    types = [
        tiny.TOKEN_TYPE_CONTROL,
        tiny.TOKEN_TYPE_CONTROL,
        tiny.TOKEN_TYPE_CONTROL,
        tiny.TOKEN_TYPE_UNKNOWN,
        tiny.TOKEN_TYPE_CONTROL,
    ]
    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        types.append(tiny.TOKEN_TYPE_BYTE)
    tokens.append("▁the")
    types.append(tiny.TOKEN_TYPE_NORMAL)
    return tokens, types


def tensors() -> list[tuple[str, list[int], int, bytes]]:
    n_vocab = len(vocabulary()[0])
    out: list[tuple[str, list[int], int, bytes]] = []
    seed = 0xD1B54A32D192ED03

    def add(name: str, ne: list[int], scale: float, offset: float = 0.0) -> None:
        nonlocal seed
        count = 1
        for dim in ne:
            count *= dim
        rng = tiny.Xorshift(seed)
        values = [offset + scale * rng.unit() for _ in range(count)]
        out.append((name, ne, tiny.GGML_TYPE_F32, tiny.pack(values, tiny.GGML_TYPE_F32)))
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

    add("token_embd.weight", [N_EMBD, n_vocab], 0.5)
    add("token_embd_norm.weight", [N_EMBD], 0.1, 1.0)
    for il in range(N_LAYER_ENC):
        if il != 0:  # block 0 reads the embedding norm, it has no attn_norm
            add(f"blk.{il}.attn_norm.weight", [N_EMBD], 0.1, 1.0)
        add(f"blk.{il}.attn_qkv.weight", [N_EMBD, 3 * N_EMBD], 0.2)
        add(f"blk.{il}.attn_output.weight", [N_EMBD, N_EMBD], 0.15)
        add(f"blk.{il}.ffn_norm.weight", [N_EMBD], 0.1, 1.0)
        add(f"blk.{il}.ffn_up.weight", [N_EMBD, 2 * N_FF_ENC], 0.2)
        add(f"blk.{il}.ffn_down.weight", [N_FF_ENC, N_EMBD], 0.15)
    add("output_norm.weight", [N_EMBD], 0.1, 1.0)
    for il in range(N_LAYER_ENC, N_LAYER):
        add(f"blk.{il}.attn_norm.weight", [N_EMBD], 0.1, 1.0)
        add(f"blk.{il}.attn_norm.bias", [N_EMBD], 0.05)
        add(f"blk.{il}.attn_qkv.weight", [N_EMBD, 3 * N_EMBD], 0.2)
        add(f"blk.{il}.attn_qkv.bias", [3 * N_EMBD], 0.05)
        add(f"blk.{il}.attn_output.weight", [N_EMBD, N_EMBD], 0.15)
        add(f"blk.{il}.attn_output.bias", [N_EMBD], 0.05)
        add(f"blk.{il}.ffn_norm.weight", [N_EMBD], 0.1, 1.0)
        add(f"blk.{il}.ffn_norm.bias", [N_EMBD], 0.05)
        add(f"blk.{il}.ffn_up.weight", [N_EMBD, N_FF_HEAD], 0.2)
        add(f"blk.{il}.ffn_up.bias", [N_FF_HEAD], 0.05)
        add(f"blk.{il}.ffn_down.weight", [N_FF_HEAD, N_EMBD], 0.15)
        add(f"blk.{il}.ffn_down.bias", [N_EMBD], 0.05)
    add("token_types.weight", [N_EMBD, N_TOKEN_TYPES], 0.5)
    add("cls.norm.weight", [N_EMBD], 0.1, 1.0)
    add("cls.norm.bias", [N_EMBD], 0.05)
    add("cls.weight", [N_EMBD, N_EMBD], 0.2)
    add("cls.bias", [N_EMBD], 0.05)
    add("cls.output.weight", [N_EMBD, 1], 0.3)
    add("cls.output.bias", [1], 0.05)
    return out


def uint32_array(values: list[int]) -> bytes:
    return struct.pack(f"<IQ{len(values)}I", tiny.T_UINT32, len(values), *values)


def metadata() -> list[tuple[str, int, bytes]]:
    tokens, types = vocabulary()
    a = ARCHITECTURE
    s = tiny.scalar
    e = tiny.encode_string
    return [
        ("general.architecture", tiny.T_STRING, e(a)),
        ("general.name", tiny.T_STRING, e(MODEL_NAME)),
        ("general.file_type", tiny.T_UINT32, s(tiny.T_UINT32, tiny.FILE_TYPE_ALL_F32)),
        (f"{a}.block_count", tiny.T_UINT32, s(tiny.T_UINT32, N_LAYER)),
        (f"{a}.context_length", tiny.T_UINT32, s(tiny.T_UINT32, N_CTX_TRAIN)),
        (f"{a}.embedding_length", tiny.T_UINT32, s(tiny.T_UINT32, N_EMBD)),
        (
            f"{a}.feed_forward_length",
            tiny.T_ARRAY,
            uint32_array([N_FF_ENC] * N_LAYER_ENC + [N_FF_HEAD] * N_LAYER_HEAD),
        ),
        (f"{a}.attention.head_count", tiny.T_UINT32, s(tiny.T_UINT32, N_HEAD)),
        (f"{a}.attention.layer_norm_epsilon", tiny.T_FLOAT32, s(tiny.T_FLOAT32, NORM_EPS)),
        (f"{a}.attention.causal", tiny.T_BOOL, s(tiny.T_BOOL, False)),
        (f"{a}.attention.sliding_window", tiny.T_UINT32, s(tiny.T_UINT32, SLIDING_WINDOW)),
        (f"{a}.attention.sliding_window_pattern", tiny.T_UINT32, s(tiny.T_UINT32, SWA_PATTERN)),
        (f"{a}.rope.freq_base", tiny.T_FLOAT32, s(tiny.T_FLOAT32, ROPE_FREQ_BASE)),
        (f"{a}.rope.freq_base_swa", tiny.T_FLOAT32, s(tiny.T_FLOAT32, ROPE_FREQ_BASE)),
        (f"{a}.hidden_activation", tiny.T_STRING, e("gelu")),
        (f"{a}.decision.type", tiny.T_STRING, e("laya")),
        (f"{a}.decision.block_count", tiny.T_UINT32, s(tiny.T_UINT32, N_LAYER_HEAD)),
        (f"{a}.decision.max_head_tokens", tiny.T_UINT32, s(tiny.T_UINT32, MAX_HEAD_TOKENS)),
        ("tokenizer.ggml.model", tiny.T_STRING, e("llama")),
        ("tokenizer.ggml.tokens", tiny.T_ARRAY, tiny.string_array(tokens)),
        (
            "tokenizer.ggml.scores",
            tiny.T_ARRAY,
            tiny.numeric_array(tiny.T_FLOAT32, [0.0] * len(tokens)),
        ),
        ("tokenizer.ggml.token_type", tiny.T_ARRAY, tiny.numeric_array(tiny.T_INT32, types)),
        ("tokenizer.ggml.bos_token_id", tiny.T_UINT32, s(tiny.T_UINT32, BOS)),
        ("tokenizer.ggml.eos_token_id", tiny.T_UINT32, s(tiny.T_UINT32, EOS)),
        ("tokenizer.ggml.unknown_token_id", tiny.T_UINT32, s(tiny.T_UINT32, UNK)),
        ("tokenizer.ggml.seperator_token_id", tiny.T_UINT32, s(tiny.T_UINT32, EOS)),
        ("tokenizer.ggml.padding_token_id", tiny.T_UINT32, s(tiny.T_UINT32, PAD)),
        ("tokenizer.ggml.mask_token_id", tiny.T_UINT32, s(tiny.T_UINT32, MASK)),
        ("tokenizer.ggml.add_bos_token", tiny.T_BOOL, s(tiny.T_BOOL, False)),
        ("tokenizer.ggml.add_eos_token", tiny.T_BOOL, s(tiny.T_BOOL, False)),
        ("tokenizer.ggml.token_type_count", tiny.T_UINT32, s(tiny.T_UINT32, N_TOKEN_TYPES)),
        ("tokenizer.chat_template.systemone", tiny.T_STRING, e(SYSTEMONE)),
        ("tokenizer.chat_templates", tiny.T_ARRAY, tiny.string_array(["systemone"])),
        ("general.alignment", tiny.T_UINT32, s(tiny.T_UINT32, tiny.ALIGNMENT)),
    ]


def build() -> bytes:
    return write_gguf(tensors(), metadata())


def write_gguf(table, kvs) -> bytes:
    header = bytearray(
        struct.pack("<IIQQ", tiny.GGUF_MAGIC, tiny.GGUF_VERSION, len(table), len(kvs))
    )
    for key, value_type, payload in kvs:
        tiny.write_kv(header, key, value_type, payload)
    offset = 0
    for name, ne, ggml_type, payload in table:
        tiny.write_string(header, name)
        header += struct.pack("<I", len(ne))
        for dim in ne:
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", ggml_type)
        header += struct.pack("<Q", offset)
        offset += (len(payload) + tiny.ALIGNMENT - 1) // tiny.ALIGNMENT * tiny.ALIGNMENT
    header += b"\0" * ((-len(header)) % tiny.ALIGNMENT)
    body = bytearray()
    for _, _, _, payload in table:
        body += payload
        body += b"\0" * ((-len(payload)) % tiny.ALIGNMENT)
    return bytes(header) + bytes(body)


# --- other decision types -------------------------------------------------


def adder(seed: int):
    """A table and the function that appends a random F32 tensor to it."""
    out: list[tuple[str, list[int], int, bytes]] = []
    state = {"seed": seed}

    def add(name: str, ne: list[int], scale: float, offset: float = 0.0) -> None:
        count = 1
        for dim in ne:
            count *= dim
        rng = tiny.Xorshift(state["seed"])
        values = [offset + scale * rng.unit() for _ in range(count)]
        out.append((name, ne, tiny.GGML_TYPE_F32, tiny.pack(values, tiny.GGML_TYPE_F32)))
        state["seed"] = (
            state["seed"] * 6364136223846793005 + 1442695040888963407
        ) & 0xFFFFFFFFFFFFFFFF

    return out, add


def tokenizer_metadata(tokens: list[str], types: list[int], bos: int, eos: int, unk: int):
    s = tiny.scalar
    e = tiny.encode_string
    # Longer pieces score higher, so the SPM merges reach them.
    scores = [
        float(len(token)) if kind == tiny.TOKEN_TYPE_NORMAL else 0.0
        for token, kind in zip(tokens, types, strict=True)
    ]
    return [
        ("tokenizer.ggml.model", tiny.T_STRING, e("llama")),
        ("tokenizer.ggml.tokens", tiny.T_ARRAY, tiny.string_array(tokens)),
        ("tokenizer.ggml.scores", tiny.T_ARRAY, tiny.numeric_array(tiny.T_FLOAT32, scores)),
        ("tokenizer.ggml.token_type", tiny.T_ARRAY, tiny.numeric_array(tiny.T_INT32, types)),
        ("tokenizer.ggml.bos_token_id", tiny.T_UINT32, s(tiny.T_UINT32, bos)),
        ("tokenizer.ggml.eos_token_id", tiny.T_UINT32, s(tiny.T_UINT32, eos)),
        ("tokenizer.ggml.unknown_token_id", tiny.T_UINT32, s(tiny.T_UINT32, unk)),
        ("tokenizer.ggml.add_bos_token", tiny.T_BOOL, s(tiny.T_BOOL, False)),
        ("tokenizer.ggml.add_eos_token", tiny.T_BOOL, s(tiny.T_BOOL, False)),
        ("tokenizer.ggml.add_space_prefix", tiny.T_BOOL, s(tiny.T_BOOL, False)),
    ]


TOKEN_TYPE_USER_DEFINED = 4

# The causal vocabulary: specials a decision template and its inputs spell,
# the bytes, then the pieces the readouts read as one token: the letters, the
# digits, yes and no, a few spaced letters (a label group of two tokens).
CAUSAL_SPECIALS = [
    ("<unk>", tiny.TOKEN_TYPE_UNKNOWN),
    ("<s>", tiny.TOKEN_TYPE_CONTROL),
    ("</s>", tiny.TOKEN_TYPE_CONTROL),
    ("<|im_start|>", tiny.TOKEN_TYPE_CONTROL),
    ("<|im_end|>", tiny.TOKEN_TYPE_CONTROL),
    ("<|box_start|>", tiny.TOKEN_TYPE_CONTROL),
    ("<|box_end|>", tiny.TOKEN_TYPE_CONTROL),
    ("<tool>", TOKEN_TYPE_USER_DEFINED),
]


def causal_vocabulary() -> tuple[list[str], list[int]]:
    tokens = [text for text, _ in CAUSAL_SPECIALS]
    types = [kind for _, kind in CAUSAL_SPECIALS]
    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        types.append(tiny.TOKEN_TYPE_BYTE)
    pieces = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    pieces += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    pieces += [str(d) for d in range(10)]
    pieces += [" ", "\u2581", "\u2581A", "\u2581B", "\u2581C", "ye", "yes", "Ye", "Yes", "no", "No"]
    for piece in pieces:
        if piece == " ":
            continue
        tokens.append(piece)
        types.append(tiny.TOKEN_TYPE_NORMAL)
    return tokens, types


def causal_trunk(add, n_vocab: int, arch: str) -> None:
    """A qwen2 (biases on q, k, v) or qwen3 (q and k norms) trunk."""
    n_embd, n_head, n_head_kv, n_ff, n_layer = 64, 4, 2, 128, 2
    head = n_embd // n_head
    add("token_embd.weight", [n_embd, n_vocab], 0.5)
    for il in range(n_layer):
        add(f"blk.{il}.attn_norm.weight", [n_embd], 0.1, 1.0)
        add(f"blk.{il}.attn_q.weight", [n_embd, n_embd], 0.2)
        add(f"blk.{il}.attn_k.weight", [n_embd, head * n_head_kv], 0.2)
        add(f"blk.{il}.attn_v.weight", [n_embd, head * n_head_kv], 0.2)
        if arch == "qwen2":
            add(f"blk.{il}.attn_q.bias", [n_embd], 0.05)
            add(f"blk.{il}.attn_k.bias", [head * n_head_kv], 0.05)
            add(f"blk.{il}.attn_v.bias", [head * n_head_kv], 0.05)
        else:
            add(f"blk.{il}.attn_q_norm.weight", [head], 0.1, 1.0)
            add(f"blk.{il}.attn_k_norm.weight", [head], 0.1, 1.0)
        add(f"blk.{il}.attn_output.weight", [n_embd, n_embd], 0.15)
        add(f"blk.{il}.ffn_norm.weight", [n_embd], 0.1, 1.0)
        add(f"blk.{il}.ffn_gate.weight", [n_embd, n_ff], 0.2)
        add(f"blk.{il}.ffn_up.weight", [n_embd, n_ff], 0.2)
        add(f"blk.{il}.ffn_down.weight", [n_ff, n_embd], 0.15)
    add("output_norm.weight", [n_embd], 0.1, 1.0)
    add("output.weight", [n_embd, n_vocab], 0.5)


def causal_metadata(arch: str, name: str) -> list:
    s = tiny.scalar
    e = tiny.encode_string
    kvs = [
        ("general.architecture", tiny.T_STRING, e(arch)),
        ("general.name", tiny.T_STRING, e(name)),
        ("general.file_type", tiny.T_UINT32, s(tiny.T_UINT32, tiny.FILE_TYPE_ALL_F32)),
        (f"{arch}.block_count", tiny.T_UINT32, s(tiny.T_UINT32, 2)),
        (f"{arch}.context_length", tiny.T_UINT32, s(tiny.T_UINT32, 4096)),
        (f"{arch}.embedding_length", tiny.T_UINT32, s(tiny.T_UINT32, 64)),
        (f"{arch}.feed_forward_length", tiny.T_UINT32, s(tiny.T_UINT32, 128)),
        (f"{arch}.attention.head_count", tiny.T_UINT32, s(tiny.T_UINT32, 4)),
        (f"{arch}.attention.head_count_kv", tiny.T_UINT32, s(tiny.T_UINT32, 2)),
        (f"{arch}.attention.layer_norm_rms_epsilon", tiny.T_FLOAT32, s(tiny.T_FLOAT32, NORM_EPS)),
        (f"{arch}.rope.freq_base", tiny.T_FLOAT32, s(tiny.T_FLOAT32, ROPE_FREQ_BASE)),
    ]
    if arch == "qwen2":
        kvs.append((f"{arch}.rope.dimension_count", tiny.T_UINT32, s(tiny.T_UINT32, 16)))
    return kvs


# The qwen35 trunk: block 0 a gated delta net, block 1 gated attention.
Q35_EMBD, Q35_FF, Q35_HEAD, Q35_HEAD_KV, Q35_HEAD_DIM = 64, 128, 4, 2, 16
Q35_SSM_STATE, Q35_SSM_GROUPS, Q35_SSM_RANK, Q35_SSM_CONV = 8, 2, 4, 4


def qwen35_trunk(add, n_vocab: int) -> None:
    n_embd = Q35_EMBD
    key_dim = Q35_SSM_STATE * Q35_SSM_GROUPS
    value_dim = Q35_SSM_STATE * Q35_SSM_RANK
    add("token_embd.weight", [n_embd, n_vocab], 0.5)
    for il in range(2):
        add(f"blk.{il}.attn_norm.weight", [n_embd], 0.1, 1.0)
        add(f"blk.{il}.post_attention_norm.weight", [n_embd], 0.1, 1.0)
        if il == 0:
            add(f"blk.{il}.attn_qkv.weight", [n_embd, key_dim * 2 + value_dim], 0.2)
            add(f"blk.{il}.attn_gate.weight", [n_embd, value_dim], 0.2)
            add(f"blk.{il}.ssm_conv1d.weight", [Q35_SSM_CONV, key_dim * 2 + value_dim], 0.3)
            add(f"blk.{il}.ssm_dt.bias", [Q35_SSM_RANK], 0.1)
            add(f"blk.{il}.ssm_a", [Q35_SSM_RANK], 0.4, -1.0)
            add(f"blk.{il}.ssm_beta.weight", [n_embd, Q35_SSM_RANK], 0.2)
            add(f"blk.{il}.ssm_alpha.weight", [n_embd, Q35_SSM_RANK], 0.2)
            add(f"blk.{il}.ssm_norm.weight", [Q35_SSM_STATE], 0.1, 1.0)
            add(f"blk.{il}.ssm_out.weight", [value_dim, n_embd], 0.15)
        else:
            add(f"blk.{il}.attn_q.weight", [n_embd, Q35_HEAD_DIM * Q35_HEAD * 2], 0.2)
            add(f"blk.{il}.attn_k.weight", [n_embd, Q35_HEAD_DIM * Q35_HEAD_KV], 0.2)
            add(f"blk.{il}.attn_v.weight", [n_embd, Q35_HEAD_DIM * Q35_HEAD_KV], 0.2)
            add(f"blk.{il}.attn_output.weight", [Q35_HEAD_DIM * Q35_HEAD, n_embd], 0.15)
            add(f"blk.{il}.attn_q_norm.weight", [Q35_HEAD_DIM], 0.1, 1.0)
            add(f"blk.{il}.attn_k_norm.weight", [Q35_HEAD_DIM], 0.1, 1.0)
        add(f"blk.{il}.ffn_gate.weight", [n_embd, Q35_FF], 0.2)
        add(f"blk.{il}.ffn_up.weight", [n_embd, Q35_FF], 0.2)
        add(f"blk.{il}.ffn_down.weight", [Q35_FF, n_embd], 0.15)
    add("output_norm.weight", [n_embd], 0.1, 1.0)


def qwen35_metadata(arch: str, name: str) -> list:
    s = tiny.scalar
    e = tiny.encode_string
    sections = struct.pack("<IQ4i", tiny.T_INT32, 4, 2, 1, 1, 0)
    recurrent = struct.pack("<IQ2?", tiny.T_BOOL, 2, True, False)
    return [
        ("general.architecture", tiny.T_STRING, e(arch)),
        ("general.name", tiny.T_STRING, e(name)),
        ("general.file_type", tiny.T_UINT32, s(tiny.T_UINT32, tiny.FILE_TYPE_ALL_F32)),
        (f"{arch}.block_count", tiny.T_UINT32, s(tiny.T_UINT32, 2)),
        (f"{arch}.context_length", tiny.T_UINT32, s(tiny.T_UINT32, 4096)),
        (f"{arch}.embedding_length", tiny.T_UINT32, s(tiny.T_UINT32, Q35_EMBD)),
        (f"{arch}.feed_forward_length", tiny.T_UINT32, s(tiny.T_UINT32, Q35_FF)),
        (f"{arch}.attention.head_count", tiny.T_UINT32, s(tiny.T_UINT32, Q35_HEAD)),
        (f"{arch}.attention.head_count_kv", tiny.T_UINT32, s(tiny.T_UINT32, Q35_HEAD_KV)),
        (f"{arch}.attention.key_length", tiny.T_UINT32, s(tiny.T_UINT32, Q35_HEAD_DIM)),
        (f"{arch}.attention.value_length", tiny.T_UINT32, s(tiny.T_UINT32, Q35_HEAD_DIM)),
        (f"{arch}.attention.layer_norm_rms_epsilon", tiny.T_FLOAT32, s(tiny.T_FLOAT32, NORM_EPS)),
        (f"{arch}.rope.dimension_count", tiny.T_UINT32, s(tiny.T_UINT32, 8)),
        (f"{arch}.rope.dimension_sections", tiny.T_ARRAY, sections),
        (f"{arch}.rope.freq_base", tiny.T_FLOAT32, s(tiny.T_FLOAT32, ROPE_FREQ_BASE)),
        (f"{arch}.ssm.conv_kernel", tiny.T_UINT32, s(tiny.T_UINT32, Q35_SSM_CONV)),
        (f"{arch}.ssm.state_size", tiny.T_UINT32, s(tiny.T_UINT32, Q35_SSM_STATE)),
        (f"{arch}.ssm.group_count", tiny.T_UINT32, s(tiny.T_UINT32, Q35_SSM_GROUPS)),
        (f"{arch}.ssm.time_step_rank", tiny.T_UINT32, s(tiny.T_UINT32, Q35_SSM_RANK)),
        (f"{arch}.ssm.inner_size", tiny.T_UINT32, s(tiny.T_UINT32, Q35_SSM_STATE * Q35_SSM_RANK)),
        (f"{arch}.attention.recurrent_layers", tiny.T_ARRAY, recurrent),
        (f"{arch}.full_attention_interval", tiny.T_UINT32, s(tiny.T_UINT32, 2)),
    ]


# The `[q|k]` projection of a pointer readout: its width is even.
KEV_EMBD_OUT = 32
# The width and the FFN of the decision heads.
HEAD_EMBD, HEAD_FF = 32, 64


def clef_head(add) -> None:
    n_embd, h, ff = Q35_EMBD, HEAD_EMBD, HEAD_FF

    def norm(name: str, size: int) -> None:
        add(f"{name}.weight", [size], 0.1, 1.0)
        add(f"{name}.bias", [size], 0.05)

    def attn(prefix: str, il: int) -> None:
        for part in ("q", "k", "v", "o"):
            add(f"dec.blk.{il}.{prefix}_{part}.weight", [h, h], 0.25)
            add(f"dec.blk.{il}.{prefix}_{part}.bias", [h], 0.05)

    for il in range(2):  # one routing block, one joint block
        if il == 0:
            norm(f"dec.blk.{il}.cross_attn_norm_kv", h)
        else:
            norm(f"dec.blk.{il}.attn_norm", h)
            attn("attn", il)
        norm(f"dec.blk.{il}.cross_attn_norm", h)
        attn("cross_attn", il)
        norm(f"dec.blk.{il}.ffn_norm", h)
        add(f"dec.blk.{il}.ffn_up.weight", [h, ff], 0.25)
        add(f"dec.blk.{il}.ffn_up.bias", [ff], 0.05)
        add(f"dec.blk.{il}.ffn_down.weight", [ff, h], 0.2)
        add(f"dec.blk.{il}.ffn_down.bias", [h], 0.05)
    norm("decision.hidden_norm", n_embd)
    norm("decision.option_summary_norm", h)
    norm("decision.field_norm", h)
    norm("decision.option_norm", h)
    for proj in (
        "memory",
        "question",
        "option_question",
        "global",
        "option_context",
        "option_lexical",
    ):
        add(f"decision.proj_{proj}.weight", [n_embd, h], 0.2)
    add("decision.scales", [3], 0.5, 1.0)
    add("token_types.weight", [h, 3], 0.5)
    add("decision.scorer.weight", [4 * h, h], 0.2)
    add("decision.scorer.bias", [h], 0.05)
    add("decision.scorer_out.weight", [h, 1], 0.5)
    add("decision.scorer_out.bias", [1], 0.05)


def vllm_decision2_head(add) -> None:
    n_embd, h = 64, HEAD_EMBD
    for name in ("option_norm", "query_norm"):
        add(f"decision.{name}.weight", [n_embd], 0.1, 1.0)
        add(f"decision.{name}.bias", [n_embd], 0.05)
    add("decision.key.weight", [n_embd, h], 0.3)
    add("decision.query.weight", [n_embd, h], 0.3)
    add("decision.option_mlp.weight", [n_embd, h], 0.3)
    add("decision.option_mlp.bias", [h], 0.05)
    add("decision.query_mlp.weight", [n_embd, h], 0.3)
    add("decision.scorer_out.weight", [h, 1], 0.5)


# The `systemone` templates of the fixtures. Values that are not text are
# written as JSON, as a real template does.
TEXT = "{{ X if X is string else X | tojson }}"


def text(name: str) -> str:
    return TEXT.replace("X", name)


LABEL_OPTIONS = (
    "{% for o in options %}"
    "{{ o.label if o.label is defined else "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'[loop.index0] }}) "
    "{{ o.key }}{% if o.description is not none %}: " + text("o.description") + "{% endif %}\n"
    "{% endfor %}"
)

LABEL_TEMPLATE = (
    "<|im_start|>" + text("state") + "\n"
    "Q ({{ type }}): " + text("instructions") + "\n" + LABEL_OPTIONS + "<|im_end|>Answer: "
)

NIMBLE_TEMPLATE = (
    "<|im_start|>" + text("state") + "\n"
    "{% for q in questions %}[{{ q.id }}] {{ q.type }}: "
    "{{ q.instructions if q.instructions is string else q.instructions | tojson }}\n"
    "{% for o in q.options %}  {{ o.label }}) {{ o.key }}\n{% endfor %}{% endfor %}"
    "Now {{ id }} ({{ type }}): "
    + text("instructions")
    + "\n"
    + LABEL_OPTIONS
    + "<|im_end|>Answer: "
)

H2O_TEMPLATE = (
    "<|im_start|>{{ state }}\n"
    "{{ instructions if instructions else 'Pick the best option.' }}\n"
    "{% for o in options %}{{ o.label }}. {{ o.key }}"
    "{% if o.description is not none %} - {{ o.description }}{% endif %}\n{% endfor %}"
    "<|im_end|>The answer is "
)

KEV_TEMPLATE = (
    "<|im_start|>{{ state }}<|im_end|>{{ instructions }}"
    "{% for o in options %}<|box_start|>"
    "{% if type == 'score' %}{{ o.description if o.description else o.key }}"
    "{% elif type == 'noul' %}{{ 'yes' if o.key == 'true' else 'no' }}"
    "{% else %}{{ o.key }}{% if o.description %}: {{ o.description }}{% endif %}{% endif %}"
    "<|box_end|>{% endfor %}<|im_start|>"
)

# Every span the joint head reads is followed by a piece it does not read, so
# that two spans never touch.
JOINT_TEMPLATE = (
    "<|im_start|>" + text("state") + "{{ sep }}\n"
    "{% for q in questions %}{{ sep }}{{ mark_question }}{{ q.type }}: "
    "{{ q.instructions if q.instructions is string else q.instructions | tojson }}{{ sep }}\n"
    "{% for o in q.options %}{{ sep }}{{ mark_option }}{{ o.key }}"
    "{% if o.description is not none %}: "
    "{{ o.description if o.description is string else o.description | tojson }}{% endif %}"
    "{{ sep }};{% endfor %}{% endfor %}{{ sep }}<|im_end|>"
)

D1OMNI_TEMPLATE = (
    "{% for i in images %}{{ i }}{% endfor %}{{ sep }}<bos>"
    "{{ sep }}{{ mark_question }}{{ type }} question: "
    + text("instructions")
    + "{% for o in options %}{{ sep }}{{ mark_option }}<mask> {{ o.key }}"
    "{% if o.description %}: " + text("o.description") + "{% endif %}{% endfor %}"
    "{{ sep }}<eos>{{ sep }}{{ mark_state }}" + text("state") + "{{ sep }}<eos>"
)

# Type -> (trunk, template, extra `<arch>.decision.*` metadata).
TYPES = {
    "openjev": ("qwen2", LABEL_TEMPLATE),
    "lev": ("qwen2", LABEL_TEMPLATE),
    "nimble": ("qwen2", NIMBLE_TEMPLATE),
    "vllm-d3": ("qwen2", LABEL_TEMPLATE),
    "lfm2-d1": ("qwen2", LABEL_TEMPLATE),
    "lfm3-d1": ("qwen2", LABEL_TEMPLATE),
    "h2o-lightning": ("qwen2", H2O_TEMPLATE),
    "vllm-decision2": ("vllm-decision2", JOINT_TEMPLATE),
    "clef": ("clef", JOINT_TEMPLATE),
    "kev": ("qwen35", KEV_TEMPLATE),
    "lfm2-d1-omni": ("modern-bert", D1OMNI_TEMPLATE),
}

H2O_LABELS = ["A", "B", "C", "D", "E", "F"]


def decision_metadata(arch: str, kind: str, template: str) -> list:
    s = tiny.scalar
    e = tiny.encode_string
    kvs = [
        (f"{arch}.decision.type", tiny.T_STRING, e(kind)),
        (f"{arch}.decision.temperature.choice", tiny.T_FLOAT32, s(tiny.T_FLOAT32, 1.25)),
        (f"{arch}.decision.temperature.choice.2", tiny.T_FLOAT32, s(tiny.T_FLOAT32, 0.75)),
        (f"{arch}.decision.temperature.noul", tiny.T_FLOAT32, s(tiny.T_FLOAT32, 1.5)),
        (f"{arch}.decision.score_bias.3.1", tiny.T_FLOAT32, s(tiny.T_FLOAT32, 0.25)),
        ("tokenizer.chat_template.systemone", tiny.T_STRING, e(template)),
        ("tokenizer.chat_templates", tiny.T_ARRAY, tiny.string_array(["systemone"])),
        ("general.alignment", tiny.T_UINT32, s(tiny.T_UINT32, tiny.ALIGNMENT)),
    ]
    if kind == "h2o-lightning":
        kvs.append((f"{arch}.decision.labels", tiny.T_ARRAY, tiny.string_array(H2O_LABELS)))
    return kvs


def build_type(kind: str) -> bytes:
    trunk, template = TYPES[kind]
    name = f"retrograd-tiny-decision-{kind}"
    s = tiny.scalar
    if trunk == "modern-bert":
        # The encoder and head of the laya fixture, read with the d1-omni layout.
        kvs = [
            kv
            for kv in metadata()
            if not kv[0].startswith("tokenizer.chat_template")
            and kv[0] not in (f"{ARCHITECTURE}.decision.type", "general.alignment", "general.name")
        ]
        kvs.append(("general.name", tiny.T_STRING, tiny.encode_string(name)))
        kvs += decision_metadata(ARCHITECTURE, kind, template)
        return write_gguf(tensors(), kvs)

    tokens, types = causal_vocabulary()
    tokenizer = tokenizer_metadata(tokens, types, bos=1, eos=2, unk=0)
    if trunk in ("qwen2", "vllm-decision2"):
        arch = "qwen3" if trunk == "vllm-decision2" else "qwen2"
        table, add = adder(0x5DEECE66D2F1A3B7)
        causal_trunk(add, len(tokens), arch)
        kvs = causal_metadata(trunk, name)
        if trunk == "vllm-decision2":
            vllm_decision2_head(add)
            kvs.append(
                (
                    f"{trunk}.attention.layer_norm_epsilon",
                    tiny.T_FLOAT32,
                    s(tiny.T_FLOAT32, NORM_EPS),
                )
            )
        return write_gguf(table, kvs + tokenizer + decision_metadata(trunk, kind, template))

    table, add = adder(0xA0761D6478BD642F)
    qwen35_trunk(add, len(tokens))
    arch = "clef" if trunk == "clef" else "qwen35"
    kvs = qwen35_metadata(arch, name)
    if trunk == "clef":
        clef_head(add)
        kvs += [
            (f"{arch}.decision.routing_block_count", tiny.T_UINT32, s(tiny.T_UINT32, 1)),
            (f"{arch}.decision.block_count", tiny.T_UINT32, s(tiny.T_UINT32, 1)),
            (f"{arch}.decision.head_count", tiny.T_UINT32, s(tiny.T_UINT32, 2)),
            (f"{arch}.attention.layer_norm_epsilon", tiny.T_FLOAT32, s(tiny.T_FLOAT32, NORM_EPS)),
        ]
    else:
        add("cls.output.weight", [Q35_EMBD, KEV_EMBD_OUT], 0.3)
        add("cls.output.bias", [KEV_EMBD_OUT], 0.05)
        kvs.append((f"{arch}.embedding_length_out", tiny.T_UINT32, s(tiny.T_UINT32, KEV_EMBD_OUT)))
    return write_gguf(table, kvs + tokenizer + decision_metadata(arch, kind, template))


def main(argv: list[str]) -> int:
    args = argv[1:]
    kind = "laya"
    if len(args) == 3 and args[0] == "--type":
        kind = args[1]
        args = args[2:]
    if len(args) != 1 or (kind != "laya" and kind not in TYPES):
        print(__doc__, file=sys.stderr)
        print("types: laya, " + ", ".join(TYPES), file=sys.stderr)
        return 2
    destination = Path(args[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build() if kind == "laya" else build_type(kind))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
