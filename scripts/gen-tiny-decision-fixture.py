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

Usage: scripts/gen-tiny-decision-fixture.py <destination.gguf>
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
    table = tensors()
    kvs = metadata()
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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    destination = Path(argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
