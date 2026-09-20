#!/usr/bin/env python3
"""Generate the tiny deterministic fixtures used by the CPU lanes.

The other CPU fixture is a 230M Q4_K_M download: quantized, and it ties its
vocabulary projection to the input embedding. This file produces the
complementary model: ``output.weight`` a tensor of its own, ``output.bias``
present, and no quantized tensor anywhere.

It writes that model at two storage precisions holding the *same numbers*:
every weight is generated, then snapped to the F16 grid in both variants. The
F32 file stores the snapped values widened back to F32; the F16 file stores
them verbatim. The only difference between the two files is the precision the
weights are stored at, so a gradient that differs between runs differs because
of precision and for no other reason.

Norms and biases stay F32 in both: they are one-dimensional, llama.cpp expects
F32 there, and a real F16 GGUF keeps them F32 too.

The bytes depend on this file alone: the GGUF is written here rather than
through ``gguf-py``, and the weights come from an explicit xorshift rather
than a library PRNG, so the digests in ``tests/fixtures/TINY_FIXTURE.toml``
and ``tests/fixtures/TINY_F16_FIXTURE.toml`` never drift with a dependency.

Usage: scripts/gen-tiny-fixture.py [--dtype f32|f16] <destination.gguf>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ARCHITECTURE = "qwen2"
MODEL_NAME = "retrograd-tiny-qwen2"

N_LAYER = 4
N_EMBD = 64
N_HEAD = 4
N_HEAD_KV = 2
N_FF = 128
N_CTX_TRAIN = 256
RMS_EPS = 1.0e-5
ROPE_FREQ_BASE = 10000.0

N_EMBD_HEAD = N_EMBD // N_HEAD
N_EMBD_GQA = N_EMBD_HEAD * N_HEAD_KV

ALIGNMENT = 32
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# llama.cpp's general.file_type for the two variants.
FILE_TYPE_ALL_F32 = 0
FILE_TYPE_MOSTLY_F16 = 1

# GGUF metadata value types.
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9

# llama.cpp token types.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_BYTE = 6


class Xorshift:
    """xorshift64*, written out so the stream is a property of this file."""

    def __init__(self, seed: int) -> None:
        self.state = seed | 1

    def next_u64(self) -> int:
        state = self.state
        state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
        self.state = state & 0xFFFFFFFFFFFFFFFF
        return (self.state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def unit(self) -> float:
        """A float in [-1, 1), from the 24 high bits."""
        bits = self.next_u64() >> 40
        return (bits / float(1 << 24)) * 2.0 - 1.0


def snap_to_f16(value: float) -> float:
    """`value` rounded to the nearest F16 and widened back to F32.

    Applied to every weight in both variants so the two files hold the same
    numbers.
    """
    return struct.unpack("<e", struct.pack("<e", value))[0]


def weights(count: int, seed: int, scale: float, offset: float = 0.0) -> list[float]:
    rng = Xorshift(seed)
    return [snap_to_f16(offset + scale * rng.unit()) for _ in range(count)]


def pack(values: list[float], ggml_type: int) -> bytes:
    fmt = "e" if ggml_type == GGML_TYPE_F16 else "f"
    return struct.pack(f"<{len(values)}{fmt}", *values)


def vocabulary() -> tuple[list[str], list[int]]:
    """A byte-fallback SPM vocabulary: three specials and all 256 bytes.

    Byte fallback keeps any input tokenizable without a merge table.
    """
    tokens = ["<unk>", "<s>", "</s>"]
    types = [TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL]
    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        types.append(TOKEN_TYPE_BYTE)
    # One ordinary piece, so the vocabulary is not exclusively special-cased
    # and a merge of two bytes is reachable. U+2581 is SPM's space marker.
    tokens.append("▁the")
    types.append(TOKEN_TYPE_NORMAL)
    return tokens, types


def tensors(matrix_type: int) -> list[tuple[str, list[int], int, bytes]]:
    """Name, ggml `ne` (fastest dimension first), ggml type, and bytes.

    `matrix_type` is the storage of the matrices; vectors stay F32.
    """
    tokens, _ = vocabulary()
    n_vocab = len(tokens)
    out: list[tuple[str, list[int], int, bytes]] = []
    seed = 0x9E3779B97F4A7C15

    def add(name: str, ne: list[int], scale: float, offset: float = 0.0) -> None:
        nonlocal seed
        count = 1
        for dim in ne:
            count *= dim
        ggml_type = matrix_type if len(ne) > 1 else GGML_TYPE_F32
        out.append((name, ne, ggml_type, pack(weights(count, seed, scale, offset), ggml_type)))
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

    add("token_embd.weight", [N_EMBD, n_vocab], 0.08)
    for layer in range(N_LAYER):
        add(f"blk.{layer}.attn_norm.weight", [N_EMBD], 0.02, 1.0)
        add(f"blk.{layer}.attn_q.weight", [N_EMBD, N_EMBD], 0.08)
        add(f"blk.{layer}.attn_q.bias", [N_EMBD], 0.01)
        add(f"blk.{layer}.attn_k.weight", [N_EMBD, N_EMBD_GQA], 0.08)
        add(f"blk.{layer}.attn_k.bias", [N_EMBD_GQA], 0.01)
        add(f"blk.{layer}.attn_v.weight", [N_EMBD, N_EMBD_GQA], 0.08)
        add(f"blk.{layer}.attn_v.bias", [N_EMBD_GQA], 0.01)
        add(f"blk.{layer}.attn_output.weight", [N_EMBD, N_EMBD], 0.08)
        add(f"blk.{layer}.ffn_norm.weight", [N_EMBD], 0.02, 1.0)
        add(f"blk.{layer}.ffn_gate.weight", [N_EMBD, N_FF], 0.08)
        add(f"blk.{layer}.ffn_up.weight", [N_EMBD, N_FF], 0.08)
        add(f"blk.{layer}.ffn_down.weight", [N_FF, N_EMBD], 0.08)
    add("output_norm.weight", [N_EMBD], 0.02, 1.0)
    add("output.weight", [N_EMBD, n_vocab], 0.08)  # untied: its own tensor
    add("output.bias", [n_vocab], 0.01)
    return out


def write_string(out: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    out += struct.pack("<Q", len(encoded))
    out += encoded


def write_kv(out: bytearray, key: str, value_type: int, payload: bytes) -> None:
    write_string(out, key)
    out += struct.pack("<I", value_type)
    out += payload


def scalar(value_type: int, value) -> bytes:
    formats = {T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?"}
    return struct.pack(formats[value_type], value)


def string_array(values: list[str]) -> bytes:
    out = bytearray(struct.pack("<IQ", T_STRING, len(values)))
    for value in values:
        write_string(out, value)
    return bytes(out)


def numeric_array(value_type: int, values: list) -> bytes:
    formats = {T_INT32: "i", T_FLOAT32: "f"}
    return struct.pack(f"<IQ{len(values)}{formats[value_type]}", value_type, len(values), *values)


def metadata(file_type: int) -> list[tuple[str, int, bytes]]:
    tokens, types = vocabulary()
    return [
        ("general.architecture", T_STRING, encode_string(ARCHITECTURE)),
        ("general.name", T_STRING, encode_string(MODEL_NAME)),
        ("general.file_type", T_UINT32, scalar(T_UINT32, file_type)),
        (f"{ARCHITECTURE}.block_count", T_UINT32, scalar(T_UINT32, N_LAYER)),
        (f"{ARCHITECTURE}.context_length", T_UINT32, scalar(T_UINT32, N_CTX_TRAIN)),
        (f"{ARCHITECTURE}.embedding_length", T_UINT32, scalar(T_UINT32, N_EMBD)),
        (f"{ARCHITECTURE}.feed_forward_length", T_UINT32, scalar(T_UINT32, N_FF)),
        (f"{ARCHITECTURE}.attention.head_count", T_UINT32, scalar(T_UINT32, N_HEAD)),
        (
            f"{ARCHITECTURE}.attention.head_count_kv",
            T_UINT32,
            scalar(T_UINT32, N_HEAD_KV),
        ),
        (
            f"{ARCHITECTURE}.attention.layer_norm_rms_epsilon",
            T_FLOAT32,
            scalar(T_FLOAT32, RMS_EPS),
        ),
        (
            f"{ARCHITECTURE}.rope.dimension_count",
            T_UINT32,
            scalar(T_UINT32, N_EMBD_HEAD),
        ),
        (f"{ARCHITECTURE}.rope.freq_base", T_FLOAT32, scalar(T_FLOAT32, ROPE_FREQ_BASE)),
        ("tokenizer.ggml.model", T_STRING, encode_string("llama")),
        ("tokenizer.ggml.tokens", T_ARRAY, string_array(tokens)),
        (
            "tokenizer.ggml.scores",
            T_ARRAY,
            numeric_array(T_FLOAT32, [0.0] * len(tokens)),
        ),
        ("tokenizer.ggml.token_type", T_ARRAY, numeric_array(T_INT32, types)),
        ("tokenizer.ggml.bos_token_id", T_UINT32, scalar(T_UINT32, 1)),
        ("tokenizer.ggml.eos_token_id", T_UINT32, scalar(T_UINT32, 2)),
        ("tokenizer.ggml.unknown_token_id", T_UINT32, scalar(T_UINT32, 0)),
        ("tokenizer.ggml.add_bos_token", T_BOOL, scalar(T_BOOL, True)),
        ("tokenizer.ggml.add_eos_token", T_BOOL, scalar(T_BOOL, False)),
        ("general.alignment", T_UINT32, scalar(T_UINT32, ALIGNMENT)),
    ]


def encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def build(matrix_type: int) -> bytes:
    table = tensors(matrix_type)
    header = bytearray()
    kvs = metadata(FILE_TYPE_MOSTLY_F16 if matrix_type == GGML_TYPE_F16 else FILE_TYPE_ALL_F32)
    header += struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(table), len(kvs))
    for key, value_type, payload in kvs:
        write_kv(header, key, value_type, payload)

    offset = 0
    for name, ne, ggml_type, payload in table:
        write_string(header, name)
        header += struct.pack("<I", len(ne))
        for dim in ne:
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", ggml_type)
        header += struct.pack("<Q", offset)
        offset += (len(payload) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    padding = (-len(header)) % ALIGNMENT
    header += b"\0" * padding

    body = bytearray()
    for _, _, _, payload in table:
        body += payload
        body += b"\0" * ((-len(payload)) % ALIGNMENT)
    return bytes(header) + bytes(body)


def main(argv: list[str]) -> int:
    matrix_type = GGML_TYPE_F32
    argv = argv[1:]
    if len(argv) >= 2 and argv[0] == "--dtype":
        choice = argv[1].lower()
        if choice not in ("f32", "f16"):
            print(__doc__, file=sys.stderr)
            return 2
        matrix_type = GGML_TYPE_F16 if choice == "f16" else GGML_TYPE_F32
        argv = argv[2:]
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    destination = Path(argv[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build(matrix_type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
