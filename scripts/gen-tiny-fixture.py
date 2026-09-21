#!/usr/bin/env python3
"""Generate the tiny deterministic fixtures used by the CPU lanes.

The other CPU fixture is a 230M Q4_K_M download: quantized, and it ties its
vocabulary projection to the input embedding. This file produces the
complementary model: ``output.weight`` a tensor of its own, ``output.bias``
present, and no quantized tensor anywhere.

It writes that model at several storage precisions holding the *same numbers*:
every weight is generated, then snapped to a grid, and every variant of one
family is snapped to the *same* grid. The F32 file stores the snapped values
widened back to F32; the F16 and BF16 files store them verbatim; the Q8_0 file
quantizes them block by block. The only difference between the files of a
family is the precision the weights are stored at, so a gradient - or a score -
that differs between runs differs because of precision and for no other reason.
That is what makes the Q8_0 file usable as a *quantized anchor* against its own
F32 twin: the two models are the same model.

There are two families, because the two grids are not nested: BF16 keeps 8
significand bits against F16's 11, so an F16-grid value is not a BF16 value.
`--grid f16` (the default) writes the F32/F16/Q8_0 family; `--grid bf16`, which
`--dtype bf16` selects on its own, writes the BF16 family and its own F32
control. Each family is exact within itself; nothing compares across them.

Norms and biases stay F32 in every variant: they are one-dimensional,
llama.cpp expects F32 there, and a real F16 GGUF keeps them F32 too.

The bytes depend on this file alone: the GGUF is written here rather than
through ``gguf-py``, and the weights come from an explicit xorshift rather
than a library PRNG, so the digests in the ``tests/fixtures/*_FIXTURE.toml``
manifests never drift with a dependency.

Usage: scripts/gen-tiny-fixture.py [--dtype f32|f16|bf16|q8_0] [--grid f16|bf16]
                                   [--layers N] [--embd N] [--heads N]
                                   [--kv-heads N] [--ff N]
                                   <destination.gguf>

The shape flags default to the pinned fixture's, so omitting them writes the
bytes the manifests record. They exist for the models that are not unit
fixtures - an optimizer campaign needs a real matrix count, not four layers.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

ARCHITECTURE = "qwen2"
MODEL_NAME = "retrograd-tiny-qwen2"

# The pinned shape. Overridable from the command line for a model that is not a
# unit fixture - an optimizer whose cost per eligible matrix is the question
# needs hundreds of matrices, and four layers of sixty-four is not that. The
# defaults are the pinned ones, so a run that names no shape writes the same
# bytes as before and the digests in the manifests still hold.
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


def configure(layers: int, embd: int, heads: int, heads_kv: int, ff: int) -> None:
    """Rebinds the shape and the two dimensions derived from it."""
    global N_LAYER, N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_EMBD_HEAD, N_EMBD_GQA
    if embd % heads != 0:
        raise SystemExit("embedding length must be a multiple of the head count")
    N_LAYER, N_EMBD, N_HEAD, N_HEAD_KV, N_FF = layers, embd, heads, heads_kv, ff
    N_EMBD_HEAD = N_EMBD // N_HEAD
    N_EMBD_GQA = N_EMBD_HEAD * N_HEAD_KV


ALIGNMENT = 32
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
# retro delta: the one quantized storage this generator writes. Q8_0 because it
# is the simplest block format ggml has - 32 values, one F16 scale, no
# sub-blocks and no importance matrix - so the twin of an F32 file can be
# written here rather than through a quantizer, which is what makes the two
# files hold the same numbers at two precisions rather than two models.
GGML_TYPE_Q8_0 = 8
QUANT_K_Q8_0 = 32
GGML_TYPE_BF16 = 30

# llama.cpp's general.file_type for the variants.
FILE_TYPE_ALL_F32 = 0
FILE_TYPE_MOSTLY_F16 = 1
FILE_TYPE_MOSTLY_Q8_0 = 7
FILE_TYPE_MOSTLY_BF16 = 32

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
    """`value` rounded to the nearest F16 and widened back to F32."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def bf16_bits(value: float) -> int:
    """The top 16 bits of `value`, round-to-nearest-even on the tie.

    `ggml_compute_fp32_to_bf16`, restated: struct has no BF16 format, and the
    rounding has to be ggml's or a value would not survive the round trip
    through the loader it is read by.
    """
    (bits,) = struct.unpack("<I", struct.pack("<f", value))
    if bits & 0x7FFFFFFF > 0x7F800000:
        return ((bits >> 16) | 64) & 0xFFFF  # NaN, forced quiet
    return ((bits + (0x7FFF + ((bits >> 16) & 1))) >> 16) & 0xFFFF


def snap_to_bf16(value: float) -> float:
    """`value` rounded to the nearest BF16 and widened back to F32."""
    return struct.unpack("<f", struct.pack("<I", bf16_bits(value) << 16))[0]


# The grid every weight is snapped to before storage, and the storage that
# holds a grid value exactly. A family shares one grid; the two grids are not
# nested, so nothing is ever compared across them.
GRIDS = {"f16": snap_to_f16, "bf16": snap_to_bf16}
EXACT_GRID = {GGML_TYPE_F16: "f16", GGML_TYPE_BF16: "bf16"}
GRID = "f16"


def weights(count: int, seed: int, scale: float, offset: float = 0.0) -> list[float]:
    rng = Xorshift(seed)
    snap = GRIDS[GRID]
    return [snap(offset + scale * rng.unit()) for _ in range(count)]


def quantize_q8_0(values: list[float]) -> bytes:
    """ggml's Q8_0, restated: 32 values per block, one F16 scale, int8 codes.

    `d = amax/127` and `q = round(x/d)`, which is `quantize_row_q8_0_ref` in
    ggml-quants.c. Written here for the same reason the weights are: the file's
    bytes must be a property of this script, not of whichever quantizer happened
    to be on the machine.
    """
    if len(values) % QUANT_K_Q8_0 != 0:
        raise SystemExit("q8_0 needs a row length that is a multiple of 32")
    out = bytearray()
    for start in range(0, len(values), QUANT_K_Q8_0):
        block = values[start : start + QUANT_K_Q8_0]
        amax = max(abs(value) for value in block)
        d = amax / 127.0
        inverse = 1.0 / d if d != 0.0 else 0.0
        out += struct.pack("<e", d)
        for value in block:
            # round-half-away-from-zero, which is C's roundf and not Python's
            # banker's rounding.
            scaled = value * inverse
            code = math.floor(abs(scaled) + 0.5)
            code = min(code, 127)
            out += struct.pack("<b", int(code if scaled >= 0.0 else -code))
    return bytes(out)


def pack(values: list[float], ggml_type: int) -> bytes:
    if ggml_type == GGML_TYPE_Q8_0:
        return quantize_q8_0(values)
    if ggml_type == GGML_TYPE_BF16:
        return struct.pack(f"<{len(values)}H", *(bf16_bits(value) for value in values))
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
    kvs = metadata(
        {
            GGML_TYPE_F16: FILE_TYPE_MOSTLY_F16,
            GGML_TYPE_Q8_0: FILE_TYPE_MOSTLY_Q8_0,
            GGML_TYPE_BF16: FILE_TYPE_MOSTLY_BF16,
        }.get(matrix_type, FILE_TYPE_ALL_F32)
    )
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


def ggml_type_name(ggml_type: int) -> str:
    return {
        GGML_TYPE_F32: "f32",
        GGML_TYPE_F16: "f16",
        GGML_TYPE_Q8_0: "q8_0",
        GGML_TYPE_BF16: "bf16",
    }[ggml_type]


SHAPE_FLAGS = {
    "--layers": "layers",
    "--embd": "embd",
    "--heads": "heads",
    "--kv-heads": "heads_kv",
    "--ff": "ff",
}


def main(argv: list[str]) -> int:
    global GRID
    matrix_type = GGML_TYPE_F32
    grid: str | None = None
    shape = {
        "layers": N_LAYER,
        "embd": N_EMBD,
        "heads": N_HEAD,
        "heads_kv": N_HEAD_KV,
        "ff": N_FF,
    }
    argv = argv[1:]
    while len(argv) >= 2 and argv[0].startswith("--"):
        flag, value = argv[0], argv[1]
        if flag == "--dtype":
            choice = value.lower()
            if choice not in ("f32", "f16", "bf16", "q8_0"):
                print(__doc__, file=sys.stderr)
                return 2
            matrix_type = {
                "f16": GGML_TYPE_F16,
                "bf16": GGML_TYPE_BF16,
                "q8_0": GGML_TYPE_Q8_0,
            }.get(choice, GGML_TYPE_F32)
        elif flag == "--grid":
            if value.lower() not in GRIDS:
                print(__doc__, file=sys.stderr)
                return 2
            grid = value.lower()
        elif flag in SHAPE_FLAGS:
            shape[SHAPE_FLAGS[flag]] = int(value)
        else:
            print(__doc__, file=sys.stderr)
            return 2
        argv = argv[2:]
    # A storage that is a float grid of its own holds the values it was given
    # exactly or it holds different numbers, which is the one thing a control
    # pair may not do. The default grid is the storage's own where it has one.
    if grid is None:
        grid = EXACT_GRID.get(matrix_type, "f16")
    if EXACT_GRID.get(matrix_type, grid) != grid:
        print(
            f"--dtype {ggml_type_name(matrix_type)} cannot store a {grid} grid exactly",
            file=sys.stderr,
        )
        return 2
    GRID = grid
    configure(**shape)
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    destination = Path(argv[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build(matrix_type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
