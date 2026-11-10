#!/usr/bin/env python3
"""Generate the tiny deterministic MoE fixture used by the serving lanes.

The other tiny fixtures are dense (qwen2, laya). This one is an OLMoE: one
expert bank per layer (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`
indexed by the router's top-k through `build_moe_ffn`), the simplest routing
the fork serves - one bank per layer, no id remapping - which makes it the
small model a placement or caching change is first measured on.

Two router shapes are available, because a cache that learns is only worth
measuring on a routing that is concentrated:

- `--router concentrated` (the default): every embedding has a positive
  first coordinate, which the residual updates preserve. The router reads
  that coordinate with opposite signs for the hot and cold experts, so every
  token selects the first `n_expert_used` experts.
- `--router flat`: ordinary xorshift weights, with input-dependent routing.

The bytes depend on this file alone: the GGUF is written here rather than
through ``gguf-py``, and the weights come from an explicit xorshift rather
than a library PRNG, so the digest in ``tests/fixtures/TINY_MOE_FIXTURE.toml``
never drifts with a dependency.

Usage: scripts/gen-tiny-moe-fixture.py [--dtype f32|f16] [--router concentrated|flat]
                                       [--layers N] [--embd N] [--heads N]
                                       [--kv-heads N] [--ff N] [--experts N]
                                       [--experts-used N]
                                       <destination.gguf>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ARCHITECTURE = "olmoe"
MODEL_NAME = "retrograd-tiny-olmoe"

# The pinned shape. Overridable from the command line for a model that is not
# a unit fixture. The defaults are the pinned ones, so a run that names no
# shape writes the same bytes as before and the manifest digest still holds.
N_LAYER = 2
N_EMBD = 64
N_HEAD = 4
# The fork's OLMoE norms K while it is still 2D, so the key norm vector spans
# the full embedding: key/value heads must equal attention heads (MHA), as in
# the real checkpoints.
N_HEAD_KV = 4
N_FF = 64
N_EXPERT = 8
N_EXPERT_USED = 2
N_CTX_TRAIN = 256
RMS_EPS = 1.0e-5
ROPE_FREQ_BASE = 10000.0

N_EMBD_HEAD = N_EMBD // N_HEAD
N_EMBD_GQA = N_EMBD_HEAD * N_HEAD_KV


def configure(
    layers: int,
    embd: int,
    heads: int,
    heads_kv: int,
    ff: int,
    experts: int,
    experts_used: int,
) -> None:
    """Rebinds the shape and the dimensions derived from it."""
    global N_LAYER, N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_EXPERT, N_EXPERT_USED
    global N_EMBD_HEAD, N_EMBD_GQA
    if heads_kv != heads:
        raise SystemExit("OLMoE requires --kv-heads to match --heads")
    if embd % (2 * heads) != 0:
        raise SystemExit("embedding length must give each head an even width")
    if experts > 1024:
        raise SystemExit("OLMoE supports at most 1024 experts")
    if experts_used > experts:
        raise SystemExit("experts_used must not exceed experts")
    N_LAYER, N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_EXPERT, N_EXPERT_USED = (
        layers,
        embd,
        heads,
        heads_kv,
        ff,
        experts,
        experts_used,
    )
    N_EMBD_HEAD = N_EMBD // N_HEAD
    N_EMBD_GQA = N_EMBD_HEAD * N_HEAD_KV


ALIGNMENT = 32
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# llama.cpp's general.file_type for the variants.
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
    """`value` rounded to the nearest F16 and widened back to F32."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


GRIDS = {"f16": snap_to_f16}
EXACT_GRID = {GGML_TYPE_F16: "f16"}
GRID = "f16"


def weights(count: int, seed: int, scale: float, offset: float = 0.0) -> list[float]:
    rng = Xorshift(seed)
    snap = GRIDS[GRID]
    return [snap(offset + scale * rng.unit()) for _ in range(count)]


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


def router_weights(n_embd: int, n_expert: int, seed: int, shape: str) -> list[float]:
    """The `ffn_gate_inp` matrix, row-major over experts (ne = [n_embd, n_expert]).

    `concentrated` reads the preserved positive embedding coordinate with
    opposite signs for hot and cold experts.
    """
    if shape == "concentrated":
        row = []
        for expert in range(n_expert):
            row.append(0.0625 if expert < N_EXPERT_USED else -0.0625)
            row.extend([0.0] * (n_embd - 1))
        return row
    return weights(n_embd * n_expert, seed, 0.05)


def tensors(matrix_type: int, router_shape: str) -> list[tuple[str, list[int], int, bytes]]:
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
        values = weights(count, seed, scale, offset)
        if router_shape == "concentrated":
            if name == "token_embd.weight":
                values[::N_EMBD] = [0.125] * n_vocab
            elif name.endswith(".attn_output.weight"):
                values[:N_EMBD] = [0.0] * N_EMBD
            elif name.endswith(".ffn_down_exps.weight"):
                stride = N_FF * N_EMBD
                for start in range(0, count, stride):
                    values[start : start + N_FF] = [0.0] * N_FF
        out.append((name, ne, ggml_type, pack(values, ggml_type)))
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

    add("token_embd.weight", [N_EMBD, n_vocab], 0.08)
    for layer in range(N_LAYER):
        add(f"blk.{layer}.attn_norm.weight", [N_EMBD], 0.02, 1.0)
        add(f"blk.{layer}.attn_q.weight", [N_EMBD, N_EMBD], 0.08)
        add(f"blk.{layer}.attn_k.weight", [N_EMBD, N_EMBD_GQA], 0.08)
        add(f"blk.{layer}.attn_v.weight", [N_EMBD, N_EMBD_GQA], 0.08)
        add(f"blk.{layer}.attn_q_norm.weight", [N_EMBD], 0.02, 1.0)
        add(f"blk.{layer}.attn_k_norm.weight", [N_EMBD], 0.02, 1.0)
        add(f"blk.{layer}.attn_output.weight", [N_EMBD, N_EMBD], 0.08)
        add(f"blk.{layer}.ffn_norm.weight", [N_EMBD], 0.02, 1.0)
        out.append(
            (
                f"blk.{layer}.ffn_gate_inp.weight",
                [N_EMBD, N_EXPERT],
                matrix_type,
                pack(router_weights(N_EMBD, N_EXPERT, seed, router_shape), matrix_type),
            )
        )
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        add(f"blk.{layer}.ffn_gate_exps.weight", [N_EMBD, N_FF, N_EXPERT], 0.08)
        add(f"blk.{layer}.ffn_up_exps.weight", [N_EMBD, N_FF, N_EXPERT], 0.08)
        add(f"blk.{layer}.ffn_down_exps.weight", [N_FF, N_EMBD, N_EXPERT], 0.08)
    add("output_norm.weight", [N_EMBD], 0.02, 1.0)
    add("output.weight", [N_EMBD, n_vocab], 0.08)  # untied: its own tensor
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
        (f"{ARCHITECTURE}.expert_count", T_UINT32, scalar(T_UINT32, N_EXPERT)),
        (
            f"{ARCHITECTURE}.expert_used_count",
            T_UINT32,
            scalar(T_UINT32, N_EXPERT_USED),
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


def build(matrix_type: int, router_shape: str) -> bytes:
    table = tensors(matrix_type, router_shape)
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


def ggml_type_name(ggml_type: int) -> str:
    return {GGML_TYPE_F32: "f32", GGML_TYPE_F16: "f16"}[ggml_type]


SHAPE_FLAGS = {
    "--layers": "layers",
    "--embd": "embd",
    "--heads": "heads",
    "--kv-heads": "heads_kv",
    "--ff": "ff",
    "--experts": "experts",
    "--experts-used": "experts_used",
}


def main(argv: list[str]) -> int:
    global GRID
    matrix_type = GGML_TYPE_F32
    grid: str | None = None
    router_shape = "concentrated"
    shape = {
        "layers": N_LAYER,
        "embd": N_EMBD,
        "heads": N_HEAD,
        "ff": N_FF,
        "experts": N_EXPERT,
        "experts_used": N_EXPERT_USED,
    }
    argv = argv[1:]
    while argv and argv[0].startswith("--"):
        if argv[0] == "--":
            argv = argv[1:]
            break
        if len(argv) < 2:
            print(__doc__, file=sys.stderr)
            return 2
        flag, value = argv[0], argv[1]
        if flag == "--dtype":
            choice = value.lower()
            if choice not in ("f32", "f16"):
                print(__doc__, file=sys.stderr)
                return 2
            matrix_type = GGML_TYPE_F16 if choice == "f16" else GGML_TYPE_F32
        elif flag == "--grid":
            if value.lower() not in GRIDS:
                print(__doc__, file=sys.stderr)
                return 2
            grid = value.lower()
        elif flag == "--router":
            if value.lower() not in ("concentrated", "flat"):
                print(__doc__, file=sys.stderr)
                return 2
            router_shape = value.lower()
        elif flag in SHAPE_FLAGS:
            try:
                dimension = int(value)
                if not 1 <= dimension <= 0xFFFFFFFF:
                    raise ValueError()
            except ValueError:
                print(f"{flag} requires a positive uint32 integer", file=sys.stderr)
                return 2
            shape[SHAPE_FLAGS[flag]] = dimension
        else:
            print(__doc__, file=sys.stderr)
            return 2
        argv = argv[2:]
    # A storage that is a float grid of its own holds the values it was given
    # exactly or it holds different numbers. The default grid is the storage's
    # own where it has one.
    if grid is None:
        grid = EXACT_GRID.get(matrix_type, "f16")
    if EXACT_GRID.get(matrix_type, grid) != grid:
        print(
            f"--dtype {ggml_type_name(matrix_type)} cannot store a {grid} grid exactly",
            file=sys.stderr,
        )
        return 2
    GRID = grid
    shape.setdefault("heads_kv", shape["heads"])
    configure(**shape)
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    destination = Path(argv[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build(matrix_type, router_shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
