"""Benchmark the A100 w4a16 GEMV ladder.

Two measurement hazards this script guards against:
  1. Cold GPU clocks -> sustained matmuls warm them before timing.
  2. Launch overhead -> a fast kernel on an A100 can finish in ~16us, i.e. mostly
     the fixed cost of *starting* a kernel. We measure that launch floor directly
     and tag any shape whose runtime isn't comfortably above it as CONTAMINATED,
     and flag any >98%-of-peak reading as physically impossible (a contamination
     alarm). Only TRUST shapes are used to judge kernel quality.

Effective bandwidth uses matrix bytes only (dominates GEMV traffic). % of peak is
against the A100-SXM4-40GB's 1555 GB/s.

Usage:
    python benchmarks/benchmark.py            # -> benchmarks/results.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import w4a16 as K  # noqa: E402

PEAK_BW = 1555.0  # GB/s, A100-SXM4-40GB

# name, bytes-per-matrix-elem, arithmetic intensity, tag, packing
KERNELS = [
    ("byte",        0.5, 4.0, "byte",   "simple"),
    ("u32",         0.5, 4.0, "u32",    "simple"),
    ("u32+v4",      0.5, 4.0, "u32_v4", "simple"),
    ("champ",       0.5, 4.0, "champ",  "simple"),
    ("marlin",      0.5, 4.0, "marlin", "interleaved"),   # SHIPPED (per-row scale)
    ("marlin-g128", 0.5, 4.0, "marlin_g128", "interleaved"),  # group-128 scale
    ("uint4 (rej.)",0.5, 4.0, "uint4",  "simple"),
    ("h2f32 (lossy)",0.5,4.0, "h2f32",  "simple"),
    ("fp16 (ref)",  2.0, 1.0, "fp16",   "dense"),
]

# large decode shapes: big enough to clear the launch floor on A100.
SHAPES = [
    (128256, 2048, "lm_head (real)"),
    (131072, 2048, "synthetic"),
    (262144, 2048, "synthetic-lg"),
]


def warm_clocks():
    b = torch.randn(4096, 4096, device="cuda")
    for _ in range(60):
        b = b @ b * 1e-6 + 1.0
    torch.cuda.synchronize()


def bench(fn, passes=7, iters=200, warm=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(passes):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters)
    ts.sort()
    return ts[len(ts) // 2]


def call(tag, packed_s, packed_i, packed_g, scale, scale_g, vec, mat):
    if tag == "byte":   return lambda: K.matvec_byte(packed_s, scale, vec)
    if tag == "u32":    return lambda: K.matvec_u32(packed_s, scale, vec)
    if tag == "u32_v4": return lambda: K.matvec_u32_v4(packed_s, scale, vec)
    if tag == "champ":  return lambda: K.matvec_champ(packed_s, scale, vec)
    if tag == "marlin": return lambda: K.w4a16_gemv(packed_i, scale, vec)
    if tag == "marlin_g128": return lambda: K.w4a16_gemv_g128(packed_g, scale_g, vec)
    if tag == "uint4":  return lambda: K.matvec_uint4(packed_s, scale, vec)
    if tag == "h2f32":  return lambda: K.matvec_h2f32(packed_s, scale, vec)
    if tag == "fp16":   return lambda: K.matvec_fp16(mat, vec)
    raise ValueError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.cuda.get_device_name(0)
    print(f"device: {dev}   peak_bw: {PEAK_BW} GB/s")
    warm_clocks()

    # measure the launch floor: a trivially small kernel, with ALL inputs
    # pre-created OUTSIDE the timed region (timing must capture only the launch).
    torch.manual_seed(0)
    m0 = torch.randn(64, 2048, device="cuda").half()
    q0, s0 = K.quantize_int4_symmetric(m0.float())
    p0 = K.pack_int4(q0).cuda(); sc0 = s0.float().cuda()
    v0 = torch.randn(2048, device="cuda").half()
    floor = bench(lambda: K.matvec_byte(p0, sc0, v0))
    print(f"launch floor: {floor*1e3:.1f} us  (shapes must run >~8x this to be TRUST)\n")

    results = {"device": dev, "peak_bw_gbps": PEAK_BW, "launch_floor_us": floor * 1e3, "shapes": []}
    for rows, cols, label in SHAPES:
        mat = torch.randn(rows, cols, device="cuda").half()
        vec = torch.randn(cols, device="cuda").half()
        q, scale = K.quantize_int4_symmetric(mat.float())
        scale = scale.float().cuda()
        packed_s = K.pack_int4(q).cuda()
        packed_i = K.pack_int4_interleaved(q).cuda()
        qg, scale_g = K.quantize_int4_group(mat.float(), group=128)
        packed_g = K.pack_int4_interleaved(qg).cuda()
        scale_g = scale_g.float().cuda()

        entry = {"rows": rows, "cols": cols, "label": label, "kernels": []}
        print(f"=== {rows}x{cols}  ({label}) ===")
        print(f"{'kernel':15s} {'ms':>9s} {'GB/s':>8s} {'%peak':>7s} {'floor%':>7s} {'trust':>6s}")
        for name, bpe, ai, tag, _pk in KERNELS:
            fn = call(tag, packed_s, packed_i, packed_g, scale, scale_g, vec, mat)
            t = bench(fn)
            bw = rows * cols * bpe / (t * 1e-3) / 1e9
            pct = bw / PEAK_BW * 100
            floor_frac = floor / t * 100.0   # launch floor as % of this kernel's runtime
            trust = (floor_frac < 10.0) and (pct < 98.0)
            entry["kernels"].append({"name": name, "ai": ai, "ms": t, "eff_bw_gbps": bw,
                                     "pct_peak": pct, "floor_frac_pct": floor_frac, "trust": trust})
            print(f"{name:15s} {t:9.4f} {bw:8.1f} {pct:6.1f}% {floor_frac:6.1f}% {'yes' if trust else 'NO':>6s}")
        results["shapes"].append(entry)
        print()
        del mat, packed_s, packed_i
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
