# w4a16: int4 weight-only GEMV kernels for LLM decode (CUDA, A100)

CUDA kernels for the matrix times vector product in single token LLM decoding,
with 4 bit weight quantization (int4 weights, fp16 activations). The repo keeps
the whole optimization ladder, including the versions that did not work, so each
step can be re-run. This is a kernel optimization study. The per-row int4 scheme
is fast and correct as a kernel but is not deployable as is: it costs quality at
this model size and the naive decode integration is slower and larger than fp16.
Group-wise scales recover most of the quality and are included as a second kernel
variant.

## What it is

In decode, each layer multiplies a weight matrix by a single activation vector.
This is memory bound: the runtime is dominated by reading the weights. Quantizing
the weights to int4 cuts the bytes read by 4x versus fp16. The kernels read int4
weights, dequantize them inside the kernel (no dequantized matrix is written to
memory), and multiply by an fp16 vector with a symmetric scale. Two scale schemes
are shipped: one scale per output row (per-row), and one scale per 128 input
weights per row (group-128).

## Results

Measured on an A100-SXM4-40GB, peak memory bandwidth 1555 GB/s, with
meta-llama/Llama-3.2-1B for the quality and end to end numbers. Latency is the
median of several passes with GPU clocks warmed. Percent of peak uses matrix bytes
against 1555 GB/s (fp16 reads 2 bytes per weight, int4 reads 0.5).

### Latency vs cuBLAS

cuBLAS is the fp16 `F.linear` path at batch 1. per-row and group-128 are the two
shipped int4 kernels.

| shape | cuBLAS fp16 | per-row int4 | group-128 int4 |
|---|---|---|---|
| 128256 x 2048 (lm_head) | 0.3723 ms (90.7%) | 0.1074 ms (78.7%) | 0.1239 ms (68.2%) |
| 131072 x 2048 | 0.3795 ms (91.0%) | 0.1096 ms (78.7%) | 0.1265 ms (68.3%) |
| 262144 x 2048 (largest) | 0.7454 ms (92.6%) | 0.2142 ms (80.6%) | 0.2471 ms (69.9%) |
| 8192 x 2048 | 0.0252 ms (85.7%) | 0.0158 ms (34.1%) | 0.0158 ms (34.2%) |

On the lm_head shape (128256 x 2048), per-row int4 runs in 0.1074 ms versus 0.3723
ms for cuBLAS, 3.5x. On the largest synthetic shape (262144 x 2048), per-row is
3.5x over cuBLAS and group-128 is 3.0x (0.7454 / 0.2471). The per-group scale
fetch costs group-128 about 13% latency over per-row. The repo's own fp16 kernel
(k_fp16) matches cuBLAS within about 1 percent on the large shapes, so the fp16
baseline is a real one. On the small shape (8192 x 2048) the runtime is partly
launch bound, per-row and group-128 are equal at about 34% of peak, so the large
shapes are the headline.

### Optimization ladder (per-row kernel)

Percent of the 1555 GB/s peak on the largest shape (262144 x 2048), standalone
GEMV. The ladder is profiler driven: each change targeted the bottleneck Nsight
Compute showed at that point.

| kernel | change | % of peak | max abs err vs reference |
|---|---|---|---|
| byte | warp per row, 1 byte per lane | 52% | 4e-5 |
| u32 | 32 bit load, 8 int4 per lane | 64% | 4e-5 |
| u32+v4 | 128 bit float4 vector load | 68% | 4e-5 |
| champ | fp16 bit trick dequant, fp32 accumulate | 72% | 4e-5 |
| marlin (shipped) | lop3 dequant, interleaved packing | 81% | 4e-5 |

The fp16 reference kernel (k_fp16) reaches 93% of peak on this shape. The int4
kernel is compute bound (the dequantize plus multiply arithmetic), which is why
per-row reaches 81% and group-128 about 70%. Rejected variants kept in the tree:
uint4 (128 bit weight load, 32 int4 per lane) at 19% of peak (too little work per
warp), and h2f32 (fp16 multiply accumulate) at 82% but with 6e-2 max error versus
4e-5.

### Quality: WikiText-2 perplexity

Full WikiText-2 test set, context 2048, stride 512. The int4 rows are measured by
replacing every Linear weight with its dequantized int4 value (the value the
kernel computes at sequence length 1) applied to every token. The group-128 kernel
is verified to reproduce those dequantized weights to about 1e-6 on real
Llama-3.2-1B layers (q_proj 1.91e-6, gate_proj 7.15e-7, down_proj 1.91e-6), so the
group-128 row is a kernel result.

| config | WikiText-2 perplexity |
|---|---|
| fp16 | 8.62 |
| int4 per-row | 31.62 |
| int4 per-row, lm_head in fp16 | 28.94 |
| int4 group-128, lm_head in fp16 | 11.17 |

Per-row symmetric int4 degrades quality severely on this model (8.62 to 31.62).
The cause is scale granularity: one scale per output row lets a single large weight
inflate the scale and cost resolution for the rest of the row. lm_head is a small
factor (31.62 to 28.94 with lm_head in fp16). Group-wise scales (one scale per 128
input weights) recover most of the quality (11.17 versus 8.62), which is what GPTQ,
AWQ, and the real Marlin use.

### End to end decode

int4 per-row kernel on the decode step (sequence length 1), dense fp16 weight on
prefill. 16 prompt tokens, 256 new tokens, so decode dominates. Prefill uses the
fp16 weight directly (no fp32 cast per call).

| model | decode tok/s | prefill ms | peak GPU memory |
|---|---|---|---|
| fp16 | 56.2 | 20.1 | 2.49 GB |
| int4 (per-row decode kernel) | 46.5 | 25.1 | 3.11 GB |

Decode is 0.83x (slower) and peak memory is 1.25x (higher) than fp16. Decode is
slower because per-token launch overhead dominates the per-layer bandwidth saving
at 1B on an A100. Memory is higher because the module keeps the dense fp16 weight
next to the packed int4 copy. The prefill fix brought prefill from 28.6 to 25.1 ms
and peak memory from 4.16 to 3.11 GB (1.67x to 1.25x); decode is unchanged by it.
This is a regression at this model size.

### The two variants

- per-row: fastest (3.5x over cuBLAS on the largest shape, 81% of peak), perplexity
  31.62, unusable.
- group-128: slightly slower (3.0x over cuBLAS, about 13% over per-row, 70% of
  peak), perplexity 11.17, close to fp16 8.62.

## How it works

The bottleneck was measured with Nsight Compute at each step and moved three times:
DRAM latency, then L1/TEX throughput, then compute. Each change targeted the
bottleneck the profiler showed at that point.

The shipped kernel decodes 8 packed int4 weights into four half2 values using the
fp16 bit pattern: `half(0x6400 | n) - 1032 == n - 8` for a 4 bit value n. The
nibble extraction uses four `lop3.b32` instructions instead of about 24 shift,
mask, and subtract operations. This needs an interleaved weight packing (slot
order 0,2,4,6,1,3,5,7 within each group of 8 columns) so the lop3 outputs come out
in column order. The group-128 kernel uses the same packing and lop3 dequant and
fetches one scale per 128 input weights per row, which is the fix for the per-row
quality loss at about 13% latency cost. The bit mapping is checked in Python and
both kernels are checked against a PyTorch reference in the test suite.

### What did not work

Kept in the tree with the measured reason:

- 128 bit weight load (uint4, 32 int4 per lane): 19% of peak. Each warp finished
  its row with too little work left, so there were too few memory requests in
  flight to cover latency.
- Shared memory staging of the activation vector: slower. The vector already hit
  in L1 at 94%, so staging it added a barrier with no bandwidth saving.
- Fast dequant applied before fixing the load path: slower. When the kernel is load
  bound, cheaper arithmetic is hidden behind the load wait, so it only helps after
  the load path is fixed.
- fp16 multiply accumulate (h2f32): 82% of peak but about 6e-2 max error versus
  4e-5. The fp16 multiply rounds each product before summing, which fp32
  accumulation cannot recover. Kept as an opt in variant.

### Why the per-row kernel stops at 81%

A probe that removes the unpack work (keeping the loads and the multiplies) reaches
about 85 to 88% of peak, and removing the multiplies alone reaches about 80%. So
after the lop3 dequant the kernel is near the fp32 multiply add floor, which is the
arithmetic of the dot product and cannot be removed. Closing the gap to the fp16
kernel's 93% would require fp16 multiplies, which cost about 1000x more error, so I
did not.

## Build and run (Colab, A100)

```bash
pip install -r requirements.txt       # ninja compiles the .cu at runtime
pytest -v -m "not slow"               # correctness suite
```

The CUDA source is in a .cu file, compiled at import with
`torch.utils.cpp_extension.load`, so there is no setup.py build step.

## Reproducibility

The full sequence that regenerates the benchmark JSON, the charts, and the tables:

```bash
pip install -r requirements.txt
pytest -v -m "not slow"               # correctness suite
python benchmarks/benchmark.py        # per-kernel GEMV -> benchmarks/results.json
python benchmarks/plots.py            # ladder.png and roofline.png from results.json
python benchmarks/perplexity.py       # WikiText-2 perplexity, fp16 vs int4 per-row
python benchmarks/end_to_end.py       # decode throughput and peak memory
```

## Correctness

The suite checks each ladder kernel (both packings) against a PyTorch reference at
`atol=1e-3`, the per-row and group-128 marlin kernels against dequantized
references, both packing roundtrips, an int4 quantization error band, and the
h2f32 variant staying outside the tight tolerance so its accuracy cost cannot
regress unnoticed. A slow test (`-m slow`) checks that all 113 `nn.Linear` layers
in Llama-3.2-1B swap to the int4 module. Max abs error versus a dequantized
reference is 4e-5 for the fp32 accumulate kernels; on real Llama layers the
group-128 kernel matches to about 1e-6.

## Limitations

- The module swap uses the per-row kernel. group-128 is provided as a second
  kernel variant (measured and tested) but is not wired into the module swap.
- per-row int4 is not deployable as a model: perplexity 31.62 versus fp16 8.62.
  group-128 recovers most of the quality (11.17) but is not wired into the swap.
- Decode only. The module uses the kernel at sequence length 1 and falls back to
  dense fp16 for prefill.
- End to end at 1B on an A100 is a regression: decode 0.83x, peak memory 1.25x.
- Quantization is symmetric. per-row needs the column count divisible by 8;
  group-128 needs it divisible by 128.
- Numbers are specific to the A100-SXM4-40GB. Effective bandwidth uses matrix bytes
  against 1555 GB/s.
