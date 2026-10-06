"""End to end decode throughput: fp16 Llama-3.2-1B vs the int4-swapped model.

Config measured is the repo's as-is path: the int4 kernel runs on the decode step
(seq==1) and prefill uses the dense fp16 weight. Short prompt, long generation so
decode dominates. Prefill time and decode tokens/s are reported separately. Peak
GPU memory is reported for both; the swapped module keeps the dense fp16 weight
next to the packed int4 copy, so its memory is likely higher than baseline.

Usage:
    python benchmarks/end_to_end.py            # 16 prompt tokens, 256 new tokens
"""
import argparse
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import w4a16 as K  # noqa: E402


@torch.no_grad()
def run(model, input_ids, new_tokens, device):
    torch.cuda.synchronize()
    # prefill: process the prompt, get the first past_key_values
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # decode: one token at a time
    t1 = time.perf_counter()
    for _ in range(new_tokens):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1
    return prefill_s, decode_s


def measure(model, input_ids, new_tokens, device, warmup=1):
    for _ in range(warmup):
        run(model, input_ids, new_tokens, device)
    torch.cuda.reset_peak_memory_stats(device)
    pf, dc = run(model, input_ids, new_tokens, device)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    return pf, dc, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=256)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = "The history of computing hardware spans"
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    if ids.size(1) > args.prompt_tokens:
        ids = ids[:, :args.prompt_tokens]
    print(f"model: {args.model}   prompt_tokens: {ids.size(1)}   new_tokens: {args.new_tokens}\n")

    m16 = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    pf16, dc16, mem16 = measure(m16, ids, args.new_tokens, dev)
    tps16 = args.new_tokens / dc16
    print(f"fp16:    prefill {pf16*1e3:7.1f} ms   decode {tps16:7.1f} tok/s   peak mem {mem16:5.2f} GB")
    del m16; torch.cuda.empty_cache()

    msw = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    K.replace_linear_with_w4a16(msw)
    pfsw, dcsw, memsw = measure(msw, ids, args.new_tokens, dev)
    tpssw = args.new_tokens / dcsw
    print(f"int4:    prefill {pfsw*1e3:7.1f} ms   decode {tpssw:7.1f} tok/s   peak mem {memsw:5.2f} GB")

    print(f"\ndecode speedup (int4 / fp16): {tpssw/tps16:.2f}x")
    print(f"peak memory ratio (int4 / fp16): {memsw/mem16:.2f}x")
    if memsw > mem16:
        print("note: int4 model uses MORE memory (keeps dense fp16 weight + packed int4 copy).")


if __name__ == "__main__":
    main()
