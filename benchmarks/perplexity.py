"""WikiText-2 perplexity: fp16 Llama-3.2-1B vs int4 per-row-scale weights.

Important: the swapped W4A16Linear only uses the int4 kernel at seq==1. A
perplexity run is seq>1, so it would hit the dense fp16 fallback and silently
measure fp16. To measure the int4 quality, this builds a plain model
whose every Linear weight is the dequantized int4 value (q * per_row_scale),
which is what the kernel computes at seq==1, applied to every token.

Sanity: fp16 ppl should match published Llama-3.2-1B numbers for the stride/ctx
used; int4 ppl must NOT equal fp16 exactly (that would mean the swap did nothing).

Usage:
    python benchmarks/perplexity.py            # ctx 2048, stride 512 (HF recipe)
"""
import argparse
import sys

import torch
import torch.nn as nn


def quant_dequant_per_row(w):
    scale = w.abs().max(dim=1, keepdim=True).values / 7.0
    q = torch.round(w / scale).clamp(-8, 7)
    return (q * scale).to(w.dtype)


def swap_to_int4_dense(model):
    # overwrite every Linear weight with its int4-dequantized value, kept dense,
    # so every token (seq>1 included) sees the quantized weights.
    for m in model.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.data = quant_dequant_per_row(m.weight.data.float()).to(m.weight.dtype)
    return model


@torch.no_grad()
def perplexity(model, enc, ctx, stride, device):
    # standard sliding-window negative-log-likelihood ppl (HF recipe)
    ids = enc.input_ids.to(device)
    n = ids.size(1)
    nlls = []
    total_tokens = 0
    prev = 0
    for begin in range(0, n, stride):
        end = min(begin + ctx, n)
        trg_len = end - prev
        inp = ids[:, begin:end]
        tgt = inp.clone()
        tgt[:, :-trg_len] = -100
        out = model(inp, labels=tgt)
        # out.loss is mean over the (trg_len) scored tokens
        nlls.append(out.loss.float() * trg_len)
        total_tokens += trg_len
        prev = end
        if end == n:
            break
    return torch.exp(torch.stack(nlls).sum() / total_tokens).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt")

    print(f"model: {args.model}   ctx: {args.ctx}   stride: {args.stride}")

    m16 = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    ppl16 = perplexity(m16, enc, args.ctx, args.stride, dev)
    print(f"fp16 perplexity:  {ppl16:.4f}")
    del m16; torch.cuda.empty_cache()

    m4 = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    swap_to_int4_dense(m4)
    ppl4 = perplexity(m4, enc, args.ctx, args.stride, dev)
    print(f"int4 perplexity:  {ppl4:.4f}  (per-row symmetric scale)")

    # sanity: the two must differ, or the swap silently did nothing
    if abs(ppl4 - ppl16) < 1e-6:
        print("ERROR: int4 == fp16 exactly. The quantization swap did not apply.", file=sys.stderr)
        sys.exit(1)
    print(f"\ndelta: int4 - fp16 = {ppl4 - ppl16:+.4f}  ({(ppl4/ppl16 - 1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
