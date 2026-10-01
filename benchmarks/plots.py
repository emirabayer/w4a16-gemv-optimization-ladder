"""Generate both charts from benchmarks/results.json:
  - ladder.png   : horizontal bar of %peak per kernel (the headline story)
  - roofline.png : classic roofline, de-cluttered (only marlin + fp16 emphasized)

Reads measured JSON so charts and README share one source. Uses the largest
(most trustworthy) shape by default.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PEAK_BW = 1555.0
PEAK_FP16 = 78000.0


def pick(shapes, shape):
    if shape:
        r, c = map(int, shape.lower().split("x"))
        return next(s for s in shapes if s["rows"] == r and s["cols"] == c)
    return max(shapes, key=lambda s: s["rows"] * s["cols"])


def ladder_chart(entry, dev, out):
    ks = entry["kernels"]
    names = [k["name"] for k in ks]
    pct = [k["pct_peak"] for k in ks]
    # color: shipped=green, fp16 ref=gray, rejected/lossy=red-ish, ladder=blue
    colors = []
    for k in ks:
        n = k["name"]
        if n.startswith("marlin"): colors.append("#2a9d3f")
        elif "fp16" in n: colors.append("#888888")
        elif "rej" in n or "lossy" in n: colors.append("#c0503a")
        else: colors.append("#3a6ea5")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(names))[::-1]
    ax.barh(y, pct, color=colors)
    for yi, p in zip(y, pct):
        ax.text(p + 1, yi, f"{p:.0f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.axvline(PEAK_BW / PEAK_BW * 100, color="k", lw=0)  # no-op keep axis
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of peak memory bandwidth (1555 GB/s)")
    ax.set_title(f"w4a16 GEMV optimization ladder on {dev}\n"
                 f"{entry['rows']}x{entry['cols']} ,  green = shipped, gray = fp16 reference, red = rejected/lossy")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def roofline_chart(entry, dev, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    ai = np.logspace(-1, 2.5, 300)
    ax.plot(ai, np.minimum(PEAK_BW * ai, PEAK_FP16), "k-", lw=2, label="A100 roofline")
    # only annotate the two that matter; plot others as small dots without labels
    for k in entry["kernels"]:
        g = 2 * entry["rows"] * entry["cols"] / (k["ms"] * 1e-3) / 1e9
        if k["name"].startswith("marlin"):
            ax.plot(k["ai"], g, "*", ms=18, color="#2a9d3f")
            ax.annotate(f"marlin (shipped) {k['pct_peak']:.0f}%", xy=(k["ai"], g),
                        xytext=(k["ai"] * 1.15, g * 0.72), fontsize=9, color="#2a9d3f")
        elif "fp16" in k["name"]:
            ax.plot(k["ai"], g, "s", ms=10, color="#888888")
            ax.annotate(f"fp16 ref {k['pct_peak']:.0f}%", xy=(k["ai"], g),
                        xytext=(k["ai"] * 0.35, g * 1.25), fontsize=9, color="#555555")
        else:
            ax.plot(k["ai"], g, "o", ms=5, color="#b8c4d0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"w4a16 GEMV roofline on {dev}\n{entry['rows']}x{entry['cols']} "
                 f"(int4 kernels share AI=4; see ladder.png for the per-kernel comparison)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--results", default=os.path.join(here, "results.json"))
    ap.add_argument("--shape", default=None)
    args = ap.parse_args()
    data = json.load(open(args.results))
    entry = pick(data["shapes"], args.shape)
    ladder_chart(entry, data["device"], os.path.join(here, "ladder.png"))
    roofline_chart(entry, data["device"], os.path.join(here, "roofline.png"))


if __name__ == "__main__":
    main()
