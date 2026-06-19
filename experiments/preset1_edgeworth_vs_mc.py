"""使い捨て: プリセット1 (_BLUEARCHIVE_7HIT_CARDS = 会心65.27%・7発) で
MC 経験分布と Edgeworth 展開 2次のみを比較するプロット。

uv run python -m experiments.preset1_edgeworth_vs_mc
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.edgeworth_animation import build_all_hits, edgeworth_pdf_cdf  # noqa: E402
from experiments.cos_compare import Scenario, mc_survival_samples  # noqa: E402
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402

N_MC = 20_000_000
SEED = 42
N_GRID = 400
N_BINS = 120
OUTPUT = "experiments/output/preset1_edgeworth_vs_mc.png"

# シナリオ構築 (λ3..λ6 まで自動計算)
hit_mixtures, _ = build_all_hits(
    _BLUEARCHIVE_7HIT_CARDS,
    global_crit=65.27, global_evade=0.0, damage_mode="post_decay",
)
sc = Scenario("メイドアリスTL", hit_mixtures)
print(f"Hit数={sc.n_hits}  平均={sc.mean:,.0f}  σ={sc.std:,.0f}")
print(f"λ3={sc.lam3:.3f}  λ4={sc.lam4:.3f}  λ5={sc.lam5:.3f}  λ6={sc.lam6:.3f}")

# MC サンプリング (真値)
print(f"MC サンプリング {N_MC:,} ...")
samples = mc_survival_samples(hit_mixtures, N_MC, SEED)
samples.sort()
n = samples.size

# x グリッド (サポート内・中心 ±5σ)
x_lo = max(sc.support_lo + 1e-6 * (sc.support_hi - sc.support_lo),
           sc.mean - 5.0 * sc.std)
x_hi = min(sc.support_hi - 1e-6 * (sc.support_hi - sc.support_lo),
           sc.mean + 5.0 * sc.std)
xs = np.linspace(x_lo, x_hi, N_GRID)
z = (xs - sc.mean) / sc.std

EDGE_LABEL = "Edgeworth 2次"
EDGE_COLOR = "tab:red"
pdf_edge, cdf_edge = edgeworth_pdf_cdf(
    xs, sc.mean, sc.std, sc.lam3, sc.lam4, 2, sc.lam5, sc.lam6,
)
pdf_edge_std = pdf_edge * sc.std       # 標準化密度

# MC 経験 CDF と両側裾確率
cdf_mc = np.searchsorted(samples, xs, side="right") / n
tail_mc = np.minimum(cdf_mc, 1.0 - cdf_mc)
near = np.minimum(
    np.searchsorted(samples, xs, side="right"),
    n - np.searchsorted(samples, xs, side="right"),
)
reliable = near >= 50

# MC 経験密度 (標準化)
z_samp = (samples - sc.mean) / sc.std
edges = np.linspace(z[0], z[-1], N_BINS + 1)
counts, _ = np.histogram(z_samp, bins=edges)
bin_w = edges[1] - edges[0]
mc_g = counts / (n * bin_w)
centers = 0.5 * (edges[:-1] + edges[1:])

# 裾の相対誤差 (信頼領域のみ)
tail_edge = np.minimum(cdf_edge, 1.0 - cdf_edge)
EPS = 1e-300
rel_tail = (tail_edge - tail_mc) / (tail_mc + EPS)
rel_tail_plot = np.where(reliable & (tail_mc > 0), rel_tail, np.nan)

# プロット (3段: 裾 → 裾相対誤差 → 密度)
fig, (ax_tail, ax_tail_err, ax_pdf) = plt.subplots(
    3, 1, figsize=(11, 11), sharex=True,
    gridspec_kw={"height_ratios": [1.2, 0.7, 1.2]},
)
fig.suptitle(
    f"{sc.name}: MC vs {EDGE_LABEL}\n"
    f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
    fontsize=12,
)

# 1段目: 両側裾確率
ax_tail.plot(z, np.where(tail_edge > 0, tail_edge, np.nan),
             color=EDGE_COLOR, lw=1.6, label=EDGE_LABEL)
ax_tail.plot(z[reliable], tail_mc[reliable], "k.", ms=3,
             label=f"MC ({n:,} 件)")
ax_tail.set_yscale("log")
ax_tail.set_ylim(max(1.0 / n / 10, 1e-7), 1.0)
ax_tail.set_ylabel("両側裾確率 min(F, 1−F)")
ax_tail.set_title("裾確率")
ax_tail.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
ax_tail.grid(True, which="both", alpha=0.3)
ax_tail.legend(loc="lower center", fontsize=9)

# 2段目: 裾の相対誤差
ax_tail_err.plot(z, rel_tail_plot, color=EDGE_COLOR, lw=1.4,
                 label=f"({EDGE_LABEL} − MC) / MC")
ax_tail_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
ax_tail_err.set_ylim(-1.0, 1.0)
ax_tail_err.set_ylabel("裾の相対誤差")
ax_tail_err.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
ax_tail_err.grid(True, alpha=0.3)
ax_tail_err.legend(loc="upper center", fontsize=9)

# 3段目: 標準化密度
ax_pdf.bar(centers, mc_g, width=bin_w * 0.95, alpha=0.3, color="gray",
           label=f"MC 経験密度 ({n:,} 件)")
ax_pdf.plot(z, pdf_edge_std, color=EDGE_COLOR, lw=1.6, label=EDGE_LABEL)
ax_pdf.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
ax_pdf.set_xlabel("z = (x − 平均) / 標準偏差")
ax_pdf.set_ylabel("密度 (標準化)")
ax_pdf.set_title("密度")
ax_pdf.grid(True, alpha=0.3)
ax_pdf.legend(loc="upper right", fontsize=9)

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"保存: {OUTPUT}")
