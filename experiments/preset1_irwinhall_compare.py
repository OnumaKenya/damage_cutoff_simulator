"""メイドアリス TL: 一般化 Irwin–Hall 厳密解を基準に各手法を比較。

experiments.irwinhall_exact の閉形式 (区分多項式) を **真値** として、
    - COS 法 (特性関数の数値反転; 準厳密)
    - Edgeworth 展開 2次 (漸近近似)
    - モンテカルロ (有限標本)
の裾確率・密度・相対誤差を 1 枚にまとめる。これまで MC を基準にしていた誤差図を、
厳密解基準に置き換えることで「COS は機械精度で厳密に一致」「Edgeworth は裾で乖離」
「MC は標本誤差」を分離して可視化できる。

    uv run python -m experiments.preset1_irwinhall_compare
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
from experiments.cos_compare import (  # noqa: E402
    Scenario, cos_cdf, cos_pdf, mc_survival_samples,
)
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402
from experiments.irwinhall_exact import exact_cdf_sf_pdf  # noqa: E402

N_MC = 20_000_000
SEED = 42
N_GRID = 600
N_BINS = 120
OUTPUT = "experiments/output/preset1_irwinhall_compare.png"


def main() -> None:
    hit_mixtures, _ = build_all_hits(
        _BLUEARCHIVE_7HIT_CARDS,
        global_crit=65.27, global_evade=0.0, damage_mode="post_decay",
    )
    sc = Scenario("メイドアリスTL", hit_mixtures)
    n_combo = 1
    for cnt, mix in sc.grouped:
        n_combo *= len(mix) ** cnt
    print(f"Hit数={sc.n_hits}  平均={sc.mean:,.0f}  σ={sc.std:,.0f}")
    print(f"λ3={sc.lam3:.3f}  λ4={sc.lam4:.3f}  組合せ数={n_combo}")

    # x グリッド: サポート全域 (端は厳密に 0/1 になるので僅かに内側へ)
    pad = 1e-9 * (sc.support_hi - sc.support_lo)
    xs = np.linspace(sc.support_lo + pad, sc.support_hi - pad, N_GRID)
    z = (xs - sc.mean) / sc.std

    # --- 厳密解 (真値) ---
    # 両側裾は min(F, SF): 上側は反射 SF=F_Y(上端−x) で評価し、1−F の桁落ち
    # (上側裾での ~1e−7 ノイズ床・負確率) を回避する。
    cdf_exact, sf_exact, pdf_exact = exact_cdf_sf_pdf(hit_mixtures, xs)
    tail_exact = np.minimum(cdf_exact, sf_exact)

    # --- COS 法 (準厳密) ---
    cdf_cos = cos_cdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)
    pdf_cos = cos_pdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)
    tail_cos = np.minimum(cdf_cos, 1.0 - cdf_cos)

    # --- Edgeworth 2次 (漸近近似) ---
    pdf_edge, cdf_edge = edgeworth_pdf_cdf(
        xs, sc.mean, sc.std, sc.lam3, sc.lam4, 2, sc.lam5, sc.lam6,
    )
    tail_edge = np.minimum(cdf_edge, 1.0 - cdf_edge)

    # --- MC (有限標本) ---
    print(f"MC サンプリング {N_MC:,} ...")
    samples = mc_survival_samples(hit_mixtures, N_MC, SEED)
    samples.sort()
    n = samples.size
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

    # --- 厳密解基準の相対誤差 (両側裾) ---
    EPS = 1e-300
    rel_cos = (tail_cos - tail_exact) / (tail_exact + EPS)
    rel_edge = (tail_edge - tail_exact) / (tail_exact + EPS)
    rel_mc = (tail_mc - tail_exact) / (tail_exact + EPS)
    rel_mc_plot = np.where(reliable & (tail_exact > 0), rel_mc, np.nan)

    # 厳密解は「淡い太線のハロー」を最背面に敷き、COS・MC・Edgeworth を細線/点で
    # その上に重ねる (厳密解が太すぎて他が隠れるのを防ぐ)。
    EXACT_C = "0.7"          # 淡いグレー (基準のハロー)
    COS_C = "magenta"
    EDGE_C = "tab:red"
    MC_C = "tab:green"

    fig, (ax_tail, ax_err, ax_pdf) = plt.subplots(
        3, 1, figsize=(11, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 0.8, 1.2]},
    )
    fig.suptitle(
        f"{sc.name}: 一般化 Irwin–Hall 厳密解 vs COS / Edgeworth / MC\n"
        f"(Hit数={sc.n_hits}, 組合せ={n_combo}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
        fontsize=12,
    )

    # 1段目: 両側裾確率
    ax_tail.plot(z, np.where(tail_exact > 0, tail_exact, np.nan),
                 color=EXACT_C, lw=5.0, solid_capstyle="round",
                 label="Irwin–Hall 厳密解", zorder=1)
    ax_tail.plot(z, np.where(tail_edge > 0, tail_edge, np.nan),
                 color=EDGE_C, lw=1.5, label="Edgeworth 2次", zorder=2)
    ax_tail.plot(z, np.where(tail_cos > 0, tail_cos, np.nan),
                 color=COS_C, lw=1.5, label="COS 法 (準厳密)", zorder=3)
    ax_tail.plot(z[reliable], tail_mc[reliable], ".", color=MC_C, ms=3.5,
                 label=f"MC ({n:,} 件)", zorder=4)
    ax_tail.set_yscale("log")
    ax_tail.set_ylim(max(1.0 / n / 10, 1e-7), 1.0)
    ax_tail.set_ylabel("両側裾確率 min(F, 1−F)")
    ax_tail.set_title("裾確率 (厳密解が真値)")
    ax_tail.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_tail.grid(True, which="both", alpha=0.3)
    ax_tail.legend(loc="lower center", fontsize=9, ncol=2)

    # 2段目: 厳密解基準の相対誤差
    ax_err.plot(z, rel_cos, color=COS_C, lw=1.3, ls="--", label="COS − 厳密")
    ax_err.plot(z, rel_edge, color=EDGE_C, lw=1.4, label="Edgeworth − 厳密")
    ax_err.plot(z, rel_mc_plot, ".", color=MC_C, ms=2.5, label="MC − 厳密")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_ylim(-1.0, 1.0)
    ax_err.set_ylabel("裾の相対誤差\n(手法 − 厳密) / 厳密")
    ax_err.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper center", fontsize=9, ncol=3)

    # 3段目: 標準化密度
    ax_pdf.bar(centers, mc_g, width=bin_w * 0.95, alpha=0.25, color=MC_C,
               label=f"MC 経験密度 ({n:,} 件)", zorder=1)
    ax_pdf.plot(z, pdf_exact * sc.std, color=EXACT_C, lw=5.0,
                solid_capstyle="round", label="Irwin–Hall 厳密解", zorder=2)
    ax_pdf.plot(z, pdf_edge * sc.std, color=EDGE_C, lw=1.5,
                label="Edgeworth 2次", zorder=3)
    ax_pdf.plot(z, pdf_cos * sc.std, color=COS_C, lw=1.5,
                label="COS 法", zorder=4)
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

    # 数値サマリ
    print("\n=== 厳密解基準の最大偏差 (サポート内全域) ===")
    print(f"  COS:       max|ΔF| = {np.max(np.abs(cdf_cos - cdf_exact)):.3e}")
    print(f"  Edgeworth: max|ΔF| = {np.max(np.abs(cdf_edge - cdf_exact)):.3e}")
    fin = reliable & (tail_exact > 0)
    print(f"  MC (信頼域): max|ΔF| = "
          f"{np.max(np.abs(cdf_mc - cdf_exact)[fin]):.3e}")


if __name__ == "__main__":
    main()
