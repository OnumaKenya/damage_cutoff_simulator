"""ゲブラカリン (積/HP依存): 指数傾斜 Irwin–Hall 厳密解を基準に COS / MC を比較。

メイドアリスの preset1_irwinhall_compare.py に対応する、積モデル版。累積ダメージ
D = H̃_1(1 − Π Y_n) の分布を
    - 指数傾斜 Irwin–Hall 厳密解 (product_irwinhall_exact; 真値・基準)
    - COS 法 (特性関数の数値反転; 準厳密)
    - モンテカルロ (HP依存漸化式; 有限標本)
で比較する。これまで MC を基準にしていた誤差図を厳密解基準に置き換える。

    uv run python -m experiments.preset_gebura_irwinhall_compare
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.simulation import inverse_decay  # noqa: E402
from experiments.edgeworth_animation import build_all_hits  # noqa: E402
from experiments.product_cos import (  # noqa: E402
    damage_dist, damage_moments, mc_damage_hits, support_bounds_hits,
)
from experiments.product_cos_app import PRESETS  # noqa: E402
from experiments.product_irwinhall_exact import (  # noqa: E402
    build_ymix_for_preset, exact_damage_pdf_cdf_sf,
)

PRESET_NAME = "ゲブラカリン (R1=1, R0=3)"
N_MC = 20_000_000
SEED = 20260602
N_GRID = 600
N_BINS = 120
OUTPUT = "experiments/output/preset_gebura_irwinhall_compare.png"


def _base_per_hit(preset: dict) -> list:
    """MC 用に基礎ダメージ x の Hit 別混合を作る (hp0→基礎 x)。"""
    R0 = preset["R0"]
    cards = [dict(c) for c in preset["cards"]]
    if preset.get("hp0_input"):
        for c in cards:
            for fld in ("crit_min", "crit_max", "normal_min", "normal_max"):
                if c.get(fld) is not None:
                    c[fld] = int(round(inverse_decay(float(c[fld])) / R0))
        mode = "pre_decay"
    else:
        mode = preset.get("damage_mode", "post_decay")
    hm, _ = build_all_hits(cards, preset["global_crit"], preset["global_evade"], mode)
    return [[u for u in mix if u.weight > 0] for mix in hm]


def main() -> None:
    preset = PRESETS[PRESET_NAME]
    ymix, Htil, beta = build_ymix_for_preset(preset)
    n_hits = len(ymix)
    n_combo = 1
    for mix in ymix:
        n_combo *= len(mix)
    mean_D, var_D = damage_moments(ymix, Htil)
    std_D = math.sqrt(max(var_D, 0.0))
    print(f"[{PRESET_NAME}] Hit数={n_hits}, 組合せ={n_combo}, β={beta:.3e}, "
          f"H̃1={Htil:,.0f}")
    print(f"  平均D={mean_D:,.0f}  σD={std_D:,.0f}")

    # MC (真値の参照点)
    print(f"MC サンプリング {N_MC:,} ...")
    rng = np.random.default_rng(SEED)
    samples = mc_damage_hits(
        _base_per_hit(preset), preset["H"], preset["H1"],
        preset["R0"], preset["R1"], N_MC, rng,
    )
    samples.sort()
    n = samples.size
    d_lo, d_hi = float(samples[0]), float(samples[-1])

    # d グリッド (MC 範囲を少し内側に)
    pad = 1e-4 * (d_hi - d_lo)
    xs = np.linspace(d_lo + pad, d_hi - pad, N_GRID)
    z = (xs - mean_D) / std_D

    # --- 厳密解 (真値) ---
    f_ex, F_ex, SF_ex = exact_damage_pdf_cdf_sf(ymix, Htil, xs)
    tail_ex = np.minimum(F_ex, SF_ex)

    # --- COS 法 ---
    f_cos, F_cos = damage_dist(ymix, Htil, xs, verbose=True)
    tail_cos = np.minimum(F_cos, 1.0 - F_cos)

    # --- MC ---
    cdf_mc = np.searchsorted(samples, xs, side="right") / n
    tail_mc = np.minimum(cdf_mc, 1.0 - cdf_mc)
    near = np.minimum(np.searchsorted(samples, xs, side="right"),
                      n - np.searchsorted(samples, xs, side="right"))
    reliable = near >= 50

    # MC 経験密度 (標準化)
    z_samp = (samples - mean_D) / std_D
    edges = np.linspace(z[0], z[-1], N_BINS + 1)
    counts, _ = np.histogram(z_samp, bins=edges)
    bin_w = edges[1] - edges[0]
    mc_g = counts / (n * bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 厳密解基準の相対誤差 (両側裾)
    EPS = 1e-300
    rel_cos = (tail_cos - tail_ex) / (tail_ex + EPS)
    rel_mc = (tail_mc - tail_ex) / (tail_ex + EPS)
    rel_mc_plot = np.where(reliable & (tail_ex > 0), rel_mc, np.nan)

    # 厳密解は淡い太線ハローを最背面に、COS・MC を細線/点で上に重ねる。
    EX_C, COS_C, MC_C = "0.7", "magenta", "tab:green"
    fig, (ax_tail, ax_err, ax_pdf) = plt.subplots(
        3, 1, figsize=(11, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 0.8, 1.2]},
    )
    fig.suptitle(
        f"ゲブラカリン (積/HP依存): 指数傾斜 Irwin–Hall 厳密解 vs COS / MC\n"
        f"(Hit数={n_hits}, 組合せ={n_combo}, R0={preset['R0']}, R1={preset['R1']})",
        fontsize=12,
    )

    # 1段目: 両側裾確率
    ax_tail.plot(z, np.where(tail_ex > 0, tail_ex, np.nan),
                 color=EX_C, lw=5.0, solid_capstyle="round",
                 label="指数傾斜 Irwin–Hall 厳密解", zorder=1)
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
    ax_err.plot(z, rel_mc_plot, ".", color=MC_C, ms=2.5, label="MC − 厳密")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_ylim(-0.5, 0.5)
    ax_err.set_ylabel("裾の相対誤差\n(手法 − 厳密) / 厳密")
    ax_err.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper center", fontsize=9, ncol=2)

    # 3段目: 標準化密度
    ax_pdf.bar(centers, mc_g, width=bin_w * 0.95, alpha=0.25, color=MC_C,
               label=f"MC 経験密度 ({n:,} 件)", zorder=1)
    ax_pdf.plot(z, f_ex * std_D, color=EX_C, lw=5.0, solid_capstyle="round",
                label="指数傾斜 Irwin–Hall 厳密解", zorder=2)
    ax_pdf.plot(z, f_cos * std_D, color=COS_C, lw=1.5, label="COS 法", zorder=4)
    ax_pdf.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    ax_pdf.set_xlabel("z = (D − 平均) / 標準偏差")
    ax_pdf.set_ylabel("密度 (標準化)")
    ax_pdf.set_title("密度")
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.legend(loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {OUTPUT}")

    print("\n=== 厳密解基準の最大偏差 (グリッド全域) ===")
    print(f"  COS:        max|ΔF| = {np.max(np.abs(F_cos - F_ex)):.3e}")
    fin = reliable & (tail_ex > 0)
    if fin.any():
        print(f"  MC (信頼域): max|ΔF| = {np.max(np.abs(cdf_mc - F_ex)[fin]):.3e}")


if __name__ == "__main__":
    main()
