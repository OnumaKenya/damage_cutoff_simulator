"""メイドアリス TL: Irwin–Hall 厳密解を基準に CF/CGF 系 4 手法を比較。

preset1_irwinhall_compare.py (MC・Edgeworth 込み) は残したまま、本スクリプトは
特性関数・CGF ベースの 4 手法だけを並べる:

    - Irwin–Hall 厳密解 (区分多項式; 真値・基準)
    - COS 法           (特性関数の有界余弦級数反転; 準厳密)
    - Gil-Pelaez       (特性関数の半直線フーリエ反転; COS と数学的に同一)
    - Lugannani–Rice   (CGF のサドルポイント漸近近似)

狙い: (1) COS と Gil-Pelaez が厳密解と機械精度近くで一致し、互いに同一であること、
(2) LR は中腹で区分多項式構造を均す系統誤差を持つが深裾では桁落ちせず頑健なこと、
を 1 枚で可視化する。

    uv run python -m experiments.preset1_inversion_compare
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.edgeworth_animation import build_all_hits  # noqa: E402
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402
from experiments.saddlepoint_compare import (  # noqa: E402
    Scenario, cos_cdf, gil_pelaez_cdf, lr_sf_array,
)
from experiments.irwinhall_exact import exact_cdf_sf_pdf  # noqa: E402

N_GRID = 600
OUTPUT = "experiments/output/preset1_inversion_compare.png"


def main() -> None:
    hit_mixtures, _ = build_all_hits(
        _BLUEARCHIVE_7HIT_CARDS,
        global_crit=65.27, global_evade=0.0, damage_mode="post_decay",
    )
    sc = Scenario("メイドアリスTL", hit_mixtures)
    n_combo = 1
    for cnt, mix in sc.grouped:
        n_combo *= len(mix) ** cnt
    print(f"Hit数={sc.n_hits}  平均={sc.mean:,.0f}  σ={sc.std:,.0f}  組合せ={n_combo}")

    pad = 1e-9 * (sc.support_hi - sc.support_lo)
    xs = np.linspace(sc.support_lo + pad, sc.support_hi - pad, N_GRID)
    z = (xs - sc.mean) / sc.std

    # --- Irwin–Hall 厳密解 (真値) ---
    #   両側裾は min(F, SF) で上側の桁落ちを回避 (SF は反射で評価)。
    cdf_ex, sf_ex, _ = exact_cdf_sf_pdf(hit_mixtures, xs)
    tail_ex = np.minimum(cdf_ex, sf_ex)

    # --- COS 法 ---
    cdf_cos = cos_cdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)
    tail_cos = np.minimum(cdf_cos, 1.0 - cdf_cos)

    # --- Gil-Pelaez ---
    cdf_gp = gil_pelaez_cdf(sc.grouped, xs, sc.mean, sc.support_lo, sc.support_hi)
    tail_gp = np.minimum(cdf_gp, 1.0 - cdf_gp)

    # --- Lugannani–Rice (生存関数を直接、両側へ) ---
    sf_lr = lr_sf_array(sc.grouped, xs, sc.mean, sc.var)
    tail_lr = np.minimum(1.0 - sf_lr, sf_lr)

    # 相対誤差 (厳密解基準, 両側裾)
    EPS = 1e-300
    rel_cos = (tail_cos - tail_ex) / (tail_ex + EPS)
    rel_gp = (tail_gp - tail_ex) / (tail_ex + EPS)
    rel_lr = (tail_lr - tail_ex) / (tail_ex + EPS)

    # 厳密解は淡い太線ハローを最背面に、COS/GP/LR を細線でその上に重ねる。
    EX_C, COS_C, GP_C, LR_C = "0.7", "magenta", "tab:blue", "tab:red"

    fig, (ax_tail, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )
    fig.suptitle(
        f"{sc.name}: Irwin–Hall 厳密解 vs COS / Gil-Pelaez / Lugannani–Rice\n"
        f"(Hit数={sc.n_hits}, 組合せ={n_combo}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
        fontsize=12,
    )

    # 1段目: 両側裾確率
    ax_tail.plot(z, np.where(tail_ex > 0, tail_ex, np.nan),
                 color=EX_C, lw=5.0, solid_capstyle="round",
                 label="Irwin–Hall 厳密解", zorder=1)
    ax_tail.plot(z, np.where(tail_lr > 0, tail_lr, np.nan),
                 color=LR_C, lw=1.5, label="Lugannani–Rice", zorder=2)
    ax_tail.plot(z, np.where(tail_cos > 0, tail_cos, np.nan),
                 color=COS_C, lw=1.6, label="COS 法", zorder=3)
    ax_tail.plot(z, np.where(tail_gp > 0, tail_gp, np.nan),
                 color=GP_C, lw=1.2, ls=":", label="Gil-Pelaez", zorder=4)
    ax_tail.set_yscale("log")
    ax_tail.set_ylim(1e-12, 1.0)
    ax_tail.set_ylabel("両側裾確率 min(F, 1−F)")
    ax_tail.set_title("裾確率 (厳密解が真値)")
    ax_tail.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_tail.grid(True, which="both", alpha=0.3)
    ax_tail.legend(loc="lower center", fontsize=9, ncol=2)

    # 2段目: 厳密解基準の相対誤差
    ax_err.plot(z, rel_cos, color=COS_C, lw=1.4, ls="--", label="COS − 厳密")
    ax_err.plot(z, rel_gp, color=GP_C, lw=1.1, ls=":", label="Gil-Pelaez − 厳密")
    ax_err.plot(z, rel_lr, color=LR_C, lw=1.5, label="Lugannani–Rice − 厳密")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_ylim(-0.4, 0.4)
    ax_err.set_ylabel("裾の相対誤差\n(手法 − 厳密) / 厳密")
    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper center", fontsize=9, ncol=3)

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {OUTPUT}")

    # 数値サマリ
    def _maxrel(rel, lo, hi):
        m = (z > lo) & (z < hi) & (tail_ex > 1e-12)
        return np.max(np.abs(rel[m])) if m.any() else float("nan")
    print("\n=== 厳密解基準の最大相対誤差 ===")
    print(f"  中腹 1<|z|<2.5:  COS={_maxrel(rel_cos,1,2.5):.2e}  "
          f"GP={_maxrel(rel_gp,1,2.5):.2e}  LR={_maxrel(rel_lr,1,2.5):.2e}")
    print(f"  全域 |z|<2.5 :   COS={_maxrel(rel_cos,-2.5,2.5):.2e}  "
          f"GP={_maxrel(rel_gp,-2.5,2.5):.2e}  LR={_maxrel(rel_lr,-2.5,2.5):.2e}")
    print(f"  COS vs GP 最大差 |ΔF| = {np.max(np.abs(cdf_cos - cdf_gp)):.2e}")


if __name__ == "__main__":
    main()
