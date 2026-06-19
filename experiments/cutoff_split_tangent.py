"""足切り関門 d の「通過率」と「通過後の達成確率」フロンティア (docs/cutoff.md)。

チェックポイント m と目標 D に対し、関門 d (= 途中の累積ダメージの下限) を固定して
リセット運用 (途中 < d なら捨て、途中 ≥ d のみ続行) を考える。増分独立性
(X = 途中 = S_m, Y = 残り = T, X ⊥ Y) のもとで、次の 2 量を d の関数として描く。

    x(d) = P(X ≥ d)                       … 通過率 (関門を超える確率)
    y(d) = P(達成 | X ≥ d)
         = P(X + Y ≥ D | X ≥ d)
         = P(Y ≥ D - X | X ≥ d)           … 通過後の達成確率 (区間条件付け)

ここで条件は「実現値 X」に対するもので、しきい値は D - X (D - d ではない)。
両者の積は同時確率そのもの:

    x(d)·y(d) = P(X ≥ d, 達成) = P(達成, 途中 ≥ d)   (= 1 試行あたりの絶対成功率)

d を上げると通過率 x を犠牲に通過後成功率 y が上がる、というトレードオフ
(効率フロンティア) になる。フロンティア自体は ρ に依らないが、リセット運用で
「どの点 (= どの関門 d) を選ぶか」は時間比 ρ = (チェックポイント到達時間)/(フル
1試行時間) で決まる:

    コスト/成功 ∝ [ρ + (1-ρ)x] / (x·y)   を最小化する d* が最適点。

各パネルは 1 本のフロンティアに、ρ をいくつか選んでその最適点と無差別曲線
(一定コスト/成功; 最小コストでフロンティアに接する) を重ねたもの。

左: メイドアリス7射 (7Hit, m=3), D=37,000,000 (1非会心のみ許容)。
右: ビナーミカ (22Hit, m=11), D=9,500,000 (分布バルク内, P≈43%)。
すべて COS のみで計算 (条件付けは増分独立性で前半 PDF × 後半 CDF の 1 次元求積)。

uv run python -m experiments.cutoff_split_tangent
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401  (副作用で日本語フォントを設定)
except ImportError:
    from matplotlib import font_manager
    _avail = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ("Noto Sans CJK JP", "IPAexGothic", "HackGen35 Console NF"):
        if _name in _avail:
            plt.rcParams["font.family"] = _name
            break

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.cos import (  # noqa: E402
    HPParams,
    build_hit_mixtures,
    build_product_dist,
    build_sum_dist,
    y_mixture,
)
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402
from experiments.product_cos_app import PRESETS, _hp0_to_base_cards  # noqa: E402

N_GRID = 20001
X_MIN = 0.005          # 通過率がこれ未満の領域は条件付けが不安定なので描画から除く
D_SUM = 37_000_000.0   # メイドアリス: 1非会心のみ許容
D_PROD = 9_500_000.0   # ビナーミカ: 分布バルク内 (P≈43%)
# 重ね描きする時間比 ρ = (チェックポイントまでの所要時間)/(フル1試行の所要時間)。
# ρ ごとに「コスト/成功 = [ρ+(1-ρ)x]/(x·y)」を最小化する最適点と無差別曲線を描く。
RHO_LIST = [0.1, 0.3, 0.5, 0.7]
OUTPUT = "experiments/output/cutoff_split_tangent.png"


def curves_sum(front, back, D, dg):
    """和モデル: x=P(X≥d), y=P(達成|X≥d), 同時=P(達成,X≥d), 無条件 P(達成)。
    同時確率 = ∫_{x≥d} P(Y ≥ D-x) f_X(x) dx を右からの累積求積で全 d 一括評価。"""
    s = np.linspace(front.support_lo, front.support_hi, N_GRID)
    w = front.pdf(s) * (1.0 - back.cdf(D - s))
    ds = s[1] - s[0]
    cells = 0.5 * (w[1:] + w[:-1]) * ds
    W_right = np.concatenate([np.cumsum(cells[::-1])[::-1], [0.0]])  # ∫_s^{hi} w
    joint = np.interp(dg, s, W_right)                # P(達成, X≥d)
    x = 1.0 - front.cdf(dg)                          # P(X≥d)
    y = np.clip(joint / np.maximum(x, 1e-300), 0.0, 1.0)
    return x, y, joint, float(W_right[0])


RHO_COLORS = plt.cm.plasma(np.linspace(0.08, 0.78, len(RHO_LIST)))


def rho_optima(x, y, joint, dg):
    """ρ ごとに コスト/成功 ∝ [ρ+(1-ρ)x]/(x·y) を最小化する (d*, x*, y*, C) を返す。"""
    msk = x >= X_MIN
    xm, ym, jm, dm = x[msk], y[msk], joint[msk], dg[msk]
    out = []
    for rho in RHO_LIST:
        obj = (rho + (1.0 - rho) * xm) / np.maximum(jm, 1e-300)  # ∝ コスト/成功
        i = int(np.argmin(obj))
        out.append((rho, dm[i], xm[i], ym[i], obj[i]))
    return out


def plot_d_curves(ax, dg, x, y, optima, title, legend_loc="center right"):
    """左: 関門 d を横軸に通過率 x(d) と通過後達成 y(d)。ρ 別最適 d* を縦線で重ねる。"""
    msk = x >= X_MIN
    dgm, xm, ym = dg[msk], x[msk], y[msk]
    ax.plot(dgm / 1e6, xm, color="tab:blue", lw=2, label="通過率 x = P(途中>=d)")
    ax.plot(dgm / 1e6, ym, color="tab:green", lw=2,
            label="通過後達成 y = P(達成|途中>=d)")
    for (rho, d, _xs, _ys, _C), col in zip(optima, RHO_COLORS):
        ax.axvline(d / 1e6, color=col, ls=":", lw=1.3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("関門 d [百万]")
    ax.set_ylabel("確率")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc=legend_loc)


def plot_frontier_rho(ax, x, y, joint, dg, optima, title, legend_loc="lower left"):
    """右: フロンティア (x(d), y(d)) に ρ ごとの最適点と無差別曲線を重ねる。

    無差別曲線 (一定コスト/成功 C) は y = [ρ+(1-ρ)x]/(C x) で、最小 C のとき
    フロンティアに接する → その接点が最適点。"""
    msk = x >= X_MIN
    x, y = x[msk], y[msk]
    ax.plot(x, y, "-", color="black", lw=2.5, zorder=5,
            label="フロンティア (x(d), y(d))")
    xx = np.linspace(X_MIN, 1.0, 400)
    for (rho, d, xs, ys, C), col in zip(optima, RHO_COLORS):
        y_ind = (rho + (1.0 - rho) * xx) / (C * np.maximum(xx, 1e-12))
        ok = y_ind <= 1.03
        ax.plot(xx[ok], y_ind[ok], ls="--", lw=1.1, color=col, alpha=0.9)
        ax.plot([xs], [ys], "o", color=col, ms=9, zorder=6,
                label=f"ρ={rho:.1f}: d*={d/1e6:.1f}M (x*={xs:.2f}, y*={ys:.2f})")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("通過率 x = P(途中>=d)")
    ax.set_ylabel("通過後達成確率 y = P(達成|途中>=d)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc=legend_loc)


# =============================================================================
# 和モデル: メイドアリス7射 (7Hit), m = 3
# =============================================================================

print("=" * 72)
print("メイドアリス7射: _BLUEARCHIVE_7HIT_CARDS (会心65.27%, 7Hit), m = 3")
print("=" * 72)

hits = build_hit_mixtures(_BLUEARCHIVE_7HIT_CARDS, 65.27, 0.0, "post_decay")
m = 3
front = build_sum_dist(hits[:m])
back = build_sum_dist(hits[m:])
full = build_sum_dist(hits)

dg_sum = np.linspace(front.support_lo, front.support_hi, N_GRID)
x_s, y_s, j_s, P_s = curves_sum(front, back, D_SUM, dg_sum)
print(f"目標 D = {D_SUM:,.0f}  (無条件達成確率 P = {P_s:.4%})")

# =============================================================================
# 積モデル: ビナーミカ (22Hit), m = 11
# =============================================================================

print()
print("=" * 72)
preset_name = "ビナーミカ (R1=2, R0=1)"
print(f"ビナーミカ: PRESETS[{preset_name!r}], m = 11")
print("=" * 72)

pr_cfg = PRESETS[preset_name]
hp = HPParams(H=pr_cfg["H"], H1=pr_cfg["H1"], R0=pr_cfg["R0"], R1=pr_cfg["R1"])
cards = _hp0_to_base_cards(pr_cfg["cards"], pr_cfg["R0"], undo_decay=True)
base_per_hit = build_hit_mixtures(cards, pr_cfg["global_crit"],
                                  pr_cfg["global_evade"], "pre_decay")
ymix_per_hit = [y_mixture(b, hp.beta) for b in base_per_hit]
mp = 11
front_p = build_product_dist(ymix_per_hit[:mp], hp)
back_p = build_product_dist(ymix_per_hit[mp:], hp)
full_p = build_product_dist(ymix_per_hit, hp)
Htil = hp.Htil


def cdf_S(dist, s):
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    below, above = s < dist.a, s > dist.b
    mid = ~(below | above)
    out[below], out[above] = 0.0, 1.0
    if mid.any():
        out[mid] = dist._cdf_S(s[mid])
    return out


def s_of(v):
    return np.log1p(-np.asarray(v, dtype=float) / Htil)


def curves_prod(D, dg):
    """積モデル: 対数座標で D_m≥d ⟺ L_m≤s(d), 達成 ⟺ L_m+L'≤s(D)。
    同時 = ∫_{ℓ≤s(d)} F_{L'}(s(D)-ℓ) f_{L_m}(ℓ) dℓ を左からの累積求積で評価。"""
    l = np.linspace(front_p.a, front_p.b, N_GRID)
    w = front_p._pdf_S(l) * cdf_S(back_p, s_of(D) - l)
    dl = l[1] - l[0]
    cells = 0.5 * (w[1:] + w[:-1]) * dl
    W_left = np.concatenate([[0.0], np.cumsum(cells)])   # ∫_{A}^{ℓ} w
    sd = s_of(dg)
    joint = np.interp(sd, l, W_left)                 # P(達成, D_m≥d)
    x = cdf_S(front_p, sd)                            # P(D_m≥d)
    y = np.clip(joint / np.maximum(x, 1e-300), 0.0, 1.0)
    return x, y, joint, float(W_left[-1])


d_lo_p = -Htil * math.expm1(front_p.b)                # 前半の最小ダメージ
d_hi_p = -Htil * math.expm1(front_p.a)               # 前半の最大ダメージ
dg_prod = np.linspace(d_lo_p, d_hi_p, N_GRID)
x_p, y_p, j_p, P_p = curves_prod(D_PROD, dg_prod)
print(f"目標 D = {D_PROD:,.0f}  (無条件達成確率 P = {P_p:.4%})")

# =============================================================================
# プロット
# =============================================================================

opt_s = rho_optima(x_s, y_s, j_s, dg_sum)
opt_p = rho_optima(x_p, y_p, j_p, dg_prod)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plot_d_curves(axes[0, 0], dg_sum, x_s, y_s, opt_s,
              f"メイドアリス7射 (m=3/7, D={D_SUM/1e6:.0f}M): d ごとの確率",
              legend_loc="upper right")
plot_frontier_rho(axes[0, 1], x_s, y_s, j_s, dg_sum, opt_s,
                  "メイドアリス7射: フロンティアと ρ別最適点",
                  legend_loc="upper right")
plot_d_curves(axes[1, 0], dg_prod, x_p, y_p, opt_p,
              f"ビナーミカ (m=11/22, D={D_PROD/1e6:.1f}M): d ごとの確率")
plot_frontier_rho(axes[1, 1], x_p, y_p, j_p, dg_prod, opt_p,
                  "ビナーミカ: フロンティアと ρ別最適点")
fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=130)
print(f"\nプロット保存: {OUTPUT}")
