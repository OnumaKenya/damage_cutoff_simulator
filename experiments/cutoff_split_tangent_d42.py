"""cutoff_split_tangent.py のメイドアリス深裾版: D = 42,000,000 固定。

S_n の台の上端 (≈43.0M) に近い深裾ターゲットなので、確率が桁で変わる。
左パネルは対数軸、右のパラメトリック表示は log-log にする。log-log 平面では
等積曲線 xy = c* が「傾き -1 の直線」になり、積最大の点で曲線 (x(d), y(d)) が
この直線に接する (接線条件 = 対数微分の釣り合い = ハザード率の一致)。

uv run python -m experiments.cutoff_split_tangent_d42
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
from app.backend.cos import build_hit_mixtures, build_sum_dist  # noqa: E402
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402

D_TARGET = 42_000_000.0
M_CHECK = 3
N_GRID = 20001
OUTPUT = "experiments/output/cutoff_split_tangent_d42.png"


def golden_max(f, lo, hi, iters=120):
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
    return 0.5 * (a + b)


print("=" * 72)
print(f"和モデル: _BLUEARCHIVE_7HIT_CARDS (会心65.27%, 7Hit), m = {M_CHECK}, "
      f"D = {D_TARGET:,.0f}")
print("=" * 72)

hits = build_hit_mixtures(_BLUEARCHIVE_7HIT_CARDS, 65.27, 0.0, "post_decay")
front = build_sum_dist(hits[:M_CHECK])
back = build_sum_dist(hits[M_CHECK:])
full = build_sum_dist(hits)

p_true = 1.0 - float(full.cdf(np.array([D_TARGET]))[0])
print(f"台の上端 (全Hit最大) = {full.support_hi:,.0f}")
print(f"無条件達成確率 P(S_n ≥ D) = {p_true:.6e}")

x_of = lambda d: 1.0 - front.cdf(d)            # noqa: E731  前: P(S_m ≥ d)
y_of = lambda d: 1.0 - back.cdf(D_TARGET - d)  # noqa: E731  後: P(T ≥ D-d)

# 積が正になりうる d の範囲: 前が届く ∧ 残りが足りる
d_lo = max(front.support_lo, D_TARGET - back.support_hi)
d_hi = min(front.support_hi, D_TARGET - back.support_lo)
print(f"関門の有効範囲: d ∈ [{d_lo:,.0f}, {d_hi:,.0f}] "
      f"(幅 {d_hi - d_lo:,.0f}; m={M_CHECK} 時点の最大 {front.support_hi:,.0f})")

dg = np.linspace(d_lo, d_hi, N_GRID)
x, y = x_of(dg), y_of(dg)
prod = x * y
i = int(np.argmax(prod))
dstar = golden_max(lambda t: float(x_of(np.array([t]))[0] * y_of(np.array([t]))[0]),
                   dg[max(i - 1, 0)], dg[min(i + 1, N_GRID - 1)])
xs = float(x_of(np.array([dstar]))[0])
ys = float(y_of(np.array([dstar]))[0])
cstar = xs * ys

eps = (d_hi - d_lo) * 1e-6
dlnx = (math.log(float(x_of(np.array([dstar + eps]))[0]))
        - math.log(float(x_of(np.array([dstar - eps]))[0]))) / (2 * eps)
dlny = (math.log(float(y_of(np.array([dstar + eps]))[0]))
        - math.log(float(y_of(np.array([dstar - eps]))[0]))) / (2 * eps)

print(f"\n最適関門 d* = {dstar:,.0f}")
print(f"  前 x* = P(S_m≥d*) = {xs:.6e}   後 y* = P(T≥D-d*) = {ys:.6e}")
print(f"  積 c* = x*·y* = {cstar:.6e}  (下界被覆率 c*/P = {cstar / p_true:.3f})")
print(f"  ハザード釣り合い: d(ln x)/dd = {dlnx:.6e},  -d(ln y)/dd = {-dlny:.6e}")
print(f"  log-log 平面の接線: ln y = ln c* - ln x (傾き -1) に (x*, y*) で接する")

# 参考: 点条件付き達成確率の上限 (m 時点で最大ダメージでも)
p_cap = 1.0 - float(back.cdf(np.array([D_TARGET - front.support_hi]))[0])
print(f"  参考: 条件付き達成確率の上限 P(達成 | S_m=最大) = {p_cap:.4e}")

# =============================================================================
# プロット (左: d 軸・対数スケール / 右: log-log パラメトリック)
# =============================================================================

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

pos = prod > 0
axL.semilogy(dg, np.maximum(x, 1e-300), color="tab:blue", lw=2,
             label="前 x(d) = P(S_m>=d)")
axL.semilogy(dg, np.maximum(y, 1e-300), color="tab:green", lw=2,
             label="後 y(d) = P(T>=D-d)")
axL.semilogy(dg[pos], prod[pos], color="tab:red", lw=2, label="積 x·y (同時確率)")
axL.axhline(p_true, color="gray", ls="--", lw=1,
            label=f"P(S_n>=D) = {p_true:.2e}")
axL.axvline(dstar, color="gray", ls=":", lw=1)
axL.plot([dstar], [cstar], "o", color="tab:red", ms=7, zorder=5)
axL.annotate(f"d*={dstar:,.0f}\nc*={cstar:.2e}", (dstar, cstar),
             textcoords="offset points", xytext=(8, 10), fontsize=9)
axL.set_title(f"和(メイドアリス): D={D_TARGET:,.0f}, m={M_CHECK}/7", fontsize=11)
axL.set_xlabel("関門 d (チェックポイントの累積ダメージ)")
axL.set_ylabel("確率 (対数)")
floor = max(prod[pos].min() if pos.any() else 1e-12, 1e-12)
axL.set_ylim(floor * 0.3, 2.0)
axL.grid(alpha=0.3, which="both")
axL.legend(fontsize=9, loc="lower center")

ok = (x > 0) & (y > 0)
axR.loglog(x[ok], y[ok], color="tab:purple", lw=2, label="曲線 (x(d), y(d))")
x_win = (xs * 1e-4, min(1.0, xs * 1e4))
y_win = (ys * 1e-4, min(1.0, ys * 1e4))
xx = np.geomspace(*x_win, 200)
axR.loglog(xx, cstar / xx, color="tab:red", ls="--", lw=1.5,
           label=f"等積直線 xy = c* = {cstar:.2e} (傾き -1)")
axR.plot([xs], [ys], "o", color="tab:red", ms=8, zorder=5)
axR.annotate(f"(x*, y*)=({xs:.2e}, {ys:.2e})", (xs, ys),
             textcoords="offset points", xytext=(8, -14), fontsize=9)
axR.set_xlim(*x_win)
axR.set_ylim(*y_win)
axR.set_title("log-log パラメトリック表示 (等積曲線 = 傾き -1 の直線)", fontsize=11)
axR.set_xlabel("前の確率 x (対数)")
axR.set_ylabel("後ろの確率 y (対数)")
axR.grid(alpha=0.3, which="both")
axR.legend(fontsize=9, loc="lower left")

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=130)
print(f"\nプロット保存: {OUTPUT}")
