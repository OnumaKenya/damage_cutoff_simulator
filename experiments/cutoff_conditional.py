"""docs/cutoff.md の検証: 既存プリセットでチェックポイント条件付き達成確率を計算する。

- 和モデル: cos_app の _BLUEARCHIVE_7HIT_CARDS (メイドアリスTL, 7Hit, 会心65.27%)
- 積モデル: product_cos_app の PRESETS["ビナーミカ (R1=2, R0=1)"] (22Hit, HP依存)

検証内容:
  1. 点条件付け  P(達成 | 途中ダメージ = d)
       和: 1 - F_T(D - d) / 積: F_{L'}(s(D) - s(d))   vs  ビン条件付き MC
  2. 区間条件付け P(達成 | 途中ダメージ ≥ d)            vs  フィルタ MC
  3. 足切りライン d*(p) の分位点閉形式                  vs  d* における条件付き MC
  4. 積モデルの再スタート同値 (H1 ← H1 - d) の MC 非依存チェック

uv run python -m experiments.cutoff_conditional
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
from app.backend.cos import (  # noqa: E402
    HPParams,
    build_hit_mixtures,
    build_product_dist,
    build_sum_dist,
    damage_moments,
    mc_sum,
    support_bounds_hits,
    y_mixture,
)
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402
from experiments.product_cos_app import PRESETS, _hp0_to_base_cards  # noqa: E402

N_MC = 4_000_000
SEED = 20260613
TARGET_TAIL = 0.30          # 目標 D は無条件達成確率がこの値になる点に置く
CUTOFF_PS = [0.25, 0.50, 0.75, 0.90, 0.99]
OUTPUT = "experiments/output/cutoff_conditional.png"


def quantile(cdf, p, lo, hi, iters=200):
    """単調 CDF の分位点 (二分法)。"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binned_conditional(partial, total, D, d_values, half_width):
    """MC: P(total ≥ D | partial ∈ [d±h]) と標準誤差。"""
    est, err = [], []
    for d in d_values:
        m = np.abs(partial - d) <= half_width
        cnt = int(m.sum())
        if cnt == 0:
            est.append(np.nan)
            err.append(np.nan)
            continue
        p = float(np.mean(total[m] >= D))
        est.append(p)
        err.append(math.sqrt(max(p * (1 - p), 1e-12) / cnt))
    return np.array(est), np.array(err)


def filtered_conditional(partial, total, D, d_values):
    """MC: P(total ≥ D | partial ≥ d) と標準誤差。"""
    est, err = [], []
    for d in d_values:
        m = partial >= d
        cnt = int(m.sum())
        if cnt == 0:
            est.append(np.nan)
            err.append(np.nan)
            continue
        p = float(np.mean(total[m] >= D))
        est.append(p)
        err.append(math.sqrt(max(p * (1 - p), 1e-12) / cnt))
    return np.array(est), np.array(err)


# =============================================================================
# 1. 和モデル: メイドアリスTL (7Hit), チェックポイント m=3 (3射目まで)
# =============================================================================

print("=" * 72)
print("和モデル: _BLUEARCHIVE_7HIT_CARDS (会心65.27%, 7Hit), m = 3")
print("=" * 72)

hits = build_hit_mixtures(_BLUEARCHIVE_7HIT_CARDS, 65.27, 0.0, "post_decay")
n_hits = len(hits)
m = 3
front = build_sum_dist(hits[:m])      # S_m
back = build_sum_dist(hits[m:])       # T
full = build_sum_dist(hits)           # S_n (D の設定にのみ使用)
assert front.av is None and back.av is None, "このプリセットに原子は無い想定"

D_sum = quantile(lambda x: float(full.cdf(np.array([x]))[0]), 1.0 - TARGET_TAIL,
                 full.support_lo, full.support_hi)
print(f"Hit数={n_hits}  E[S_n]={full.mean:,.0f}  σ={math.sqrt(full.var):,.0f}")
print(f"目標 D (無条件達成確率 {TARGET_TAIL:.0%}) = {D_sum:,.0f}")

# --- COS: 点条件付け曲線 (構築は back の 1 回だけ。d にも D にも依存しない) ---
d_lo = quantile(lambda x: float(front.cdf(np.array([x]))[0]), 0.005,
                front.support_lo, front.support_hi)
d_hi = quantile(lambda x: float(front.cdf(np.array([x]))[0]), 0.995,
                front.support_lo, front.support_hi)
d_grid_sum = np.linspace(d_lo, d_hi, 200)
point_cos_sum = 1.0 - back.cdf(D_sum - d_grid_sum)

# --- COS: 区間条件付け (右からの累積で全 d を一括) ---
s_fine = np.linspace(front.support_lo, front.support_hi, 20001)
w = front.pdf(s_fine) * (1.0 - back.cdf(D_sum - s_fine))
ds = s_fine[1] - s_fine[0]
cells = 0.5 * (w[1:] + w[:-1]) * ds
W_right = np.concatenate([np.cumsum(cells[::-1])[::-1], [0.0]])  # ∫_s^{hi} w
num_sum = np.interp(d_grid_sum, s_fine, W_right)
den_sum = 1.0 - front.cdf(d_grid_sum)
interval_cos_sum = np.clip(num_sum / np.maximum(den_sum, 1e-300), 0.0, 1.0)

# --- MC (S_m ⊥ T なので独立サンプルの和で S_n を作る) ---
rng = np.random.default_rng(SEED)
sm = mc_sum(hits[:m], N_MC, rng)
sn = sm + mc_sum(hits[m:], N_MC, rng)
sd_m = float(np.std(sm))
d_pts_sum = np.quantile(sm, [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
pt_mc_sum, pt_err_sum = binned_conditional(sm, sn, D_sum, d_pts_sum, 0.02 * sd_m)
iv_mc_sum, iv_err_sum = filtered_conditional(sm, sn, D_sum, d_pts_sum)
iv_cos_at = np.interp(d_pts_sum, d_grid_sum, interval_cos_sum)

print("\n点条件付け P(S_n≥D | S_m=d):  COS vs ビンMC (±0.02σ_m)")
for d, p_mc, e in zip(d_pts_sum, pt_mc_sum, pt_err_sum):
    p_cos = 1.0 - float(back.cdf(np.array([D_sum - d]))[0])
    print(f"  d={d:>13,.0f}  COS={p_cos:.4f}  MC={p_mc:.4f} ±{e:.4f}"
          f"  差={p_cos - p_mc:+.4f}")

print("\n区間条件付け P(S_n≥D | S_m≥d):  COS求積 vs フィルタMC")
for d, p_cos, p_mc, e in zip(d_pts_sum, iv_cos_at, iv_mc_sum, iv_err_sum):
    print(f"  d={d:>13,.0f}  COS={p_cos:.4f}  MC={p_mc:.4f} ±{e:.4f}"
          f"  差={p_cos - p_mc:+.4f}")

# --- 足切りライン d*(p) = D - q_T(1-p) ---
print("\n足切りライン d*(p) = D - q_T(1-p) と、d* における条件付き MC:")
d_max_front = front.support_hi
p_cap_sum = 1.0 - float(back.cdf(np.array([D_sum - d_max_front]))[0])
print(f"  (m={m} 時点の最大可能ダメージ {d_max_front:,.0f}"
      f" → 条件付き確率の上限 {p_cap_sum:.4f})")
cut_sum = {}
for p in CUTOFF_PS:
    qT = quantile(lambda x: float(back.cdf(np.array([x]))[0]), 1.0 - p,
                  back.support_lo, back.support_hi)
    dstar = D_sum - qT
    if dstar > d_max_front:
        print(f"  p={p:.2f}  d*={dstar:>13,.0f}  → 最大 {d_max_front:,.0f} を超え"
              "、この時点では保証不能")
        continue
    cut_sum[p] = dstar
    mask = np.abs(sm - dstar) <= 0.02 * sd_m
    if mask.any():
        print(f"  p={p:.2f}  d*={dstar:>13,.0f}  MC@d*={float(np.mean(sn[mask] >= D_sum)):.4f}")
    else:
        print(f"  p={p:.2f}  d*={dstar:>13,.0f}  MC@d*=標本なし"
              " (d* が S_m の確率ギャップ内: 多峰分布の谷)")

# =============================================================================
# 2. 積モデル: ビナーミカ (HP依存, 22Hit), チェックポイント m=11 (大ダメージHit直後)
# =============================================================================

print()
print("=" * 72)
preset_name = "ビナーミカ (R1=2, R0=1)"
print(f"積モデル: PRESETS[{preset_name!r}], m = 11")
print("=" * 72)

pr = PRESETS[preset_name]
H, H1, R0, R1 = pr["H"], pr["H1"], pr["R0"], pr["R1"]
hp = HPParams(H=H, H1=H1, R0=R0, R1=R1)
cards = _hp0_to_base_cards(pr["cards"], R0, undo_decay=True)  # hp0入力 → 基礎 x (生値)
base_per_hit = build_hit_mixtures(cards, pr["global_crit"], pr["global_evade"],
                                  "pre_decay")
ymix_per_hit = [y_mixture(b, hp.beta) for b in base_per_hit]
n_hits_p = len(ymix_per_hit)
mp = 11

front_p = build_product_dist(ymix_per_hit[:mp], hp)   # L_m (S レベルで使う)
back_p = build_product_dist(ymix_per_hit[mp:], hp)    # L'
full_p = build_product_dist(ymix_per_hit, hp)
assert front_p.av is None and back_p.av is None
Htil = hp.Htil
mean_D, var_D = damage_moments(ymix_per_hit, Htil)
print(f"Hit数={n_hits_p}  H̃₁={Htil:,.0f}  E[D]={mean_D:,.0f}  σ={math.sqrt(var_D):,.0f}")


def cdf_S(dist, s):
    """ProductDist の S レベル CDF (台の外は 0/1 に飽和)。"""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    below, above = s < dist.a, s > dist.b
    mid = ~(below | above)
    out[below], out[above] = 0.0, 1.0
    if mid.any():
        out[mid] = dist._cdf_S(s[mid])
    return out


def s_of(x):
    return np.log1p(-np.asarray(x, dtype=float) / Htil)


D_prod = quantile(lambda x: float(full_p.cdf(np.array([x]))[0]), 1.0 - TARGET_TAIL,
                  0.0, full_p.d_max)
print(f"目標 D (無条件達成確率 {TARGET_TAIL:.0%}) = {D_prod:,.0f}")

# --- MC: 漸化式 H_{k+1} = H_k - (βH_k + R0)x を直接回し、m Hit 後を記録 (真値) ---
rng = np.random.default_rng(SEED + 1)
Hn = np.full(N_MC, H1, dtype=float)
Dm_mc = None
for i, base in enumerate(base_per_hit):
    wgt = np.array([c.weight for c in base])
    wgt = wgt / wgt.sum()
    comp = rng.choice(len(base), size=N_MC, p=wgt)
    x = np.empty(N_MC)
    for j, c in enumerate(base):
        msk = comp == j
        cnt = int(msk.sum())
        if cnt:
            x[msk] = rng.uniform(c.lo, c.hi, size=cnt) if c.hi > c.lo else c.lo
    Hn = Hn - (hp.beta * Hn + R0) * x
    if i + 1 == mp:
        Dm_mc = H1 - Hn
Dn_mc = H1 - Hn

# --- COS: 点条件付け F_{L'}(s(D) - s(d)) (back_p の構築 1 回、(d,D) 非依存) ---
d_lo_p, d_hi_p = np.quantile(Dm_mc, [0.005, 0.995])
d_grid_prod = np.linspace(d_lo_p, d_hi_p, 200)
point_cos_prod = cdf_S(back_p, s_of(D_prod) - s_of(d_grid_prod))

# --- COS: 区間条件付け (対数座標で左からの累積; D_m≥d ⟺ L_m ≤ s(d)) ---
l_fine = np.linspace(front_p.a, front_p.b, 20001)
w_p = front_p._pdf_S(l_fine) * cdf_S(back_p, s_of(D_prod) - l_fine)
dl = l_fine[1] - l_fine[0]
cells_p = 0.5 * (w_p[1:] + w_p[:-1]) * dl
W_left = np.concatenate([[0.0], np.cumsum(cells_p)])  # ∫_{A}^{ℓ} w
sd_grid = s_of(d_grid_prod)
num_prod = np.interp(sd_grid, l_fine, W_left)
den_prod = cdf_S(front_p, sd_grid)
interval_cos_prod = np.clip(num_prod / np.maximum(den_prod, 1e-300), 0.0, 1.0)

sd_mp = float(np.std(Dm_mc))
d_pts_prod = np.quantile(Dm_mc, [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
pt_mc_prod, pt_err_prod = binned_conditional(Dm_mc, Dn_mc, D_prod, d_pts_prod,
                                             0.02 * sd_mp)
iv_mc_prod, iv_err_prod = filtered_conditional(Dm_mc, Dn_mc, D_prod, d_pts_prod)
iv_cos_at_p = np.interp(d_pts_prod, d_grid_prod, interval_cos_prod)

print("\n点条件付け P(D_n≥D | D_m=d) = F_{L'}(s(D)-s(d)):  COS vs ビンMC (±0.02σ_m)")
for d, p_mc, e in zip(d_pts_prod, pt_mc_prod, pt_err_prod):
    p_cos = float(cdf_S(back_p, s_of(D_prod) - s_of(d)))
    print(f"  d={d:>13,.0f}  COS={p_cos:.4f}  MC={p_mc:.4f} ±{e:.4f}"
          f"  差={p_cos - p_mc:+.4f}")

print("\n区間条件付け P(D_n≥D | D_m≥d):  COS求積 vs フィルタMC")
for d, p_cos, p_mc, e in zip(d_pts_prod, iv_cos_at_p, iv_mc_prod, iv_err_prod):
    print(f"  d={d:>13,.0f}  COS={p_cos:.4f}  MC={p_mc:.4f} ±{e:.4f}"
          f"  差={p_cos - p_mc:+.4f}")

# --- 再スタート同値: H1 ← H1 - d の独立構築と一致するか (MC 非依存) ---
print("\n再スタート同値チェック (P = 1 - F_restart(D-d) との差):")
max_diff = 0.0
for d in d_pts_prod:
    hp_restart = HPParams(H=H, H1=H1 - d, R0=R0, R1=R1)
    restart = build_product_dist(ymix_per_hit[mp:], hp_restart)
    p_restart = 1.0 - float(restart.cdf(np.array([D_prod - d]))[0])
    p_shift = float(cdf_S(back_p, s_of(D_prod) - s_of(d)))
    max_diff = max(max_diff, abs(p_restart - p_shift))
    print(f"  d={d:>13,.0f}  しきい値シフト={p_shift:.10f}  再スタート={p_restart:.10f}")
print(f"  最大差 = {max_diff:.3e}")

# --- 足切りライン d*(p) = H̃₁ - (H̃₁-D) e^{-q_{L'}(p)} ---
print("\n足切りライン (H̃₁-d*) = (H̃₁-D)·e^{-q_{L'}(p)} と、d* における条件付き MC:")
d_max_front_p = -Htil * math.expm1(front_p.a)
p_cap_prod = float(cdf_S(back_p, s_of(D_prod) - s_of(d_max_front_p)))
print(f"  (m={mp} 時点の最大可能ダメージ {d_max_front_p:,.0f}"
      f" → 条件付き確率の上限 {p_cap_prod:.4f})")
cut_prod = {}
for p in CUTOFF_PS:
    qL = quantile(lambda x: float(cdf_S(back_p, x)), p, back_p.a, back_p.b)
    dstar = Htil - (Htil - D_prod) * math.exp(-qL)
    if dstar > d_max_front_p:
        print(f"  p={p:.2f}  d*={dstar:>13,.0f}  → 最大 {d_max_front_p:,.0f} を超え"
              "、この時点では保証不能")
        continue
    cut_prod[p] = dstar
    mask = np.abs(Dm_mc - dstar) <= 0.02 * sd_mp
    p_mc = float(np.mean(Dn_mc[mask] >= D_prod)) if mask.any() else np.nan
    note = "" if mask.any() else " (ビン内に標本なし: 深裾)"
    print(f"  p={p:.2f}  d*={dstar:>13,.0f}  傾き e^(-q)={math.exp(-qL):.4f}"
          f"  MC@d*={p_mc:.4f}{note}")

# =============================================================================
# プロット
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
panels = [
    (axes[0, 0], d_grid_sum, point_cos_sum, d_pts_sum, pt_mc_sum, pt_err_sum,
     cut_sum, f"和: P(S_n>=D | S_m=d)  D={D_sum:,.0f}, m={m}/{n_hits}"),
    (axes[0, 1], d_grid_sum, interval_cos_sum, d_pts_sum, iv_mc_sum, iv_err_sum,
     None, "和: P(S_n>=D | S_m>=d)"),
    (axes[1, 0], d_grid_prod, point_cos_prod, d_pts_prod, pt_mc_prod, pt_err_prod,
     cut_prod, f"積(ビナーミカ): P(D_n>=D | D_m=d)  D={D_prod:,.0f}, m={mp}/{n_hits_p}"),
    (axes[1, 1], d_grid_prod, interval_cos_prod, d_pts_prod, iv_mc_prod, iv_err_prod,
     None, "積(ビナーミカ): P(D_n>=D | D_m>=d)"),
]
for ax, dg, cos_curve, dp, mc, err, cuts, title in panels:
    ax.plot(dg, cos_curve, "-", color="tab:blue", lw=2, label="COS (閉形式/求積)")
    ax.errorbar(dp, mc, yerr=err, fmt="o", color="tab:red", ms=5, capsize=3,
                label="条件付き MC", zorder=5)
    if cuts:
        for p, dstar in cuts.items():
            if dg[0] <= dstar <= dg[-1]:
                ax.axvline(dstar, color="gray", ls=":", lw=1)
                ax.annotate(f"d*({p:.0%})", (dstar, 0.03), rotation=90,
                            fontsize=8, color="gray")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("チェックポイントの累積ダメージ d")
    ax.set_ylabel("条件付き達成確率")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=130)
print(f"\nプロット保存: {OUTPUT}")
