"""メイドアリスTL (7Hit 和モデル) で「1回でも非会心を出したら到達不能」な
目標 D = 40,000,000 を設定し、足切りライン(チェックポイント条件付き達成確率)を描く。

設定の意味:
  全Hit会心の最大ダメージ      Σ crit_max = 42,992,695
  1回だけ非会心の最大ダメージ   M* = Σ crit_max - min_i(crit_max_i - norm_max_i)
                                  = 38,816,499  (Hit1/2 を非会心にした場合が最大)
  D = 40,000,000 は M* < D ≤ Σ crit_max なので、
  「どれか1Hitでも非会心 → 残りを全部最大会心で引いても D に届かない」。
  つまり達成は「全7Hit会心 かつ 一様引きが十分高い」ときに限られる。

描くもの:
  左 : 点条件付け P(S_n ≥ D | S_m = d) を m = 2,4,6 について (COS, MC検証)
       + 足切りライン d*(p)(条件付き達成確率が p になる途中ダメージ)
  右 : チェックポイント生存ライン
        L(m) = D - (残りHitの最大会心和)  …これ未満だと以後どう引いても到達不能
       と「全会心トラック」「1回非会心トラック」の比較。
       L(m) が "1回非会心トラックの上限" を上回る = 非会心1回で即脱落、を可視化。

uv run python -m experiments.cutoff_maidalice_allcrit
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
    build_hit_mixtures,
    build_sum_dist,
    expand_card_to_hit_mixture,
    mc_sum,
)
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402

GLOBAL_CRIT = 65.27
GLOBAL_EVADE = 0.0
DAMAGE_MODE = "post_decay"
D_TARGET = 40_000_000.0
N_MC = 8_000_000
SEED = 20260617
CHECKPOINTS = [3]                       # 3射目時点で条件付け
CUTOFF_PS = [0.10, 0.25, 0.50, 0.75, 0.90]
OUTPUT = "experiments/output/cutoff_maidalice_allcrit.png"
OUTPUT_PARAM = "experiments/output/cutoff_maidalice_param.png"

try:
    import japanize_matplotlib  # noqa: F401  (副作用で日本語フォントを設定)
except ImportError:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ("Noto Sans CJK JP", "IPAexGothic", "HackGen35 Console NF",
                  "HackGen Console NF"):
        if _name in available:
            plt.rcParams["font.family"] = _name
            break


def quantile(cdf, p, lo, hi, iters=200):
    """単調 CDF の一般化逆 (二分法)。"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def per_hit_extremes():
    """各Hitの会心最大/最小・非会心最大 (post_decay 後) を返す。"""
    crit_max, crit_min, norm_max = [], [], []
    for card in _BLUEARCHIVE_7HIT_CARDS:
        cc = dict(card, crit_rate=100.0, evade_rate=0.0)
        nc = dict(card, crit_rate=0.0, evade_rate=0.0)
        cmix = expand_card_to_hit_mixture(cc, GLOBAL_CRIT, GLOBAL_EVADE, DAMAGE_MODE)
        nmix = expand_card_to_hit_mixture(nc, GLOBAL_CRIT, GLOBAL_EVADE, DAMAGE_MODE)
        cmx, cmn = max(u.hi for u in cmix), min(u.lo for u in cmix)
        nmx = max(u.hi for u in nmix)
        for _ in range(int(card.get("hits") or 1)):
            crit_max.append(cmx)
            crit_min.append(cmn)
            norm_max.append(nmx)
    return np.array(crit_max), np.array(crit_min), np.array(norm_max)


# =============================================================================
# 1. 設定の確認: D が「1回非会心で到達不能」域にあることを検算
# =============================================================================
crit_max, crit_min, norm_max = per_hit_extremes()
n_hits = len(crit_max)
sum_crit_max = float(crit_max.sum())
gaps = crit_max - norm_max
m_star = sum_crit_max - float(gaps.min())   # 1回だけ非会心の最大到達

print("=" * 72)
print("メイドアリスTL (7Hit, 会心65.27%/後半60.17%), 和モデル")
print("=" * 72)
print(f"全会心の最大ダメージ Σcrit_max = {sum_crit_max:,.0f}")
print(f"1回非会心の最大到達   M*       = {m_star:,.0f}"
      f"  (最小gap = {gaps.min():,.0f}, Hit{int(gaps.argmin())+1})")
print(f"目標 D                         = {D_TARGET:,.0f}")
assert m_star < D_TARGET <= sum_crit_max, "D が条件域 (M*, Σcrit_max] に無い"
print(f"→ M* < D ≤ Σcrit_max を満たす: 1回でも非会心が出たら到達不能\n")

# 達成は全会心に限られる。全会心の確率(参考)
crit_rates = []
for card in _BLUEARCHIVE_7HIT_CARDS:
    cr = card.get("crit_rate")
    cr = (GLOBAL_CRIT if cr is None else cr) / 100.0
    crit_rates.extend([cr] * int(card.get("hits") or 1))
p_all_crit = float(np.prod(crit_rates))
print(f"全7Hit会心の確率 = {p_all_crit:.4%} (達成確率の上限)")

# =============================================================================
# 2. COS: 点条件付け P(S_n ≥ D | S_m = d) = 1 - F_T(D - d)
# =============================================================================
hits = build_hit_mixtures(_BLUEARCHIVE_7HIT_CARDS, GLOBAL_CRIT, GLOBAL_EVADE, DAMAGE_MODE)
full = build_sum_dist(hits)
p_uncond = 1.0 - float(full.cdf(np.array([D_TARGET]))[0])
print(f"無条件達成確率 P(S_n ≥ D) = {p_uncond:.4%}\n")

# MC 一式 (各 m の S_m と最終 S_n、生存判定用)
rng = np.random.default_rng(SEED)
inc = np.stack([mc_sum([h], N_MC, rng) for h in hits], axis=1)  # (N, n_hits)
csum = np.cumsum(inc, axis=1)
sn = csum[:, -1]
print(f"MC 無条件達成確率 = {float(np.mean(sn >= D_TARGET)):.4%} (N={N_MC:,})\n")

curves = {}     # m -> (d_grid, cos_curve, dstar{p})
mc_pts = {}     # m -> (d_pts, mc_est, mc_err)
for m in CHECKPOINTS:
    front = build_sum_dist(hits[:m])
    back = build_sum_dist(hits[m:])
    # 生存に最低限必要な途中ダメージ L(m) = D - (残りの最大会心和)
    L_m = D_TARGET - float(crit_max[m:].sum())
    d_hi = front.support_hi
    d_grid = np.linspace(max(front.support_lo, L_m - 0.05 * (d_hi - L_m)), d_hi, 400)
    cos_curve = 1.0 - back.cdf(D_TARGET - d_grid)
    # 足切りライン d*(p) = D - q_T(1-p)
    dstar = {}
    for p in CUTOFF_PS:
        qT = quantile(lambda x: float(back.cdf(np.array([x]))[0]), 1.0 - p,
                      back.support_lo, back.support_hi)
        ds = D_TARGET - qT
        dstar[p] = ds if ds <= d_hi else None
    curves[m] = (d_grid, cos_curve, dstar, L_m)

    # MC 検証 (点条件付け: 途中ダメージ d±h のビン)
    sm = csum[:, m - 1]
    sd_m = float(np.std(sm))
    # 生存域に標本が出る分位点で評価点を取る
    d_pts = np.quantile(sm[sm >= L_m], [0.2, 0.5, 0.8]) if np.any(sm >= L_m) \
        else np.array([])
    est, err = [], []
    for d in d_pts:
        msk = np.abs(sm - d) <= 0.02 * sd_m
        c = int(msk.sum())
        if c == 0:
            est.append(np.nan); err.append(np.nan); continue
        pr = float(np.mean(sn[msk] >= D_TARGET))
        est.append(pr); err.append(math.sqrt(max(pr * (1 - pr), 1e-12) / c))
    mc_pts[m] = (d_pts, np.array(est), np.array(err))

    print(f"m={m}: 生存ライン L(m)=D-Σ_{{i>m}}crit_max = {L_m:,.0f}"
          f"  (これ未満は以後どう引いても到達不能)")
    for p in CUTOFF_PS:
        ds = dstar[p]
        s = f"{ds:,.0f}" if ds is not None else "達成不能 (この時点では p 保証不可)"
        print(f"    足切り d*({p:.0%}) = {s}")

# =============================================================================
# 3. 全会心トラック vs 1回非会心トラック (右パネル用)
# =============================================================================
ms = np.arange(1, n_hits + 1)
track_crit_max = np.cumsum(crit_max)                  # 全会心の累積上限
track_crit_min = np.cumsum(crit_min)                  # 全会心の累積下限
remain_crit_max = sum_crit_max - track_crit_max       # 残りの最大会心和
survive_line = D_TARGET - remain_crit_max             # L(m)
# m 時点で「ちょうど1回非会心」だった場合の累積上限 = Σ_{i≤m}crit_max - (前半の最小gap)
one_miss_max = np.array([track_crit_max[i] - float(gaps[:i + 1].min())
                         for i in range(n_hits)])

# =============================================================================
# プロット
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
m0 = CHECKPOINTS[0]
d_grid, cos_curve, dstar, L_m = curves[m0]
ax.plot(d_grid / 1e6, cos_curve, "-", color="tab:blue", lw=2.5,
        label="COS 厳密 (パラメトリック)")
dp, est, errv = mc_pts[m0]
if dp.size:
    ax.errorbar(dp / 1e6, est, yerr=errv, fmt="o", color="tab:red", ms=6,
                capsize=3, zorder=5, label="条件付き MC")
# 生存ライン (これ未満は以後どう引いても到達不能 → 確率 0)
ax.axvline(L_m / 1e6, color="gray", ls="-", lw=1.2)
ax.annotate(f"生存ライン L(3)={L_m/1e6:.2f}M\n(これ未満は到達不能)",
            (L_m / 1e6, 0.55), fontsize=8, color="gray",
            ha="left", va="center")
# 足切りライン d*(p)
for p, ds in dstar.items():
    if ds is not None and d_grid[0] <= ds <= d_grid[-1]:
        ax.axvline(ds / 1e6, color="tab:green", ls=":", lw=1, alpha=0.8)
        ax.annotate(f"d*({p:.0%})", (ds / 1e6, 0.02), rotation=90,
                    fontsize=8, color="tab:green", va="bottom")
ax.set_title(f"3射目時点の足切り(パラメトリック)\n"
             f"P(S_7 >= D | S_3 = d),  D = {D_TARGET/1e6:.0f}M "
             f"(1回でも非会心なら到達不能)", fontsize=11)
ax.set_xlabel("3射目時点の累積ダメージ d  [百万]")
ax.set_ylabel("達成確率  P(最終 >= D | 途中 = d)")
ax.set_ylim(-0.03, 1.03)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="center left")

ax2 = axes[1]
ax2.fill_between(ms, track_crit_min / 1e6, track_crit_max / 1e6,
                 color="tab:green", alpha=0.18, label="全会心トラックの帯")
ax2.plot(ms, track_crit_max / 1e6, "-o", color="tab:green", lw=2,
         label="全会心の累積上限")
ax2.plot(ms, one_miss_max / 1e6, "-s", color="tab:orange", lw=2,
         label="1回だけ非会心の累積上限")
ax2.plot(ms, survive_line / 1e6, "--D", color="tab:red", lw=2,
         label="生存ライン L(m)=D−残り最大会心")
ax2.axhline(D_TARGET / 1e6, color="black", ls=":", lw=1.5)
ax2.annotate(f"D = {D_TARGET/1e6:.0f}M", (1, D_TARGET / 1e6),
             fontsize=9, va="bottom")
ax2.axvline(m0, color="tab:blue", ls="-", lw=1, alpha=0.4)
ax2.annotate("3射目", (m0, track_crit_min[0] / 1e6), color="tab:blue",
             fontsize=9, ha="center", va="bottom")
ax2.set_title("全会心トラック vs 1回非会心トラックと生存ライン\n"
              "生存ラインが『1回非会心の上限』を上回る区間=非会心1回で即脱落",
              fontsize=11)
ax2.set_xlabel("チェックポイント m (射目)")
ax2.set_ylabel("累積ダメージ  [百万]")
ax2.set_xticks(ms)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9, loc="upper left")

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=130)
print(f"\nプロット保存: {OUTPUT}")

# =============================================================================
# 4. パラメトリック曲線 (独立画像): 両軸を前/後ろの確率に取る
#       パラメータ      : d = 3射目時点の累積ダメージ
#       横軸 (前の確率) : P(S_3 >= d)        … 足切り d を通過する確率
#       縦軸 (後ろの確率): P(T   >= D - d)    … 通過後に残り4Hitで D-d を出す確率
#    d を上げると「前」は下がり「後ろ」は上がるトレードオフを 1 本の曲線で表す。
# =============================================================================
m0 = CHECKPOINTS[0]
front0 = build_sum_dist(hits[:m0])
back0 = build_sum_dist(hits[m0:])
# d の走査域: S_3 が実際に取りうる範囲 [support_lo, support_hi]
d_par = np.linspace(front0.support_lo, front0.support_hi, 800)
p_front = 1.0 - front0.cdf(d_par)            # 前: 足切り通過確率
p_back = 1.0 - back0.cdf(D_TARGET - d_par)   # 後ろ: 残りで D-d 達成確率

figp, axp = plt.subplots(figsize=(7.5, 7))
sc = axp.scatter(p_front, p_back, c=d_par / 1e6, cmap="viridis", s=9, zorder=3)
axp.plot(p_front, p_back, "-", color="gray", lw=0.8, alpha=0.6, zorder=2)
cb = figp.colorbar(sc, ax=axp)
cb.set_label("パラメータ d = 3射目時点の累積ダメージ [百万]", fontsize=10)

# 代表的な d を注記 (会心/非会心の境目あたりを含む)
mark_d = np.quantile(d_par, [0.1, 0.35, 0.6, 0.8, 0.95])
for dm in mark_d:
    pf = 1.0 - float(front0.cdf(np.array([dm]))[0])
    pb = 1.0 - float(back0.cdf(np.array([D_TARGET - dm]))[0])
    axp.plot(pf, pb, "o", color="tab:red", ms=6, zorder=4)
    axp.annotate(f"d={dm/1e6:.1f}M", (pf, pb), fontsize=8,
                 xytext=(6, 4), textcoords="offset points")

axp.set_title("3射目時点のパラメトリック曲線(両軸=前/後ろの確率)\n"
              f"D = {D_TARGET/1e6:.0f}M (1回でも非会心なら到達不能)", fontsize=11)
axp.set_xlabel("前の確率  P(S_3 >= d)  = 足切り d を通過する確率")
axp.set_ylabel("後ろの確率  P(T >= D - d)  = 通過後に残り4Hitで D-d を出す確率")
axp.set_xlim(-0.03, 1.03)
axp.set_ylim(-0.03, 1.03)
axp.grid(alpha=0.3)
figp.tight_layout()
figp.savefig(OUTPUT_PARAM, dpi=130)
print(f"パラメトリック保存: {OUTPUT_PARAM}")
