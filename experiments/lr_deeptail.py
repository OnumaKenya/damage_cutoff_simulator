"""極深裾で Lugannani-Rice (LR公式) が COS法に優位になる例の実験。

docs/saddlepoint.md の結び (「中腹〜中程度の裾は COS法、極深裾は LR公式」) を
数値で裏づける。COS法は CDF F(x) を余弦級数和として求めるため、右の極深裾で
1-F(x) ≲ 1e-13 になると「ほぼ 1 同士の差」で桁落ちし、値がノイズ化・符号反転する。
一方 LR公式は

    P(S>x) ≈ 1-Φ(ŵ) + φ(ŵ)(1/û − 1/ŵ),   ŵ=sgn(t̂)√(2(t̂x−K)), û=t̂√K''

と「指数の肩」で裾を直接表すので、この桁落ちが無く極深裾でも頑健。

実験の要 (基準のとり方):
    極深裾には MC 真値が無い (1e8 サンプルでも ~1e-7 が限界)。そこで「同じ区間
    [a,b]・同じ項数 N の COS法を高精度演算 (mpmath, 60桁) で評価したもの」を
    準厳密な基準とする。倍精度 COS と高精度 COS は **演算精度だけ** が違うので、
    両者の差は純粋に桁落ち (catastrophic cancellation) に由来する。LR (倍精度) が
    この高精度基準に追従し続ける一方、倍精度 COS が 1e-13 付近で床にぶつかって
    崩れる様子を可視化する。これにより「LR が極深裾で優位」を分離して示せる。

シナリオは原子 (miss の点質量) を持たない混合 (常に命中、会心/非会心の2成分) に
する。原子があると COS は別要因 (Gibbs) でも崩れるため、純粋に「桁落ちだけ」が
COS を壊す状況に絞るための選択。会心が稀で大きいため右歪みを持ち、右裾が正規より
重く、深裾の比較が意味を持つ。

実行例:
    uv run python -m experiments.lr_deeptail
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.edgeworth_animation import Uniform  # noqa: E402
from experiments.saddlepoint_compare import (  # noqa: E402
    GroupedHits,
    Scenario,
    cos_cdf,
    lugannani_rice_sf,
    mc_survival_samples,
)


# =============================================================================
# 高精度 (mpmath) COS による準厳密な裾確率 — 「桁落ちの無い」基準線
#   倍精度 cos_cdf と「同じ区間 [a,b]・同じ項数 N」を使い、演算だけを多倍長にする。
#   F(x) を 60桁で積み上げてから 1-F を取るので、極深裾でも 1-F が正確に出る。
# =============================================================================

def _mixture_cf_mp(mixture: list[Uniform], u: mp.mpf) -> mp.mpc:
    """1Hit 混合の特性関数 φ(u)=Σ_j w_j e^{iu c_j} sinc(u h_j) を多倍長で。"""
    phi = mp.mpc(0)
    for comp in mixture:
        w = mp.mpf(comp.weight)
        c = mp.mpf(comp.center)
        h = mp.mpf(comp.half_width)
        uh = u * h
        sinc = mp.mpf(1) if uh == 0 else mp.sin(uh) / uh
        phi += w * mp.exp(mp.mpc(0, 1) * u * c) * sinc
    return phi


def _sum_cf_mp(grouped: GroupedHits, u: mp.mpf) -> mp.mpc:
    """Hit 和 S_n の特性関数 φ_S(u)=Π_hit φ_hit(u) を多倍長で。"""
    phi = mp.mpc(1)
    for cnt, mix in grouped:
        phi *= _mixture_cf_mp(mix, u) ** cnt
    return phi


def cos_sf_mpmath(
    grouped: GroupedHits, xs: np.ndarray, a: float, b: float, n_terms: int,
    dps: int = 60,
) -> np.ndarray:
    """高精度 COS による生存関数 1-F(x) を返す (準厳密・桁落ち無し)。

    倍精度 cos_cdf と同一の (a, b, n_terms) を用い、演算のみ多倍長にする。"""
    mp.mp.dps = dps
    a_m = mp.mpf(a)
    L = mp.mpf(b) - a_m
    us = [mp.mpf(k) * mp.pi / L for k in range(n_terms)]
    # 余弦係数 F_k = (2/L) Re[φ_S(u_k) e^{-i u_k a}] (x に依らないので先に計算)
    Fk = []
    for u in us:
        phi = _sum_cf_mp(grouped, u)
        Fk.append((mp.mpf(2) / L) * (phi * mp.exp(mp.mpc(0, -1) * u * a_m)).real)

    out = np.empty(len(xs), dtype=float)
    for i, x in enumerate(xs):
        dx = mp.mpf(float(x)) - a_m
        F = mp.mpf("0.5") * Fk[0] * dx            # k=0 項
        for k in range(1, n_terms):
            F += Fk[k] * mp.sin(us[k] * dx) / us[k]
        out[i] = float(mp.mpf(1) - F)             # 1-F を多倍長で確定してから float 化
    return out


def lr_sf_array(sc: Scenario, xs: np.ndarray) -> np.ndarray:
    """LR公式の生存関数 P(S>x) を xs 上で (倍精度で)。"""
    return np.array([
        lugannani_rice_sf(sc.grouped, float(x), sc.mean, sc.var, sc.t_cap)
        for x in xs
    ])


def cos_sf_double(sc: Scenario, xs: np.ndarray) -> np.ndarray:
    """倍精度 COS の生存関数 1-F(x)。深裾で桁落ちする側。"""
    F = cos_cdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)
    return 1.0 - F


# =============================================================================
# シナリオ: 原子なし・右歪みの中規模 Hit (LR が深裾で勝つ設定)
# =============================================================================

def deeptail_scenario() -> Scenario:
    """常に命中 (原子なし)、会心が稀で大きい → 右歪み・右裾が重い 25 Hit。

    会心 15%  U(3.0M, 5.0M)
    非会心 85% U(0.4M, 1.2M)
    点質量成分が無いので COS は Gibbs では壊れず、深裾の桁落ちだけが COS を壊す。"""
    mix = [
        Uniform(0.15, 3_000_000, 5_000_000),
        Uniform(0.85, 400_000, 1_200_000),
    ]
    hits = [mix for _ in range(25)]
    return Scenario("深裾 (会心15%×25発・原子なし・右歪み)", hits)


# =============================================================================
# 実験本体
# =============================================================================

N_MC = 100_000_000
SEED = 7
N_GRID = 70
OUT_DIR = "experiments/output"


def run() -> None:
    sc = deeptail_scenario()
    print(f"=== {sc.name} ===")
    print(
        f"  Hit数={sc.n_hits}  平均={sc.mean:,.0f}  標準偏差={sc.std:,.0f}  "
        f"λ3={sc.lam3:.4f}  λ4={sc.lam4:.4f}"
    )
    print(f"  COS区間=[{sc.cos_a:,.0f}, {sc.cos_b:,.0f}]  項数N={sc.cos_n}")

    # 右裾の z グリッド (中腹 ~ 極深裾)。上端は COS 区間 b の少し内側で止める。
    # 1-F ~ 1e-17 まで届かせ、COS倍精度の桁落ち床 (絶対誤差 ~1e-16) を突き抜けさせる。
    z_max = min(12.0, (sc.cos_b - sc.mean) / sc.std - 0.5)
    zs = np.linspace(2.5, z_max, N_GRID)
    xs = sc.mean + zs * sc.std

    print("  LR (倍精度) 評価中 ...")
    sf_lr = lr_sf_array(sc, xs)
    print("  COS (倍精度) 評価中 ...")
    sf_cos = cos_sf_double(sc, xs)
    print(f"  COS (高精度 mpmath {60}桁) 評価中 ... (基準線)")
    sf_ref = cos_sf_mpmath(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, dps=60)

    # MC: 浅い裾の足場 (全手法が一致することの確認用)。深裾は届かない。
    print(f"  MC {N_MC:,} 件 ...")
    samples = mc_survival_samples(sc.hit_mixtures, N_MC, SEED)
    samples.sort()
    n = samples.size
    cnt_gt = n - np.searchsorted(samples, xs, side="right")
    sf_mc = cnt_gt / n
    mc_reliable = cnt_gt >= 30          # 30 件以上ある所だけ信頼

    _print_table(sc, zs, xs, sf_lr, sf_cos, sf_ref, sf_mc, mc_reliable)
    _plot(sc, zs, sf_lr, sf_cos, sf_ref, sf_mc, mc_reliable,
          os.path.join(OUT_DIR, "lr_deeptail.png"))


def _print_table(
    sc, zs, xs, sf_lr, sf_cos, sf_ref, sf_mc, mc_reliable,
) -> None:
    print("\n  右裾 P(S>x): 高精度COS を基準とした相対誤差 (深裾で COS倍精度 が破綻)")
    print("    z      x(ダメージ)   P_ref(高精度)   LR(倍)誤差   COS(倍)誤差   MC")
    targets_z = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.5, 11.0, 11.5, 12.0]
    for zt in targets_z:
        i = int(np.argmin(np.abs(zs - zt)))
        ref = sf_ref[i]
        if ref <= 0:
            continue
        lr_err = (sf_lr[i] - ref) / ref
        cos_err = (sf_cos[i] - ref) / ref
        mc_str = f"{sf_mc[i]:.2e}" if mc_reliable[i] else "  —  (不足)"
        print(
            f"    {zs[i]:5.2f}  {xs[i]:13,.0f}  {ref:13.3e}  "
            f"{lr_err:+10.2%}  {cos_err:+12.2%}  {mc_str}"
        )


def _plot(
    sc, zs, sf_lr, sf_cos, sf_ref, sf_mc, mc_reliable, output_path,
) -> None:
    fig, (ax_sf, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        f"極深裾で LR公式 が COS法 に優位 — {sc.name}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}; 基準=高精度COS, COS倍精度は1-Fの桁落ちで破綻)",
        fontsize=12,
    )

    # 上段: 生存関数 P(S>x) (log y)
    ax_sf.plot(zs, np.where(sf_ref > 0, sf_ref, np.nan), color="black", lw=2.4,
               label="高精度COS (mpmath 60桁・準厳密な基準)")
    ax_sf.plot(zs, np.where(sf_lr > 0, sf_lr, np.nan), color="tab:red", lw=1.7,
               label="LR公式 (倍精度)")
    # COS倍精度: 桁落ちで負になった点は別マーカーで示す
    pos = sf_cos > 0
    ax_sf.plot(zs, np.where(pos, sf_cos, np.nan), color="tab:cyan", lw=1.4,
               label="COS法 (倍精度)")
    ax_sf.plot(zs[~pos], np.full((~pos).sum(), 1e-300), "x", color="tab:cyan",
               ms=6, label="COS法 (倍精度) が ≤0 (桁落ち破綻)")
    ax_sf.plot(zs[mc_reliable], sf_mc[mc_reliable], "k.", ms=5,
               label="MC (1e8 件・浅裾の足場)")
    ax_sf.axhline(1e-14, color="gray", ls=":", lw=1.0)
    ax_sf.text(zs[0], 1.3e-14, "COS倍精度 の桁落ち床 (~1e-14)", fontsize=8,
               color="gray")
    ax_sf.set_yscale("log")
    ax_sf.set_ylabel("右裾確率 P(S > x)")
    ax_sf.set_ylim(1e-20, 1.0)
    ax_sf.grid(True, which="both", alpha=0.3)
    ax_sf.legend(loc="lower left", fontsize=9)

    # 下段: 高精度COS基準の相対誤差
    eps = 1e-300
    lr_rel = (sf_lr - sf_ref) / (sf_ref + eps)
    cos_rel = (sf_cos - sf_ref) / (sf_ref + eps)
    ax_err.plot(zs, lr_rel, color="tab:red", lw=1.7, label="LR公式 (倍精度)")
    ax_err.plot(zs, cos_rel, color="tab:cyan", lw=1.4, label="COS法 (倍精度)")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.set_ylabel("相対誤差 (手法 − 高精度COS) / 高精度COS")
    ax_err.set_ylim(-1.2, 1.2)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper left", fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  保存: {output_path}")


if __name__ == "__main__":
    run()
