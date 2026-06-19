"""Esscher 傾斜 + COS 法による「極深裾」確率の数値反転実験。

docs/saddlepoint.md の末尾で指摘した COS 法の限界 —
    F(x) を余弦級数の和として求めるため、1−F(x) が ~10^{-12} を切る極深裾では
    「ほぼ 1 同士の差」になって桁落ちする —
を、指数傾斜 (Esscher 変換) で解消する手法を試す。

アイデア:
    評価点 x ごとに、サドルポイント t̂ (K'(t̂)=x を満たす) で分布を傾斜させた
    密度 f_t(y) = e^{t̂ y − K(t̂)} f(y) を考える。傾斜分布の平均は K'(t̂)=x なので、
    「元の分布では深い裾だった x」が「傾斜分布のど真ん中」に来る。中心付近なら
    COS 法は何の桁落ちもなく機械精度で f_t を反転できる。

    傾斜分布の特性関数は CGF で閉じて書ける:
        φ_t(u) = E_t[e^{iuY}] = M_S(t̂+iu)/M_S(t̂) = exp(K(t̂+iu) − K(t̂))
    これを [a,b]=x±L·σ_t (σ_t=√K''(t̂), 傾斜分散) 上で余弦級数に展開し、傾斜密度
    f_t(y) ≈ Σ'_k F_k cos(u_k(y−a)) を得る。

    あとは傾斜を「戻す」だけ。上側裾は
        P(S>x) = ∫_x^∞ f(y) dy = e^{K(t̂)} ∫_x^b e^{−t̂ y} f_t(y) dy
    で、被積分の e^{−t̂ y} cos(u_k(y−a)) は解析的に積分できる (下記 _tail_integral)。
    結果は ~ e^{K(t̂)−t̂ x}·O(1) という正しい裾スケールの数で、「1−1」の引き算を
    一切経由しないため、e^{K−t̂x} がアンダーフローする (~10^{-300}) 限界まで桁落ち
    しない。サドルポイント (Lugannani–Rice) と同じ「指数の肩で裾を表す」頑健さを、
    COS の準厳密さと両立させたもの。

比較対象 (右裾 P(S>x)):
    - モンテカルロ 経験裾確率 (mid-tail の真値・標本誤差の範囲で基準)
    - 素の COS 法 (1−F; 深裾で桁落ちして雑音/負値に落ちるのを可視化)
    - Lugannani–Rice サドルポイント (深裾で頑健だが中腹で区分多項式構造を均す系統誤差)
    - Esscher 傾斜 + COS (本手法; mid で COS と一致, 深裾で LR と一致するはず)

実行例:
    uv run python -m experiments.tilted_cos_compare
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cos_compare import (  # noqa: E402
    GroupedHits,
    Scenario,
    mc_survival_samples,
    sum_cf,
    atom_part_distribution,
    atom_cf,
    _atom_step_cdf,
    toy_scenario,
    real_scenario,
    discrete_scenario,
)
from experiments.saddlepoint_compare import (  # noqa: E402
    cgf_derivs,
    solve_saddlepoint,
    lr_sf_array,
)


# =============================================================================
# 複素 sinhc と 傾斜特性関数 φ_t(u) = M_S(t+iu)/M_S(t)
# =============================================================================

def _sinhc(z: np.ndarray) -> np.ndarray:
    """sinh(z)/z (z 複素, z→0 で 1)。小 |z| は分子で桁落ちするので Taylor。

        sinh z / z = 1 + z²/6 + z⁴/120 + z⁶/5040 + z⁸/362880 + ...
    """
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 0.5
    zb = np.where(small, 1.0, z)            # small 位置のゼロ割回避ダミー
    big = np.sinh(zb) / zb
    z2 = z * z
    sm = 1.0 + z2 / 6.0 + z2**2 / 120.0 + z2**3 / 5040.0 + z2**4 / 362880.0
    return np.where(small, sm, big)


def tilted_cf(grouped: GroupedHits, t: float, u: np.ndarray) -> np.ndarray:
    """Esscher 傾斜分布 (傾斜パラメータ t) の特性関数
        φ_t(u) = M_S(t+iu)/M_S(t) = exp(Σ_hit [log M_hit(t+iu) − log M_hit(t)])。

    オーバーフロー回避: 各群で最大 (t≥0) / 最小 (t<0) 中心 c* を括り出し、
        log M_hit(t+iu) − log M_hit(t) = iu·c* + log B_c(u) − log B_r,
        B_c(u)=Σ_j w_j e^{(t+iu)(c_j−c*)} sinhc((t+iu)h_j),
        B_r   =Σ_j w_j e^{t(c_j−c*)}     sinhc(t h_j)        (= 実 MGF/e^{t c*})
    として相対量だけ評価する (e^{t(c_j−c*)}≤1)。各 log は主値でも、最後に
    exp(Σ) を取るので分岐は積に巻き戻り正しい φ_t を与える。"""
    u = np.asarray(u, dtype=float)
    s = t + 1j * u
    acc = np.zeros_like(u, dtype=complex)
    for cnt, mix in grouped:
        cs = [c.center for c in mix]
        c_star = max(cs) if t >= 0 else min(cs)
        Bc = np.zeros_like(u, dtype=complex)
        Br = 0.0
        for comp in mix:
            w, c, h = comp.weight, comp.center, comp.half_width
            if h == 0.0:
                gc = np.ones_like(u, dtype=complex)
                gr = 1.0
            else:
                gc = _sinhc(s * h)
                gr = float(_sinhc(np.array([t * h], dtype=complex))[0].real)
            ec = np.exp(s * (c - c_star))
            Bc += w * ec * gc
            Br += w * math.exp(t * (c - c_star)) * gr
        diff = 1j * u * c_star + np.log(Bc) - math.log(Br)
        acc += cnt * diff
    return np.exp(acc)


# =============================================================================
# Esscher 傾斜 + COS による上側裾確率 P(S>x)
# =============================================================================

_TILT_L = 12.0
_TILT_TERMS_PER_SIGMA = 256
_TILT_N_MIN = 1024
_TILT_N_MAX = 1 << 16


def tilted_cos_sf_scalar(
    grouped: GroupedHits, x: float, t_cap: float,
    support_lo: float, support_hi: float,
) -> float:
    """Esscher 傾斜 + COS による P(S>x) (スカラー, 右裾 x>mean 用)。

    1. サドルポイント t̂: K'(t̂)=x を解く (傾斜平均 = x)。
    2. 傾斜分散 σ_t²=K''(t̂)。窓 [a,b]=x±L·σ_t をサポートでクリップ。
    3. 傾斜密度の余弦係数 F_k=(2/L)Re[φ_t(u_k)e^{−i u_k a}]。
    4. 裾を戻す: P(S>x)=e^{K(t̂)} ∫_x^b e^{−t̂ y} f_t(y) dy を項別積分。
    """
    that = solve_saddlepoint(grouped, x, t_cap)
    K, Kp, Kpp = cgf_derivs(grouped, that)
    if that <= 0.0 or Kpp <= 0.0:
        return float("nan")              # 右裾 (that>0) 専用
    # サドルポイントが収束しない = x が有界サポートの端に張り付いている
    # (t̂ が cap に達し K'(t̂)≠x)。そこは確率ほぼ 0 で σ_t→0、窓も潰れるので
    # 値を主張せず nan を返す (プロットで端の数点が欠けるだけ)。
    if abs(Kp - x) > 1e-6 * max(abs(x), math.sqrt(Kpp)):
        return float("nan")
    sig_t = math.sqrt(Kpp)
    a = max(support_lo, x - _TILT_L * sig_t)
    b = min(support_hi, x + _TILT_L * sig_t)
    L = b - a
    width_sigma = L / sig_t
    n = max(_TILT_N_MIN, min(_TILT_N_MAX,
                             int(math.ceil(_TILT_TERMS_PER_SIGMA * width_sigma))))
    k = np.arange(n)
    u = k * math.pi / L
    phi = tilted_cf(grouped, that, u)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))

    # ∫_x^b e^{−t y} cos(u_k(y−a)) dy の原始関数 P_k(y)=e^{−t y}·Q_k(y),
    #   Q_k(y) = (−t cos(u_k(y−a)) + u_k sin(u_k(y−a))) / (t²+u_k²)
    # P(S>x)=e^{K} Σ'_k F_k (P_k(b)−P_k(x)) を e^{K−t x}, e^{K−t b} に畳んで評価。
    denom = that * that + u * u
    Qx = (-that * np.cos(u * (x - a)) + u * np.sin(u * (x - a))) / denom
    Qb = (-that * np.cos(u * (b - a)) + u * np.sin(u * (b - a))) / denom
    exp_x = math.exp(K - that * x)       # 裾スケール (LR の指数の肩と同じ); 桁落ち無し
    exp_b = math.exp(K - that * b)       # b>x なので exp_b ≪ exp_x
    contrib = Fk * (exp_b * Qb - exp_x * Qx)
    contrib[0] *= 0.5                    # Σ' は k=0 を半分に
    return float(contrib.sum())


def tilted_cos_sf(
    grouped: GroupedHits, xs: np.ndarray, mean: float,
    support_lo: float, support_hi: float, t_cap: float | None = None,
) -> np.ndarray:
    """Esscher 傾斜 + COS の P(S>x) を xs 上に返す (各 x で t̂ を解くのでループ)。"""
    if t_cap is None:
        cmax = 0.0
        for _cnt, mix in grouped:
            cmax = max(cmax, max(abs(u.center) for u in mix))
        t_cap = 50.0 / cmax if cmax > 0 else 1.0
    out = np.empty(len(xs))
    for i, x in enumerate(xs):
        out[i] = tilted_cos_sf_scalar(grouped, float(x), t_cap,
                                      support_lo, support_hi)
    return out


# =============================================================================
# 素の COS 法 (非クリップ) の上側裾 — 深裾の桁落ちを可視化する用
# =============================================================================

def plain_cos_sf(
    grouped: GroupedHits, xs: np.ndarray, a: float, b: float, n_terms: int,
) -> np.ndarray:
    """素の COS 法 (原子分離ハイブリッド) の 1−F(x)。クリップしない生値を返す
    ので、深裾で「ほぼ1−ほぼ1」の桁落ちが雑音・負値として現れる。"""
    L = b - a
    k = np.arange(n_terms)
    u = k * np.pi / L
    av, ap = atom_part_distribution(grouped)
    has_atom = av.size > 0
    phi = sum_cf(grouped, u)
    if has_atom:
        phi = phi - atom_cf(grouped, u)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))
    xs = np.asarray(xs, dtype=float)
    dx = xs - a
    cdf = 0.5 * Fk[0] * dx
    arg = np.outer(dx, u[1:])
    cdf += (Fk[1:][None, :] * np.sin(arg) / u[1:][None, :]).sum(axis=1)
    if has_atom:
        cdf += _atom_step_cdf(av, ap, xs)
    return 1.0 - cdf                     # 生の 1−F (clip しない)


# =============================================================================
# 比較プロット
# =============================================================================

def make_deeptail_plot(sc: Scenario, n_mc: int, seed: int, output_path: str) -> None:
    print(f"\n=== {sc.name} ===")
    print(f"  Hit数={sc.n_hits}  平均={sc.mean:,.1f}  標準偏差={sc.std:,.1f}  "
          f"λ3={sc.lam3:.4f}  λ4={sc.lam4:.4f}")
    print(f"  サポート上限={sc.support_hi:,.0f}  (= mean+{(sc.support_hi-sc.mean)/sc.std:.1f}σ)")

    # 右裾 x グリッド: z=0.5 から「e^{K−t̂x} がアンダーフローしない」深さまで。
    # サポート上限の直前でクリップ (有界分布なので z_max は支持で決まる)。
    z_max = min(0.98 * (sc.support_hi - sc.mean) / sc.std, 40.0)
    zs = np.linspace(0.5, z_max, 240)
    xs = sc.mean + zs * sc.std

    print(f"  MC サンプリング {n_mc:,} 件 ...")
    samples = mc_survival_samples(sc.hit_mixtures, n_mc, seed)
    samples.sort()
    n = samples.size
    sf_mc = 1.0 - np.searchsorted(samples, xs, side="right") / n
    mc_count = n - np.searchsorted(samples, xs, side="right")
    mc_reliable = mc_count >= 50

    print("  Esscher 傾斜 + COS ...")
    sf_tilt = tilted_cos_sf(sc.grouped, xs, sc.mean, sc.support_lo, sc.support_hi)
    print("  Lugannani–Rice ...")
    sf_lr = lr_sf_array(sc.grouped, xs, sc.mean, sc.var)
    print("  素の COS (非クリップ) ...")
    sf_cos = plain_cos_sf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n)

    # ---- 自己検証: mid-tail (1e-3〜1e-9) で傾斜COS と 素COS が一致するか ----
    overlap = (sf_cos > 1e-9) & (sf_cos < 1e-3) & (sf_tilt > 0)
    if overlap.any():
        rel = np.abs(sf_tilt[overlap] - sf_cos[overlap]) / sf_cos[overlap]
        print(f"  [自己検証] mid-tail で傾斜COS vs 素COS 最大相対差 = {rel.max():.2e}")

    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )
    fig.suptitle(
        f"右裾 P(S>x): Esscher傾斜+COS vs 素COS vs Lugannani–Rice — {sc.name}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
        fontsize=13,
    )

    # 上段: 生存関数 (log y)
    ax.plot(zs, np.where(sf_tilt > 0, sf_tilt, np.nan), color="magenta", lw=2.0,
            label="Esscher傾斜 + COS (本手法)")
    ax.plot(zs, np.where(sf_lr > 0, sf_lr, np.nan), color="tab:green", lw=1.4,
            ls="--", label="Lugannani–Rice")
    ax.plot(zs, np.where(sf_cos > 0, sf_cos, np.nan), color="tab:cyan", lw=1.2,
            label="素の COS (1−F)")
    # 素 COS の桁落ち (負値) を×印で
    bad = sf_cos <= 0
    if bad.any():
        ax.plot(zs[bad], np.full(bad.sum(), 1e-300), "x", color="tab:cyan", ms=4)
    ax.plot(zs[mc_reliable], sf_mc[mc_reliable], "k.", ms=4,
            label=f"モンテカルロ ({n:,} 件)")
    ax.set_yscale("log")
    ax.set_ylabel("上側裾確率 P(S>x)")
    floor = max(1e-300, np.nanmin(sf_tilt[sf_tilt > 0]) * 0.1) if (sf_tilt > 0).any() else 1e-300
    ax.set_ylim(floor, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    # 下段: 傾斜COS を基準にした相対差 (LR・素COS・MC)
    eps = 1e-300
    ref = sf_tilt
    valid = ref > 0
    rel_lr = np.where(valid, (sf_lr - ref) / (ref + eps), np.nan)
    rel_cos = np.where(valid & (sf_cos > 0), (sf_cos - ref) / (ref + eps), np.nan)
    rel_mc = np.where(valid & mc_reliable, (sf_mc - ref) / (ref + eps), np.nan)
    ax_err.plot(zs, rel_lr, color="tab:green", lw=1.4, label="Lugannani–Rice")
    ax_err.plot(zs, rel_cos, color="tab:cyan", lw=1.2, label="素の COS")
    ax_err.plot(zs, rel_mc, "k.", ms=3, label="モンテカルロ")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.set_ylabel("相対差 (手法 − 傾斜COS) / 傾斜COS")
    ax_err.set_ylim(-0.6, 0.6)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper left", fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {output_path}")

    _print_deeptail_table(sc, samples)


def _print_deeptail_table(sc: Scenario, samples: np.ndarray) -> None:
    """代表的な深さ z で各手法の P(S>x) を「厳密に同じ x」で比較する。"""
    print("  右裾 P(S>x) 比較 (同一 x で評価):")
    print("    z      x            傾斜COS      素COS        LR           MC")
    n = samples.size
    z_targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0]
    for z in z_targets:
        x = sc.mean + z * sc.std
        if x >= sc.support_hi:
            continue
        xa = np.array([x])
        sf_t = tilted_cos_sf(sc.grouped, xa, sc.mean, sc.support_lo, sc.support_hi)[0]
        sf_c = plain_cos_sf(sc.grouped, xa, sc.cos_a, sc.cos_b, sc.cos_n)[0]
        sf_l = lr_sf_array(sc.grouped, xa, sc.mean, sc.var)[0]
        mc_c = n - np.searchsorted(samples, x, side="right")
        sf_m = mc_c / n
        mc_str = f"{sf_m:.3e}" if mc_c >= 30 else "   --   "
        print(f"    {z:4.1f}  {x:11,.0f}  {sf_t:.4e}  {sf_c:+.3e}  {sf_l:.4e}  {mc_str}")


# =============================================================================
# 設定 & main
# =============================================================================

N_MC = 20_000_000
SEED = 42
OUT_DIR = "experiments/output"


def main() -> None:
    scenarios = [
        (toy_scenario(), os.path.join(OUT_DIR, "tilted_cos_toy.png")),
        (real_scenario(), os.path.join(OUT_DIR, "tilted_cos_real.png")),
        (discrete_scenario(), os.path.join(OUT_DIR, "tilted_cos_discrete.png")),
    ]
    for sc, path in scenarios:
        make_deeptail_plot(sc, N_MC, SEED, path)
    print("\nDone.")


if __name__ == "__main__":
    main()
