"""合計ダメージ S_n の裾確率を出す各種「特性関数/CGF ベース」手法。

docs/saddlepoint.md・docs/gilpelaez.md の理論を実装する:

  - Gil-Pelaez 反転公式 (1951): CDF を特性関数 φ から直接フーリエ反転。
        F(x) = 1/2 − (1/π) ∫_0^∞ Im[e^{−itx} φ(t)] / t dt
    COS 法 (有界余弦級数版) と数学的に同一で、被積分は φ/t ~ t^{−(n+1)} と速く
    減衰する。原子が無ければ機械精度近くまで一致する (本シナリオは回避 0%)。

  - Lugannani–Rice (1980) サドルポイント近似: CGF K=Σ log M_i を使い、各 x で
    サドルポイント方程式 K'(t̂)=x を解いて
        ŵ = sgn(t̂)√(2(t̂x − K(t̂))),  û = t̂√(K''(t̂))
        P(S>x) ≈ 1 − Φ(ŵ) + φ(ŵ)(1/û − 1/ŵ)
    指数の肩で裾を直接表すため極深裾でも桁落ちしない (が、漸近近似なので中腹で
    区分多項式構造を均す系統誤差を持つ)。

Scenario / cos_cdf / mc_survival_samples は cos_compare から再エクスポートする
(docs と experiments.lr_deeptail が experiments.saddlepoint_compare を参照するため)。
"""
from __future__ import annotations

import math

import numpy as np

from experiments.cos_compare import (  # noqa: F401  (re-export)
    GroupedHits,
    Scenario,
    cos_cdf,
    cos_pdf,
    mc_survival_samples,
    sum_cf,
)
from experiments.edgeworth_animation import Uniform

_SQRT_2PI = math.sqrt(2.0 * math.pi)


# =============================================================================
# Gil-Pelaez 反転 (CDF を φ から直接)
# =============================================================================

def gil_pelaez_cdf(
    grouped: GroupedHits, xs: np.ndarray,
    mean: float, support_lo: float, support_hi: float,
    h_cut: float = 50.0, pts_per_period: int = 16,
) -> np.ndarray:
    """Gil-Pelaez 反転公式で CDF F(x)=P(S_n≤x) を xs 上に返す。

    F(x) = 1/2 − (1/π) ∫_0^∞ Im[e^{−itx} φ(t)] / t dt を有限区間 [0, T] で台形
    積分する。φ(t)=Π_hit φ_hit(t) は **各 Hit の半幅 h** が定める sinc 包絡
    (∏ 1/(t h_i) ~ t^{−n}) で減衰するので、上限は最小半幅基準 T = h_cut/h_min
    に取る (σ 基準だと早すぎる打ち切りになる)。刻みは最速位相 (t·support_hi) を
    pts_per_period 点で解像するよう取る。

    Returns: F(xs) (clip しない素の値; 桁落ちは COS と同様に深裾で出る)
    """
    xs = np.asarray(xs, dtype=float)
    # 最小半幅 (sinc 包絡の減衰スケール) と最速振動 (位相 t·x の最大 x)
    h_min = min(
        min((u.half_width for u in mix if u.half_width > 0.0), default=np.inf)
        for _cnt, mix in grouped
    )
    t_max = h_cut / h_min
    fastest = max(abs(support_hi), abs(support_lo), abs(mean))
    dt = (2.0 * math.pi / fastest) / pts_per_period
    n_t = max(4096, int(t_max / dt) + 1)
    t = np.linspace(0.0, t_max, n_t)
    dt = t[1] - t[0]

    phi = sum_cf(grouped, t)            # φ(t) (len n_t, 複素)
    out = np.empty_like(xs)
    inv_pi = 1.0 / math.pi
    t_nz = t[1:]
    phi_nz = phi[1:]
    for i, x in enumerate(xs):
        integ = np.empty(n_t)
        integ[0] = mean - x                                   # t→0 の極限
        integ[1:] = np.imag(np.exp(-1j * t_nz * x) * phi_nz) / t_nz
        area = np.trapezoid(integ, dx=dt)                     # numpy2: trapz→trapezoid
        out[i] = 0.5 - inv_pi * area
    return out


# =============================================================================
# CGF (一様混合) と Lugannani–Rice サドルポイント近似
# =============================================================================

def _f012(u: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """中心化一様 V~U(−h,h) の傾斜モーメント関数 f0,f1,f2 (引数 u=t·h)。

        f0 = sinh u / u
        f1 = (u cosh u − sinh u) / u²
        f2 = (u² sinh u − 2u cosh u + 2 sinh u) / u³
    小 |u| は分子で桁落ちするので Taylor (docs/saddlepoint.md)。"""
    u = np.asarray(u, dtype=float)
    small = np.abs(u) < 0.5
    f0 = np.empty_like(u)
    f1 = np.empty_like(u)
    f2 = np.empty_like(u)
    # 大 |u|: 閉形式
    us = np.where(small, 1.0, u)           # small 位置のゼロ割回避用ダミー
    sh = np.sinh(us)
    ch = np.cosh(us)
    f0_big = sh / us
    f1_big = (us * ch - sh) / us ** 2
    f2_big = (us ** 2 * sh - 2.0 * us * ch + 2.0 * sh) / us ** 3
    # 小 |u|: Taylor
    u2 = u * u
    f0_sm = 1.0 + u2 / 6.0 + u2 ** 2 / 120.0 + u2 ** 3 / 5040.0
    f1_sm = u / 3.0 + u * u2 / 30.0 + u * u2 ** 2 / 840.0
    f2_sm = 1.0 / 3.0 + u2 / 10.0 + u2 ** 2 / 168.0
    f0 = np.where(small, f0_sm, f0_big)
    f1 = np.where(small, f1_sm, f1_big)
    f2 = np.where(small, f2_sm, f2_big)
    return f0, f1, f2


def _mixture_K_derivs(mixture: list[Uniform], t: float) -> tuple[float, float, float]:
    """1Hit 混合の (K_i(t), K_i'(t), K_i''(t)) を数値安定に返す。

    各成分 M_j=e^{tc}f0(th) を、混合内の最大中心 cmax で e^{t·cmax} を括り出して
    比 (A1/A0, A2/A0−(A1/A0)²) を取るのでオーバーフロー無し (e^{t(c−cmax)}≤1)。"""
    cmax = max(u.center for u in mixture)
    A0 = A1 = A2 = 0.0
    for u in mixture:
        w = u.weight
        c = u.center
        h = u.half_width
        e = math.exp(t * (c - cmax))
        if h == 0.0:
            f0, f1, f2 = 1.0, 0.0, 0.0
        else:
            uh = t * h
            f0, f1, f2 = (float(v) for v in _f012(uh))
        A0 += w * e * f0
        A1 += w * e * (c * f0 + h * f1)
        A2 += w * e * (c * c * f0 + 2.0 * c * h * f1 + h * h * f2)
    Kp = A1 / A0
    Kpp = A2 / A0 - Kp * Kp
    Ki = t * cmax + math.log(A0)
    return Ki, Kp, Kpp


def cgf_derivs(grouped: GroupedHits, t: float) -> tuple[float, float, float]:
    """和 S_n の (K(t), K'(t), K''(t)) = Σ_hit (cnt 倍)。"""
    K = Kp = Kpp = 0.0
    for cnt, mix in grouped:
        ki, kp, kpp = _mixture_K_derivs(mix, t)
        K += cnt * ki
        Kp += cnt * kp
        Kpp += cnt * kpp
    return K, Kp, Kpp


def solve_saddlepoint(
    grouped: GroupedHits, x: float, t_cap: float,
    tol: float = 1e-12, max_iter: int = 100,
) -> float:
    """K'(t̂)=x を safeguarded Newton (ブラケット二分フォールバック) で解く。
    K' は狭義単調増加。ブラケットを [lo,hi] に拡張してから Newton/二分。"""
    # ブラケット拡張
    lo, hi = -t_cap, t_cap
    _, kp_lo, _ = cgf_derivs(grouped, lo)
    _, kp_hi, _ = cgf_derivs(grouped, hi)
    if x <= kp_lo:
        return lo
    if x >= kp_hi:
        return hi
    t = 0.0
    _, kp, kpp = cgf_derivs(grouped, t)
    for _ in range(max_iter):
        f = kp - x
        if f > 0:
            hi = t
        else:
            lo = t
        if abs(f) < tol * max(1.0, abs(x)):
            return t
        # Newton 候補
        t_new = t - f / kpp if kpp > 0 else 0.5 * (lo + hi)
        if not (lo < t_new < hi):
            t_new = 0.5 * (lo + hi)           # 範囲外なら二分
        t = t_new
        _, kp, kpp = cgf_derivs(grouped, t)
    return t


def lugannani_rice_sf(
    grouped: GroupedHits, x: float, mean: float, var: float, t_cap: float,
) -> float:
    """Lugannani–Rice による生存関数 P(S_n>x) (スカラー)。

    平均ごく近傍は ŵ,û→0 の 0/0 不定形なので、極限値
        1/2 − λ3/(6√(2π)),  λ3 = K'''(0)/K''(0)^{3/2}
    で繋ぐ (裾の精度には影響しない)。"""
    std = math.sqrt(var)
    if abs(x - mean) < 1e-6 * std:
        # 平均近傍の極限。K'''(0)=κ3 を数値微分でなく λ3 で近似
        # (呼び出し側は通常裾で使うのでここは保険)。
        lam3 = _lambda3_at_zero(grouped, var)
        return 0.5 - lam3 / (6.0 * _SQRT_2PI)
    that = solve_saddlepoint(grouped, x, t_cap)
    K, Kp, Kpp = cgf_derivs(grouped, that)
    arg = 2.0 * (that * x - K)
    if arg <= 0.0:
        # 数値的に負になりうる平均直近: 極限へ
        lam3 = _lambda3_at_zero(grouped, var)
        return 0.5 - lam3 / (6.0 * _SQRT_2PI)
    w = math.copysign(math.sqrt(arg), that)
    uu = that * math.sqrt(Kpp)
    if abs(w) < 1e-6:
        lam3 = _lambda3_at_zero(grouped, var)
        return 0.5 - lam3 / (6.0 * _SQRT_2PI)
    Phi = 0.5 * (1.0 + math.erf(w / math.sqrt(2.0)))
    phi = math.exp(-0.5 * w * w) / _SQRT_2PI
    return (1.0 - Phi) + phi * (1.0 / uu - 1.0 / w)


def _lambda3_at_zero(grouped: GroupedHits, var: float) -> float:
    """K'''(0)/K''(0)^{3/2} を t=0 まわりの中心差分で評価 (平均近傍の極限用)。"""
    h = 1e-3 / max(1.0, math.sqrt(var))
    _, _, kpp_p = cgf_derivs(grouped, h)
    _, _, kpp_m = cgf_derivs(grouped, -h)
    k3 = (kpp_p - kpp_m) / (2.0 * h)          # K'''(0) ≈ dK''/dt|0
    return k3 / var ** 1.5


def lr_sf_array(
    grouped: GroupedHits, xs: np.ndarray, mean: float, var: float,
    t_cap: float | None = None,
) -> np.ndarray:
    """Lugannani–Rice の生存関数 P(S>x) を xs 上に返す。"""
    if t_cap is None:
        cmax = 0.0
        for _cnt, mix in grouped:
            cmax = max(cmax, max(abs(u.center) for u in mix))
        t_cap = 50.0 / cmax if cmax > 0 else 1.0
    return np.array([
        lugannani_rice_sf(grouped, float(x), mean, var, t_cap) for x in xs
    ])
