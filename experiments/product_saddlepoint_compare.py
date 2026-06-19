"""HP依存ダメージ (積で表される分布) の裾確率: COS 法 vs Esscher+Lugannani-Rice。

docs/product.md (積→対数→和→COS) の発展。`product_cos.py` の COS 法は
バルク〜中程度裾では準厳密だが、F_D を級数和で求めるため極深裾 (1−F_D ≲ 10⁻¹²)
で「ほぼ1同士の差」の桁落ち床を持つ (docs/saddlepoint.md の和の場合と同じ)。

ここでは和の場合 (saddlepoint_compare.py) に持っている Esscher 傾斜 + Lugannani-Rice
を、積側の S = ln P = Σ ln Y_n に対しても適用する。S は独立和なので CGF は加法的で、
ln Y (Y~U(p,q) の対数 = 対数一様 = 切断指数) の MGF は閉形式:

    M_{lnY}(t) = E[Y^t] = (q^{t+1} − p^{t+1}) / ((q−p)(t+1)).

s = ln Y を中心 s_c=(ln p+ln q)/2・半幅 s_h=(ln q−ln p)/2 で表すと、saddlepoint_compare
の中心化一様の傾斜モーメント f0,f1,f2 がそのまま使える (u=(t+1)s_h):

    M_j   = e^{t s_c} · f0(u)/f0(s_h)
    M_j'  = e^{t s_c} · (s_c·ψ + s_h·f1(u)/f0(s_h))            ψ := f0(u)/f0(s_h)
    M_j'' = e^{t s_c} · (s_c²·ψ + 2 s_c·s_h·f1(u)/f0(s_h) + s_h²·f2(u)/f0(s_h))

(f0'=f1, f1'=f2 を使った。点質量 Y=c は s_h=0, M_j=e^{t ln c}。) あとは和 S の
サドルポイント方程式 K_S'(θ)=s* を解いて Lugannani-Rice。D=H̃_1(1−e^S) は S の単調
減少なので、P(D>x)=P(S≤s*(x)), P(D≤x)=P(S≥s*(x)) (s*(x)=ln(1−x/H̃_1))。深裾側は
LR の CDF 形 Φ(ŵ)−φ(ŵ)(1/û−1/ŵ) を「指数の肩」で直接表すので桁落ちしない。

実行例:
    uv run python -m experiments.product_saddlepoint_compare
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
from experiments.edgeworth_animation import Uniform  # noqa: E402
from experiments.saddlepoint_compare import _f012  # noqa: E402  (中心化一様 f0,f1,f2)
from experiments.product_cos import (  # noqa: E402
    HPScenario,
    base_mixture,
    damage_moments,
    damage_pdf_cdf,
    mc_damage,
    moments_hits,
    support_bounds_hits,
    y_mixture,
)

OUT_DIR = "experiments/output"
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# =============================================================================
# ln Y 混合の CGF (K, K', K'') — s = ln Y を中心化して f0,f1,f2 を再利用
# =============================================================================

def _s_center_halfwidth(comp: Uniform) -> tuple[float, float]:
    """Y~U(lo,hi) (lo>0) の s=ln Y の中心 s_c と半幅 s_h。点質量 (hi<=lo) は s_h=0。"""
    if comp.hi <= comp.lo:
        return math.log(comp.lo), 0.0
    a, b = math.log(comp.lo), math.log(comp.hi)
    return 0.5 * (a + b), 0.5 * (b - a)


def _lny_K_derivs(ymix: list[Uniform], t: float) -> tuple[float, float, float]:
    """1Hit の ln Y 混合の (K(t), K'(t), K''(t)) を数値安定に返す。

    M_j=e^{t s_c}·ψ_j (ψ_j=f0(u)/f0(s_h), u=(t+1)s_h) で最大中心 s_cmax を括り出し、
    比で K=t·s_cmax+ln A0, K'=A1/A0, K''=A2/A0−(A1/A0)² を取る (オーバーフロー無し)。"""
    scs = [_s_center_halfwidth(c) for c in ymix]
    s_cmax = max(sc for sc, _ in scs)
    A0 = A1 = A2 = 0.0
    for comp, (s_c, s_h) in zip(ymix, scs):
        w = comp.weight
        if s_h == 0.0:                       # 点質量 Y=c: ψ=1, ψ'=ψ''=0
            psi = 1.0
            psi1 = 0.0
            psi2 = 0.0
        else:
            u = (t + 1.0) * s_h
            f0u, f1u, f2u = (float(v) for v in _f012(u))
            D = float(_f012(s_h)[0])         # f0(s_h)
            psi = f0u / D                    # ψ        = f0(u)/f0(s_h)
            psi1 = s_h * f1u / D             # dψ/dt    = s_h·f1(u)/f0(s_h)
            psi2 = s_h * s_h * f2u / D       # d²ψ/dt²  = s_h²·f2(u)/f0(s_h)
        e = math.exp(t * (s_c - s_cmax))     # ≤ 1 (s_c≤s_cmax) ; 大 |t| でも安全側
        A0 += w * e * psi
        A1 += w * e * (s_c * psi + psi1)
        A2 += w * e * (s_c * s_c * psi + 2.0 * s_c * psi1 + psi2)
    Kp = A1 / A0
    Kpp = A2 / A0 - Kp * Kp
    K = t * s_cmax + math.log(A0)
    return K, Kp, Kpp


def cgf_S_derivs(ymix_per_hit: list[list[Uniform]], t: float) -> tuple[float, float, float]:
    """S = Σ ln Y_n の (K_S(t), K_S'(t), K_S''(t)) = Σ_hit。"""
    K = Kp = Kpp = 0.0
    for ymix in ymix_per_hit:
        ki, kp, kpp = _lny_K_derivs(ymix, t)
        K += ki
        Kp += kp
        Kpp += kpp
    return K, Kp, Kpp


def _t_cap(ymix_per_hit: list[list[Uniform]]) -> float:
    """e^{t(s_c−s_cmax)} と sinh((t+1)s_h) が共に有限に収まる θ の上限。"""
    max_sh = 0.0
    max_dc = 0.0
    for ymix in ymix_per_hit:
        scs = [_s_center_halfwidth(c) for c in ymix]
        s_cmax = max(sc for sc, _ in scs)
        for s_c, s_h in scs:
            max_sh = max(max_sh, s_h)
            max_dc = max(max_dc, abs(s_c - s_cmax))
    cap_sh = 350.0 / max_sh if max_sh > 0 else math.inf      # sinh の肩 < 350
    cap_dc = 600.0 / max_dc if max_dc > 0 else math.inf      # exp の肩 < 600
    return min(cap_sh, cap_dc)


def solve_saddlepoint_S(
    ymix_per_hit: list[list[Uniform]], s_star: float, t_cap: float,
    tol: float = 1e-12, max_iter: int = 100,
) -> float:
    """K_S'(θ)=s_star を safeguarded Newton で解く。K_S' は狭義単調増加。"""
    lo, hi = -t_cap, t_cap
    _, kp_lo, _ = cgf_S_derivs(ymix_per_hit, lo)
    _, kp_hi, _ = cgf_S_derivs(ymix_per_hit, hi)
    if s_star <= kp_lo:
        return lo
    if s_star >= kp_hi:
        return hi
    t = 0.0
    _, kp, kpp = cgf_S_derivs(ymix_per_hit, t)
    for _ in range(max_iter):
        f = kp - s_star
        if f > 0:
            hi = t
        else:
            lo = t
        if abs(f) < tol * max(1.0, abs(s_star)):
            return t
        t_new = t - f / kpp if kpp > 0 else 0.5 * (lo + hi)
        if not (lo < t_new < hi):
            t_new = 0.5 * (lo + hi)
        t = t_new
        _, kp, kpp = cgf_S_derivs(ymix_per_hit, t)
    return t


# =============================================================================
# Lugannani-Rice: S の左裾 CDF と右裾 SF を「指数の肩」で直接 (桁落ち無し)
# =============================================================================

def _lr_wu(ymix_per_hit, s_star, t_cap):
    """サドルポイント θ̂ と (ŵ, û) を返す。台外・平均直近は None。"""
    that = solve_saddlepoint_S(ymix_per_hit, s_star, t_cap)
    K, Kp, Kpp = cgf_S_derivs(ymix_per_hit, that)
    arg = 2.0 * (that * s_star - K)
    if arg <= 0.0 or Kpp <= 0.0 or abs(that) < 1e-300:
        return None
    w = math.copysign(math.sqrt(arg), that)
    uu = that * math.sqrt(Kpp)
    return w, uu


def lr_S_cdf(ymix_per_hit, s_star, mean_S, t_cap):
    """P(S ≤ s_star) を Lugannani-Rice の CDF 形で。左裾 (s*<mean) で桁落ち無し。
        P(S≤x) = Φ(ŵ) − φ(ŵ)(1/û − 1/ŵ).
    平均直近は不定形なので正規近似で繋ぐ (裾精度には無関係)。"""
    wu = _lr_wu(ymix_per_hit, s_star, t_cap)
    if wu is None:
        _, _, Kpp0 = cgf_S_derivs(ymix_per_hit, 0.0)
        return 0.5 * (1.0 + math.erf((s_star - mean_S) / math.sqrt(2.0 * Kpp0)))
    w, uu = wu
    if abs(w) < 1e-6:
        _, _, Kpp0 = cgf_S_derivs(ymix_per_hit, 0.0)
        return 0.5 * (1.0 + math.erf((s_star - mean_S) / math.sqrt(2.0 * Kpp0)))
    Phi = 0.5 * (1.0 + math.erf(w / math.sqrt(2.0)))
    phi = math.exp(-0.5 * w * w) / _SQRT_2PI
    return Phi - phi * (1.0 / uu - 1.0 / w)


def lr_S_sf(ymix_per_hit, s_star, mean_S, t_cap):
    """P(S ≥ s_star) を Lugannani-Rice の SF 形で。右裾 (s*>mean) で桁落ち無し。
        P(S>x) = 1 − Φ(ŵ) + φ(ŵ)(1/û − 1/ŵ)."""
    wu = _lr_wu(ymix_per_hit, s_star, t_cap)
    if wu is None:
        _, _, Kpp0 = cgf_S_derivs(ymix_per_hit, 0.0)
        return 0.5 * (1.0 - math.erf((s_star - mean_S) / math.sqrt(2.0 * Kpp0)))
    w, uu = wu
    if abs(w) < 1e-6:
        _, _, Kpp0 = cgf_S_derivs(ymix_per_hit, 0.0)
        return 0.5 * (1.0 - math.erf((s_star - mean_S) / math.sqrt(2.0 * Kpp0)))
    Phi = 0.5 * (1.0 + math.erf(w / math.sqrt(2.0)))
    phi = math.exp(-0.5 * w * w) / _SQRT_2PI
    return (1.0 - Phi) + phi * (1.0 / uu - 1.0 / w)


# =============================================================================
# Esscher + COS (傾斜COS): 評価点ごとに θ̂ 傾斜 → 傾斜密度 f_θ を COS 反転 → untilt
#
#   f_θ(s) = e^{θs} f_S(s) / M(θ),  M(θ)=e^{K(θ)} は台 [A,B] を保つ準厳密密度。
#   その CF は閉形式 φ_θ(u) = φ_S(u−iθ)/M(θ) = Π_n φ_{lnY_n}(u−iθ)/M_n(θ) で、
#   1Hit 成分 Y~U(p,q) は φ_{lnY}(u−iθ) ∝ (q^{θ+1}e^{iu ln q} − p^{θ+1}e^{iu ln p})/(1+θ+iu)。
#
#   裾確率は untilt して
#       P(S≤s*) = M(θ)∫_A^{s*} e^{−θs} f_θ ds = e^{K(θ)−θs*} ∫_A^{s*} e^{−θ(s−s*)} f_θ ds.
#   後半の積分 J は f_θ の COS 係数 G_k と ∫e^{−θ(s−s*)}cos(u_k(s−A))ds の閉形式で
#   O(1) として求まり、裾の小ささは前因子 e^{K(θ)−θs*} に分離される。plain COS が
#   F_S(s*) を「O(0.1) の級数を 10⁻¹⁴ まで引き算消去」して出すのと違い、消去が無い。
# =============================================================================

_TCOS_TERMS_PER_SIGMA = 24
_TCOS_N_MIN = 1024
_TCOS_N_MAX = 1 << 16


def _tilted_hit_cf(ymix: list[Uniform], theta: float, u: np.ndarray) -> np.ndarray:
    """1Hit の ln Y の θ 傾斜 CF φ_θ,n(u)=Π成分/M_n を返す (u=0 で 1 に正規化)。

    各成分の上端 ln(q) の最大 smax で e^{(θ+1)smax} を括り出して比を取る (桁あふれ回避)。"""
    tp1 = theta + 1.0
    smax = max((math.log(c.hi) if c.hi > c.lo else math.log(c.lo)) for c in ymix)
    num = np.zeros_like(u, dtype=complex)
    M = 0.0
    for c in ymix:
        w = c.weight
        if c.hi <= c.lo:                                   # 点質量 Y=c
            lc = math.log(c.lo)
            base = math.exp(theta * lc - tp1 * smax)       # m0_red = e^{θ lc}/e^{(θ+1)smax}
            num += w * base * np.exp(1j * u * lc)
            M += w * base
        else:
            p, q = c.lo, c.hi
            lp, lq = math.log(p), math.log(q)
            Eq = math.exp(tp1 * (lq - smax))               # q^{θ+1}/e^{(θ+1)smax}
            Ep = math.exp(tp1 * (lp - smax))
            num += w * (Eq * np.exp(1j * u * lq) - Ep * np.exp(1j * u * lp)) \
                / ((q - p) * (1.0 + theta + 1j * u))
            M += w * (Eq - Ep) / ((q - p) * tp1)
    return num / M


def tilted_cos_S(ymix_per_hit, s_star, A, B, mean_S, t_cap):
    """傾斜COS で S の裾確率を返す: (値, side)。
    side='lower' なら値=P(S≤s*) (左裾), 'upper' なら値=P(S≥s*) (右裾)。"""
    theta = solve_saddlepoint_S(ymix_per_hit, s_star, t_cap)
    K, _Kp, Kpp = cgf_S_derivs(ymix_per_hit, theta)
    if not (Kpp > 0.0) or abs(theta) < 1e-9:               # 平均直近: 傾斜の意味が薄い
        return None, None
    std_t = math.sqrt(Kpp)
    L = B - A
    n_terms = int(math.ceil(_TCOS_TERMS_PER_SIGMA * L / std_t))
    n_terms = max(_TCOS_N_MIN, min(_TCOS_N_MAX, n_terms))

    k = np.arange(n_terms)
    u = k * np.pi / L
    phi = np.ones_like(u, dtype=complex)
    for ymix in ymix_per_hit:
        phi = phi * _tilted_hit_cf(ymix, theta, u)
    Gk = (2.0 / L) * np.real(phi * np.exp(-1j * u * A))     # f_θ の COS 係数

    # 積分区間: 左裾は [A, s*] で P(S≤s*)、右裾は [s*, B] で P(S≥s*)
    side = "lower" if theta < 0.0 else "upper"
    alpha, beta = (A, s_star) if side == "lower" else (s_star, B)

    def psi(s: float) -> np.ndarray:
        """Ψ_k(s)=e^{−θ(s−s*)}(−θ cos(u_k(s−A))+u_k sin(u_k(s−A)))/(θ²+u_k²)。k=0 は −e/θ。"""
        e = math.exp(-theta * (s - s_star))
        out = np.empty_like(u)
        out[0] = -e / theta                                # u_0=0
        d = theta * theta + u[1:] ** 2
        ang = u[1:] * (s - A)
        out[1:] = e * (-theta * np.cos(ang) + u[1:] * np.sin(ang)) / d
        return out

    Tk = psi(beta) - psi(alpha)
    Gk2 = Gk.copy()
    Gk2[0] *= 0.5                                          # Σ' は k=0 を半分に
    J = float(np.dot(Gk2, Tk))
    tail = math.exp(K - theta * s_star) * J                # = P(S≤s*) or P(S≥s*)
    return tail, side


def tilted_cos_damage_two_sided_tail(sc: HPScenario, xs: np.ndarray) -> np.ndarray:
    """両側裾確率 min(P(D≤x), P(D>x)) を傾斜COS で xs 上に返す。
    P(D>x)=P(S≤s*) (左裾, θ<0), P(D≤x)=P(S≥s*) (右裾, θ>0) を side に従い採用。"""
    ymix = y_mixture(sc.base_mixture, sc.beta)
    ymix_per_hit = [ymix] * sc.n_hits
    mean_S, _ = moments_hits(ymix_per_hit)
    A, B = support_bounds_hits(ymix_per_hit)
    t_cap = _t_cap(ymix_per_hit)
    out = np.full(len(xs), np.nan)
    for i, x in enumerate(np.asarray(xs, dtype=float)):
        arg = 1.0 - x / sc.Htil
        if not (x >= 0.0 and arg > 0.0):
            out[i] = 0.0
            continue
        s_star = math.log1p(-x / sc.Htil)
        if not (A < s_star < B):
            out[i] = 0.0
            continue
        val, _side = tilted_cos_S(ymix_per_hit, s_star, A, B, mean_S, t_cap)
        if val is not None and math.isfinite(val) and val > 0:
            out[i] = val
    return out


def lr_damage_two_sided_tail(sc: HPScenario, xs: np.ndarray) -> np.ndarray:
    """両側裾確率 min(P(D≤x), P(D>x)) を Esscher+LR で xs 上に返す。

    s*(x)=ln(1−x/H̃_1)。D は S の単調減少なので
        P(D>x)=P(S≤s*)  (上側ダメージ裾, s*<mean_S, LR-CDF が小さく桁落ち無し)
        P(D≤x)=P(S≥s*)  (下側ダメージ裾, s*>mean_S, LR-SF が小さく桁落ち無し)。
    min を取れば各点で「小さい側 = その裾で桁落ちしない推定量」が自動採用される。"""
    ymix = y_mixture(sc.base_mixture, sc.beta)
    ymix_per_hit = [ymix] * sc.n_hits
    mean_S, _ = moments_hits(ymix_per_hit)
    t_cap = _t_cap(ymix_per_hit)
    out = np.empty(len(xs))
    for i, x in enumerate(np.asarray(xs, dtype=float)):
        arg = 1.0 - x / sc.Htil
        if not (x >= 0.0 and arg > 0.0):           # 台外
            out[i] = 0.0
            continue
        s_star = math.log1p(-x / sc.Htil)
        p_upper_dmg = lr_S_cdf(ymix_per_hit, s_star, mean_S, t_cap)  # P(D>x)
        p_lower_dmg = lr_S_sf(ymix_per_hit, s_star, mean_S, t_cap)   # P(D≤x)
        out[i] = min(max(p_upper_dmg, 0.0), max(p_lower_dmg, 0.0))
    return out


# =============================================================================
# 比較
# =============================================================================

def run(sc: HPScenario, output_path: str, n_mc: int = 4_000_000) -> None:
    print(f"[{sc.name}] N={sc.n_hits} hits, H={sc.H:,.0f}, H1={sc.H1:,.0f}, "
          f"R0={sc.R0}, R1={sc.R1}, H̃1={sc.Htil:,.0f}")

    ymix = y_mixture(sc.base_mixture, sc.beta)
    ymix_per_hit = [ymix] * sc.n_hits
    A, B = support_bounds_hits(ymix_per_hit)
    mean_exact, var_exact = damage_moments(ymix_per_hit, sc.Htil)
    std_exact = math.sqrt(var_exact)
    d_max = -sc.Htil * math.expm1(A)            # D_max = H̃_1(1−e^A)

    # --- CGF 健全性: K_S'(0)=E[S], K_S''(0)=Var[S] が解析と一致するか ---
    mean_S, var_S = moments_hits(ymix_per_hit)
    K0, Kp0, Kpp0 = cgf_S_derivs(ymix_per_hit, 0.0)
    print(f"  CGF健全性: K_S(0)={K0:.2e} (→0), "
          f"K_S'(0)={Kp0:.6f} vs E[S]={mean_S:.6f} (相対 {abs(Kp0-mean_S)/abs(mean_S):.1e}), "
          f"K_S''(0)={Kpp0:.3e} vs Var[S]={var_S:.3e} (相対 {abs(Kpp0-var_S)/var_S:.1e})")
    print(f"  台 S=[{A:.4f}, {B:.4f}], E[D]={mean_exact:,.0f}, SD[D]={std_exact:,.0f}, "
          f"D_max={d_max:,.0f} (z_max={(d_max-mean_exact)/std_exact:.2f})")

    # --- MC 真値 ---
    rng = np.random.default_rng(20260602)
    samples = mc_damage(sc, n_mc, rng)
    sorted_s = np.sort(samples)

    # =========================================================================
    # 深裾テーブル: 上側ダメージ裾 P(D>x) を z=2,3,4,5,6 で COS/LR/MC 比較
    # =========================================================================
    print("\n  上側ダメージ裾 P(D>x) — plain COS vs 傾斜COS vs Esscher+LR vs MC")
    print(f"  {'z':>4} {'x (=E+zσ)':>12} {'plainCOS':>11} {'傾斜COS':>11} "
          f"{'LR':>11} {'MC':>11}  注")
    z_targets = [2.0, 3.0, 4.0, 5.0, 6.0, 6.5]
    for z in z_targets:
        x = mean_exact + z * std_exact
        if x >= d_max:
            print(f"  {z:>4.1f} {x:>12,.0f}  --- 台外 (x≥D_max) ---")
            continue
        _, F_cos = damage_pdf_cdf(sc, np.array([x]))   # plain COS の F_D(x)
        plain_tail = float(1.0 - F_cos[0])             # P(D>x); 深裾で桁落ち
        tcos_tail = float(tilted_cos_damage_two_sided_tail(sc, np.array([x]))[0])
        lr_tail = float(lr_damage_two_sided_tail(sc, np.array([x]))[0])
        mc_tail = float(np.count_nonzero(samples > x)) / n_mc
        mc_str = f"{mc_tail:.3e}" if mc_tail > 0 else f"<{1.0/n_mc:.0e}"
        note = ""
        if plain_tail <= 0 or plain_tail < 1e-13:
            note = "← plain COS 桁落ち"
        print(f"  {z:>4.1f} {x:>12,.0f} {plain_tail:>11.3e} {tcos_tail:>11.3e} "
              f"{lr_tail:>11.3e} {mc_str:>11}  {note}")

    # =========================================================================
    # プロット: 両側裾確率 (標準化 z 軸, 片対数)
    # =========================================================================
    z_lo = (max(0.0, sorted_s[0]) - mean_exact) / std_exact
    z_hi = (d_max - mean_exact) / std_exact
    xs = mean_exact + np.linspace(z_lo, z_hi * 0.9995, 700) * std_exact
    zs = (xs - mean_exact) / std_exact

    _, F_cos_arr = damage_pdf_cdf(sc, xs)
    cos_tail = np.minimum(F_cos_arr, 1.0 - F_cos_arr)
    tcos_tail = tilted_cos_damage_two_sided_tail(sc, xs)
    lr_tail = lr_damage_two_sided_tail(sc, xs)
    mc_cdf = np.searchsorted(sorted_s, xs, side="right") / n_mc
    mc_tail = np.minimum(mc_cdf, 1.0 - mc_cdf)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(zs, np.where(mc_tail > 0, mc_tail, np.nan),
                color="0.5", lw=3.0, label="MC 真値")
    ax.semilogy(zs, np.where(cos_tail > 1e-16, cos_tail, np.nan),
                color="magenta", lw=1.8, ls="-", label="plain COS (product_cos)")
    ax.semilogy(zs, np.where(tcos_tail > 0, tcos_tail, np.nan),
                color="green", lw=1.6, ls="-", label="Esscher + COS (傾斜COS)")
    ax.semilogy(zs, np.where(lr_tail > 0, lr_tail, np.nan),
                color="navy", lw=1.6, ls="--", label="Esscher + Lugannani-Rice")
    ax.axhline(1.0 / n_mc, color="0.7", lw=0.8, ls=":", label=f"MC 分解能 1/{n_mc:.0e}")
    ax.set_title(f"HP依存ダメージ 両側裾確率 — {sc.name}\n"
                 f"D = H̃₁(1 − Π(1 − β x)),  S = ln P を COS 反転 / Esscher 傾斜+LR")
    ax.set_xlabel("標準化ダメージ z = (D − E[D]) / σ")
    ax.set_ylabel("min(P(D≤x), P(D>x))")
    ax.set_ylim(1e-30, 1.0)
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  保存: {output_path}")


def main() -> None:
    sc = HPScenario(
        name="ミカ (R1=2, R0=1)",
        H=1_000_000.0,
        H1=1_000_000.0,
        R0=1.0,
        R1=2.0,
        n_hits=20,
        base_mixture=base_mixture(
            normal_lo=8_000.0, normal_hi=12_000.0,
            crit_lo=16_000.0, crit_hi=24_000.0,
            crit_rate=0.5, evade_rate=0.0,
        ),
    )
    run(sc, os.path.join(OUT_DIR, "product_saddlepoint_mika.png"))


if __name__ == "__main__":
    main()
