"""積/HP依存ダメージ (ゲブラ等) の厳密解 = 指数傾斜 Irwin–Hall。

product_cos.py のモデルでは累積ダメージ D = H̃_1(1 − P)、P = Π_n Y_n で、
S = ln P = Σ_n ln Y_n の和を COS 反転している。各 Y_n は一様混合 U(p,q) なので
被加算項 ln Y_n は **対数一様 (= 切断指数)** であって一様ではない。よって素の
Irwin–Hall (一様和=区分多項式) は使えないが、次の閉形式が成り立つ:

  組合せ c (各 Hit が成分を1つ選ぶ) で条件付けると、ln Y_i ~ U(α_i,γ_i) の対数
  (α=ln p, γ=ln q, 幅 L=ln(q/p)) の和 S_c の密度は

      f_{S_c}(s) = const_c · e^{s} · f_IH(s − A_c; {L_i}),
          A_c = Σ ln p_i,  const_c = Π L_i / Π (q_i − p_i)

  すなわち **e^{s} で指数傾斜した一般化 Irwin–Hall**。f_IH は irwinhall_exact の
  切断べき乗そのもの。CDF は e^{±t}·(t−σ)^{m−1} の不完全ガンマ積分で閉形式:

      G(τ)  = ∫_0^τ e^{ t} f_IH(t) dt   (下側; 低ダメージ側で安定)
      H(ξ)  = ∫_0^ξ e^{−r} f_IH(r) dr   (反射 ∫_τ^{ΣL} e^t f_IH = e^{ΣL}H(ΣL−τ);
                                          高ダメージ側=上側裾で桁落ちなく安定)

混合全体は重み W_c=Π(混合重み) で加算。これは COS とは独立な真の厳密解で、
ゲブラカリン (7Hit・各2成分・回避0% → 128 組合せ) で COS の検証基準になる。

    uv run python -m experiments.product_irwinhall_exact
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from experiments.edgeworth_animation import Uniform
from experiments.irwinhall_exact import _truncated_power_pdf

_ATOM_TOL = 1e-15      # 幅 L=ln(q/p) がこれ以下なら点質量 (s 空間, O(0.1) スケール)
_SERIES_MAX = 256      # P_k/Q_k 級数の最大項数 (この問題は x~0.1 で十分速く収束)


# =============================================================================
# 不完全ガンマ的積分 P_k(x)=∫_0^x u^k e^{u} du, Q_k(x)=∫_0^x u^k e^{−u} du
#   この問題の x = τ−σ は O(0.1) と小さいので、正項/交代の整級数で機械精度。
#   P_k(x) = Σ_{j≥0} x^{k+1+j} / ((k+1+j) j!)
#   Q_k(x) = Σ_{j≥0} (−1)^j x^{k+1+j} / ((k+1+j) j!)
# =============================================================================

def _Pk_series(x: np.ndarray, k: int, sign: float) -> np.ndarray:
    """∫_0^x u^k e^{sign·u} du を整級数で。sign=+1→P_k, sign=−1→Q_k。x>=0 前提。"""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    term = x ** (k + 1) / (k + 1)          # j=0 項
    j = 0
    while True:
        out = out + (sign ** j) * term
        # 次項: term_{j+1} = term_j · x/(j+1) · (k+1+j)/(k+2+j)
        j += 1
        if j > _SERIES_MAX:
            break
        term = term * x / j * (k + j) / (k + j + 1)
        if np.all(term <= 1e-18 * (np.abs(out) + 1e-300)):
            out = out + (sign ** j) * term
            break
    return out


# =============================================================================
# 1 組合せの指数傾斜 IH: 下側 G(τ) と上側反射 H(ξ)
# =============================================================================

def _tilt_lower_G(tau: np.ndarray, widths: list[float]) -> np.ndarray:
    """G(τ)=∫_0^τ e^{t} f_IH(t;{L}) dt。包除: Σ_T(−1)^|T| e^{σ_T} P_{m−1}(τ−σ_T)。"""
    m = len(widths)
    L = np.asarray(widths, dtype=float)
    prodL = float(np.prod(L))
    inv = 1.0 / (math.factorial(m - 1) * prodL)
    acc = np.zeros_like(tau, dtype=float)
    for r in range(m + 1):
        for subset in itertools.combinations(range(m), r):
            sign = -1.0 if (r & 1) else 1.0
            sigma = float(L[list(subset)].sum()) if subset else 0.0
            d = tau - sigma
            mask = d > 0.0
            if mask.any():
                acc[mask] += sign * math.exp(sigma) * _Pk_series(d[mask], m - 1, +1.0)
    return inv * acc


def _tilt_upper_H(xi: np.ndarray, widths: list[float]) -> np.ndarray:
    """H(ξ)=∫_0^ξ e^{−r} f_IH(r;{L}) dr。包除: Σ_T(−1)^|T| e^{−σ_T} Q_{m−1}(ξ−σ_T)。
    下端から積分するので上側裾でも桁落ちしない (Q は減衰指数の不完全ガンマ)。"""
    m = len(widths)
    L = np.asarray(widths, dtype=float)
    prodL = float(np.prod(L))
    inv = 1.0 / (math.factorial(m - 1) * prodL)
    acc = np.zeros_like(xi, dtype=float)
    for r in range(m + 1):
        for subset in itertools.combinations(range(m), r):
            sign = -1.0 if (r & 1) else 1.0
            sigma = float(L[list(subset)].sum()) if subset else 0.0
            d = xi - sigma
            mask = d > 0.0
            if mask.any():
                acc[mask] += sign * math.exp(-sigma) * _Pk_series(d[mask], m - 1, -1.0)
    return inv * acc


# =============================================================================
# S = Σ ln Y の厳密 CDF / 生存関数 / 密度 (全組合せの重み付き和)
# =============================================================================

def exact_S_cdf_sf_pdf(
    ymix_per_hit: list[list[Uniform]], s_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S=Σ ln Y の (CDF F_S, 生存 SF_S, 密度 f_S) を s_grid 上で返す。

    F_S は下端から (低ダメージ側で安定)、SF_S は上端反射 (高ダメージ側で安定) で
    別々に評価する。両側裾は呼び出し側で min(F_S, SF_S) 等を使うこと。"""
    s = np.asarray(s_grid, dtype=float)
    F = np.zeros_like(s)
    SF = np.zeros_like(s)
    f = np.zeros_like(s)

    comp_lists = [[c for c in mix if c.weight > 0.0] for mix in ymix_per_hit]
    for combo in itertools.product(*comp_lists):
        W = 1.0
        A = 0.0           # Σ ln p
        sumL = 0.0        # Σ L (= B − A)
        log_pdeg = 0.0    # Σ_deg ln p (退化成分)
        widths: list[float] = []
        denom_cont = 1.0  # Π_nd (q − p)
        for c in combo:
            W *= c.weight
            p, q = c.lo, c.hi
            lp = math.log(p)
            A += lp
            L = math.log(q) - lp if q > p else 0.0
            if L > _ATOM_TOL:
                widths.append(L)
                sumL += L
                denom_cont *= (q - p)
            else:
                log_pdeg += lp
        if W == 0.0:
            continue
        m = len(widths)
        if m == 0:
            # 全成分が点質量: S は A に集中する純原子
            F += W * (s >= A)
            SF += W * (s < A)
            continue
        prodL = float(np.prod(widths))
        const_c = prodL / denom_cont / math.exp(log_pdeg)
        B = A + sumL
        # F_S = const·e^{A}·G(s−A);  SF_S = const·e^{B}·H(B−s);  f_S = const·e^{s}·f_IH(s−A)
        #
        # 各組合せはサポート [A, B] の外では F=1 / SF=0 に **プラトー** する。包除は
        # サポート外で破綻的に相殺する (inv~1/(m!ΠL)~1e18 と P_k(τ≫sumL) の積が暴走)
        # ので、評価点を [0, sumL] にクランプしてから G/H を呼ぶ。クランプ上端で
        # G→E[e^{ Y}], H→E[e^{−Y}] となり const·e^{A}·E[e^Y]=1 で正しくプラトーする。
        tau = np.minimum(s - A, sumL)         # 上端越えは sumL で頭打ち (G→1 相当)
        xi = np.minimum(B - s, sumL)          # 下端割れは sumL で頭打ち (H→1 相当)
        F += W * const_c * math.exp(A) * _tilt_lower_G(tau, widths)
        SF += W * const_c * math.exp(B) * _tilt_upper_H(xi, widths)
        # 密度 f_IH(s−A) は対称 f_IH(t)=f_IH(ΣL−t) なので、各組合せの上端付近では
        # 反射した小引数 (B−s) で評価する。これをしないと _truncated_power_pdf の
        # 包除が上端で破滅的相殺 (inv~1e18) を起こし密度が裾でノイズ化する。
        arg_small = np.minimum(s - A, B - s)  # 対称軸の近い側 (どちらも同値・安定側)
        f += W * const_c * np.exp(s) * _truncated_power_pdf(arg_small, widths)
    return F, SF, f


# =============================================================================
# S → D 写像 (product_cos.damage_dist と同じ変換) で D の厳密分布へ
# =============================================================================

def exact_damage_pdf_cdf_sf(
    ymix_per_hit: list[list[Uniform]], Htil: float, d_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D = H̃_1(1 − e^{S}) の厳密 (密度 f_D, CDF F_D, 生存 SF_D) を d_grid 上で返す。

    S(D)=ln(1 − D/H̃_1)。β>0 (H̃_1>0) なら D は S の減少関数、β<0 なら増加関数。
    高ダメージ側の裾 (SF_D) は S 空間の安定側 (反射) を用いるので桁落ちしない。"""
    from experiments.product_cos import support_bounds_hits

    a, b = support_bounds_hits(ymix_per_hit)
    d = np.asarray(d_grid, dtype=float)
    arg = 1.0 - d / Htil                      # = e^{S(D)} (台の内側で正)
    pos = (d >= 0.0) & (arg > 0.0)
    s_of_d = np.full_like(d, np.nan)
    s_of_d[pos] = np.log1p(-d[pos] / Htil)
    decreasing = Htil > 0.0                   # D が S の減少関数か (β>0)

    f_D = np.zeros_like(d)
    F_D = np.zeros_like(d)
    SF_D = np.zeros_like(d)
    in_range = pos & (s_of_d >= a) & (s_of_d <= b)
    if in_range.any():
        sv = s_of_d[in_range]
        F_S, SF_S, f_S = exact_S_cdf_sf_pdf(ymix_per_hit, sv)
        if decreasing:                        # F_D = P(S≥S(D)) = SF_S
            F_D[in_range] = SF_S
            SF_D[in_range] = F_S
        else:                                 # β<0: F_D = P(S≤S(D)) = F_S
            F_D[in_range] = F_S
            SF_D[in_range] = SF_S
        f_D[in_range] = f_S / np.abs(Htil - d[in_range])
    # 台の外側
    if decreasing:                            # β>0: S(D)<a (D 大) または D≥H̃_1
        hi_tail = pos & (s_of_d < a)
        F_D[hi_tail | ~pos] = 1.0
    else:                                     # β<0: S(D)>b (D 大)
        hi_tail = pos & (s_of_d > b)
        F_D[hi_tail] = 1.0
        SF_D[pos & (s_of_d < a)] = 1.0
    return f_D, F_D, SF_D


# =============================================================================
# ゲブラカリン等プリセットから ymix_per_hit を組む補助
# =============================================================================

def build_ymix_for_preset(preset: dict) -> tuple[list[list[Uniform]], float, float]:
    """product_cos_app のプリセット dict から (ymix_per_hit, Htil, beta) を作る。
    アプリ run() と同じ手順 (hp0→基礎x→Y=1−βx)。"""
    from app.backend.simulation import inverse_decay
    from experiments.edgeworth_animation import build_all_hits
    from experiments.product_cos import y_mixture

    H, H1, R0, R1 = preset["H"], preset["H1"], preset["R0"], preset["R1"]
    beta = (R1 - R0) / H
    Htil = H1 + R0 / beta

    cards = preset["cards"]
    if preset.get("hp0_input"):
        # hp0 入力: 減衰を戻し /R0 で基礎 x へ (アプリ _hp0_to_base_cards 相当)
        conv = []
        for c in cards:
            nc = dict(c)
            for fld in ("crit_min", "crit_max", "normal_min", "normal_max"):
                v = c.get(fld)
                if v is not None:
                    nc[fld] = int(round(inverse_decay(float(v)) / R0))
            conv.append(nc)
        cards = conv
        build_mode = "pre_decay"
    else:
        build_mode = preset.get("damage_mode", "post_decay")

    hit_mixtures, _ = build_all_hits(
        cards, float(preset["global_crit"] or 0),
        float(preset["global_evade"] or 0), build_mode,
    )
    ymix_per_hit = [
        y_mixture([Uniform(c.weight, *((c.lo, c.hi) if c.hi >= c.lo else (c.hi, c.lo)))
                   for c in mix if c.weight > 0], beta)
        for mix in hit_mixtures
    ]
    return ymix_per_hit, Htil, beta


if __name__ == "__main__":
    # 自己検証: ゲブラカリンで COS・MC と一致するか
    import numpy as np
    from experiments.product_cos import (
        damage_dist, mc_damage_hits, support_bounds_hits, moments_hits,
    )
    from experiments.product_cos_app import PRESETS, _GEBURA_KARIN_CARDS  # noqa: F401

    preset = PRESETS["ゲブラカリン (R1=1, R0=3)"]
    ymix, Htil, beta = build_ymix_for_preset(preset)
    n_hits = len(ymix)
    n_combo = 1
    for mix in ymix:
        n_combo *= len(mix)
    a, b = support_bounds_hits(ymix)
    mean_S, var_S = moments_hits(ymix)
    print(f"ゲブラカリン: Hit数={n_hits}, 組合せ={n_combo}, β={beta:.3e}, H̃1={Htil:,.0f}")
    print(f"  S 台=[{a:.5f}, {b:.5f}], 平均S={mean_S:.5f}, σS={math.sqrt(var_S):.5f}")

    # S 空間で正規化チェック: F_S(b)=1, SF_S(a)=1, ∫f_S≈1
    Fb, _, _ = exact_S_cdf_sf_pdf(ymix, np.array([b]))
    _, SFa, _ = exact_S_cdf_sf_pdf(ymix, np.array([a]))
    print(f"  F_S(b)={Fb[0]:.10f} (→1), SF_S(a)={SFa[0]:.10f} (→1)")

    # D 空間で COS・MC と比較
    rng = np.random.default_rng(20260602)
    base_per_hit = None
    # MC は基礎 x が必要。プリセット手順を再現
    samples = None
    H, H1, R0, R1 = preset["H"], preset["H1"], preset["R0"], preset["R1"]
    from app.backend.simulation import inverse_decay
    from experiments.edgeworth_animation import build_all_hits
    cards = [dict(c) for c in preset["cards"]]
    for c in cards:
        for fld in ("crit_min", "crit_max", "normal_min", "normal_max"):
            if c.get(fld) is not None:
                c[fld] = int(round(inverse_decay(float(c[fld])) / R0))
    hm, _ = build_all_hits(cards, preset["global_crit"], preset["global_evade"], "pre_decay")
    base_per_hit = [[u for u in mix if u.weight > 0] for mix in hm]
    samples = mc_damage_hits(base_per_hit, H, H1, R0, R1, 5_000_000, rng)

    d_lo, d_hi = float(samples.min()), float(samples.max())
    xs = np.linspace(d_lo, d_hi, 21)
    _, F_cos = damage_dist(ymix, Htil, xs)
    f_ex, F_ex, SF_ex = exact_damage_pdf_cdf_sf(ymix, Htil, xs)
    samples.sort()
    F_mc = np.searchsorted(samples, xs, side="right") / samples.size

    print("\n  D             F_exact     F_COS       |ex−COS|    F_MC")
    for x, fe, fc, fm in zip(xs, F_ex, F_cos, F_mc):
        print(f"  {x:12,.0f}  {fe:.6f}   {fc:.6f}   {abs(fe-fc):.2e}   {fm:.6f}")
    print(f"\nmax|F_exact − F_COS| = {np.max(np.abs(F_ex - F_cos)):.3e}")
