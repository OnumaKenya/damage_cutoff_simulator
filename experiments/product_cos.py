"""積で表される分布の COS 法近似 (HP依存ダメージ; Qiita abira 記事の手法)。

docs/product.md の理論に基づく。これまで (docs/edge.md, docs/saddlepoint.md) は
独立な確率変数の「和」の分布を扱ってきたが、HP依存ダメージ (ミカ等) では累積ダメージが
独立な確率変数の「積」

    P = Π_{n=1}^N Y_n,     Y_n = 1 - β x_n,   β = ΔR / H

の単調関数

    D = H̃_1 (1 - P),      H̃_1 = H_1 + (R_0 / ΔR) H

として現れる (x_n は 1Hit の基礎ダメージ = 一様分布の混合、ΔR = R_1 - R_0)。
対数を取れば積は和に戻る:

    S = ln P = Σ_{n=1}^N ln Y_n.

ln Y_n の特性関数は閉形式で書けるので、和 S の特性関数は積 φ_S = Π_n φ_{ln Y_n} と
なり、COS 法 (Fang & Oosterlee 2008) でフーリエ余弦級数反転すれば S の分布が準厳密に
得られる。あとは D = H̃_1 (1 - e^S) でヤコビアン込みに D へ写すだけ。Qiita の記事は
同じ和 (記事の Λ = -S/β) を FFT で畳み込むが、ここでは特性関数を直接反転する。

MC (真値) は HP依存の漸化式 H_{n+1} = H_n - (β H_n + R_0) x_n を直接回した
累積ダメージ D = H_1 - H_{N+1} で、解析式 D = H̃_1(1 - Π(1-β x_n)) と一致する。

実行例:
    uv run python -m experiments.product_cos
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import 副作用で japanize / CJK フォント設定が済む。Uniform も再利用する。
from experiments.edgeworth_animation import Uniform  # noqa: E402

OUT_DIR = "experiments/output"

# COS 法の項数設定。S = ln P はコンパクト台 [A, B] を持ち (有界な積の対数)、
# 端点では密度が多項式的に 0 に落ちる (N≥2 の和) ため、台をそのまま [a,b] に
# 使えば Gibbs はほぼ出ない。項数は台幅 (σ 単位) に比例させる。
_COS_TERMS_PER_SIGMA = 256
_COS_N_MIN = 1024
_COS_N_MAX = 1 << 16

# 原子分離 (DP) の設定。S = ln P の純原子部 (全 Hit が点質量成分を取る組合せ) を
# 厳密に列挙して連続部から差し引く。値は ln スケール (O(1)) なのでマージ許容差は
# 機械精度近く、列挙数の上限を超えたら諦めて通常 COS に fallback する。
_ATOM_MERGE_TOL = 1e-9
_ATOM_MAX = 5_000_000


# =============================================================================
# シナリオ: HP依存ダメージ (ミカ R_1=2, R_0=1)
# =============================================================================

@dataclass
class HPScenario:
    name: str
    H: float           # 敵の最大HP (倍率の基準)
    H1: float          # 攻撃開始時の敵HP
    R0: float          # HP=0 時の倍率
    R1: float          # HP満タン時の倍率
    n_hits: int
    base_mixture: list[Uniform]   # 1Hit の基礎ダメージ x の一様混合 (全Hit同分布)

    @property
    def dR(self) -> float:
        return self.R1 - self.R0

    @property
    def beta(self) -> float:
        return self.dR / self.H

    @property
    def Htil(self) -> float:
        # H̃_1 = H_1 + (R_0/ΔR) H = H_1 + R_0/β
        return self.H1 + self.R0 / self.beta


def base_mixture(
    normal_lo: float, normal_hi: float,
    crit_lo: float, crit_hi: float,
    crit_rate: float, evade_rate: float = 0.0,
) -> list[Uniform]:
    """1Hit 基礎ダメージ x の一様混合を作る (命中×会心/非会心 + miss 1点分布)。"""
    mix: list[Uniform] = []
    hit = 1.0 - evade_rate
    if hit * crit_rate > 0:
        mix.append(Uniform(hit * crit_rate, crit_lo, crit_hi))
    if hit * (1.0 - crit_rate) > 0:
        mix.append(Uniform(hit * (1.0 - crit_rate), normal_lo, normal_hi))
    if evade_rate > 0:
        mix.append(Uniform(evade_rate, 0.0, 0.0))   # miss: x=0 → Y=1 (点質量)
    total = sum(u.weight for u in mix)
    for u in mix:
        u.weight /= total
    return mix


# =============================================================================
# x の混合 → Y = 1 - β x の混合 (アフィン変換、向きが反転)
# =============================================================================

def y_mixture(base: list[Uniform], beta: float) -> list[Uniform]:
    """基礎ダメージ x の一様混合を Y = 1 - β x の一様混合へ写す。
    x ∈ U(a,b) → Y は端点 1-βa, 1-βb の間。β>0 なら下端 1-βb < 上端 1-βa、
    β<0 (R1<R0) なら Y=1+|β|x>1 で大小が反転するため min/max を取り直す。
    退化 (a==b) は点質量 Y = 1-βa。"""
    out: list[Uniform] = []
    for u in base:
        y1 = 1.0 - beta * u.lo
        y2 = 1.0 - beta * u.hi
        lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)
        if not (lo > 0.0):
            raise ValueError(f"β x ≥ 1 となり Y≤0 (x_hi={u.hi}, β={beta}). HPが負になる設定。")
        out.append(Uniform(u.weight, lo, hi))
    return out


# =============================================================================
# ln Y の特性関数 (対数一様 = 切断指数分布)
#   Y ∈ U(p,q) (0<p≤q) のとき s = ln Y は密度 f(s) = e^s/(q-p) on [ln p, ln q]。
#   φ(u) = E[e^{ius}] = (q e^{iu ln q} - p e^{iu ln p}) / ((q-p)(1+iu))。
#   点質量 (p==q=c) なら φ(u) = e^{iu ln c}。u=0 で総質量 Σw_j = 1 を返す。
# =============================================================================

def logmix_cf(ymix: list[Uniform], u: np.ndarray) -> np.ndarray:
    """1Hit の ln Y の特性関数 φ(u) = Σ_j w_j φ_j(u)。"""
    phi = np.zeros_like(u, dtype=complex)
    for comp in ymix:
        w, p, q = comp.weight, comp.lo, comp.hi
        if q <= p:                                   # 点質量 Y=c
            phi += w * np.exp(1j * u * math.log(p))
        else:
            lp, lq = math.log(p), math.log(q)
            num = q * np.exp(1j * u * lq) - p * np.exp(1j * u * lp)
            phi += w * num / ((q - p) * (1.0 + 1j * u))
    return phi


def sum_cf(ymix: list[Uniform], n_hits: int, u: np.ndarray) -> np.ndarray:
    """独立同分布 n_hits 個の和 S = Σ ln Y_n の特性関数 φ_S = φ_{lnY}(u)^n。"""
    return logmix_cf(ymix, u) ** n_hits


def cf_S_hits(ymix_per_hit: list[list[Uniform]], u: np.ndarray) -> np.ndarray:
    """Hit ごとに異なる Y 混合を許す一般版: φ_S = Π_n φ_{lnY_n}(u)。"""
    phi = np.ones_like(u, dtype=complex)
    for ymix in ymix_per_hit:
        phi = phi * logmix_cf(ymix, u)
    return phi


# =============================================================================
# 原子分離 (DP): S = ln P の純原子部を厳密に列挙し、連続部のみ COS で反転する
#   各 Hit の ln Y は点質量成分 (Y=c の退化, miss なら c=1 で s=0) を持ちうる。
#   全 Hit が点質量成分を取る組合せだけが S の「純原子」を生む (1つでも連続成分を
#   含む項は絶対連続)。純原子部 CF φ_S^atom は u→∞ で減衰しないため、連続部
#   φ_cont = φ_S − φ_S^atom を COS で反転し、原子部は厳密な階段 CDF として加える。
#   cos_compare.py の DP を ln スケール (積→和) に移植したもの。
# =============================================================================

def _atom_points(ymix: list[Uniform]) -> list[tuple[float, float]] | None:
    """1Hit の ln Y の点質量成分 [(weight, s=ln c), ...]。点質量が無ければ None。"""
    pts = [(c.weight, math.log(c.lo)) for c in ymix if c.hi <= c.lo]
    return pts or None


def atom_cf_hits(ymix_per_hit: list[list[Uniform]], u: np.ndarray) -> np.ndarray:
    """純原子部の特性関数 φ_S^atom(u) = Π_n (Σ_pts w e^{ius})。どれか1 Hit でも
    点質量成分を持たなければ純原子部は無く 0 を返す。"""
    phi = np.ones_like(u, dtype=complex)
    for ymix in ymix_per_hit:
        pts = _atom_points(ymix)
        if pts is None:
            return np.zeros_like(u, dtype=complex)
        g = np.zeros_like(u, dtype=complex)
        for w, s in pts:
            g = g + w * np.exp(1j * u * s)
        phi = phi * g
    return phi


def atom_part_distribution(
    ymix_per_hit: list[list[Uniform]],
) -> tuple[np.ndarray, np.ndarray]:
    """S = Σ ln Y の純原子部の (値 s, 確率) を逐次畳み込みで返す。純原子部が無い、
    または列挙数が _ATOM_MAX を超える (準連続) なら空配列を返す (→ 通常 COS)。"""
    per_hit: list[list[tuple[float, float]]] = []
    for ymix in ymix_per_hit:
        pts = _atom_points(ymix)
        if pts is None:
            return np.empty(0), np.empty(0)
        per_hit.append(pts)

    vals = np.array([0.0])
    probs = np.array([1.0])
    for pts in per_hit:
        pw = np.array([w for w, _ in pts])
        ps = np.array([s for _, s in pts])
        if vals.size * ps.size > _ATOM_MAX:
            return np.empty(0), np.empty(0)
        v = (vals[:, None] + ps[None, :]).ravel()
        p = (probs[:, None] * pw[None, :]).ravel()
        order = np.argsort(v)
        v, p = v[order], p[order]
        keep = np.concatenate([[True], np.diff(v) > _ATOM_MERGE_TOL])
        grp = np.cumsum(keep) - 1
        vals = v[keep]
        probs = np.zeros(vals.size)
        np.add.at(probs, grp, p)
    return vals, probs


def _atom_step_cdf(av: np.ndarray, ap: np.ndarray, s: np.ndarray) -> np.ndarray:
    """純原子部の厳密な階段 CDF Σ_{a_k ≤ s} p_k を s 上で返す。"""
    order = np.argsort(av, kind="mergesort")
    cum = np.concatenate([[0.0], np.cumsum(ap[order])])
    idx = np.searchsorted(av[order], np.asarray(s, dtype=float), side="right")
    return cum[idx]


# =============================================================================
# ln Y のモーメント (COS の台幅・項数決定 & 健全性チェック用)
# =============================================================================

def logmix_moments(ymix: list[Uniform]) -> tuple[float, float]:
    """1Hit の ln Y の (平均, 分散)。Y∈U(p,q) で
        E[lnY]   = (q ln q - p ln p)/(q-p) - 1
        E[ln²Y]  = [y((ln y)²-2 ln y+2)]_p^q / (q-p)。"""
    mean = ex2 = 0.0
    for comp in ymix:
        w, p, q = comp.weight, comp.lo, comp.hi
        if q <= p:
            m1 = math.log(p)
            e2 = m1 * m1
        else:
            lp, lq = math.log(p), math.log(q)
            m1 = (q * lq - p * lp) / (q - p) - 1.0
            anti = lambda y, ly: y * (ly * ly - 2.0 * ly + 2.0)
            e2 = (anti(q, lq) - anti(p, lp)) / (q - p)
        mean += w * m1
        ex2 += w * e2
    var = ex2 - mean * mean
    return mean, var


def support_bounds(ymix: list[Uniform], n_hits: int) -> tuple[float, float]:
    """S = Σ ln Y の厳密な台 [A, B]。各Hit の ln Y は [ln(min p), ln(max q)]。"""
    lo = min(math.log(c.lo) for c in ymix)   # lo は下端 p
    hi = max(math.log(c.hi) for c in ymix)   # hi は上端 q
    return n_hits * lo, n_hits * hi


def support_bounds_hits(ymix_per_hit: list[list[Uniform]]) -> tuple[float, float]:
    """Hit ごとに異なる Y 混合を許す一般版の台 [A, B]。"""
    lo = sum(min(math.log(c.lo) for c in ymix) for ymix in ymix_per_hit)
    hi = sum(max(math.log(c.hi) for c in ymix) for ymix in ymix_per_hit)
    return lo, hi


def moments_hits(ymix_per_hit: list[list[Uniform]]) -> tuple[float, float]:
    """Hit ごとに異なる Y 混合を許す一般版の S = Σ ln Y の (平均, 分散)。"""
    mean = var = 0.0
    for ymix in ymix_per_hit:
        m, v = logmix_moments(ymix)
        mean += m
        var += v
    return mean, var


def y_raw_moment(ymix: list[Uniform], k: int) -> float:
    """1Hit の Y = 1-βx の k 次積率 E[Y^k] = Σ_j w_j E[Y_j^k]。
    Y∈U(p,q) なら E[Y^k] = (q^{k+1}-p^{k+1})/((k+1)(q-p))、点質量 Y=c なら c^k。"""
    m = 0.0
    for c in ymix:
        w, p, q = c.weight, c.lo, c.hi
        if q <= p:
            m += w * p ** k
        else:
            m += w * (q ** (k + 1) - p ** (k + 1)) / ((k + 1) * (q - p))
    return m


def damage_moments(ymix_per_hit: list[list[Uniform]], Htil: float) -> tuple[float, float]:
    """累積ダメージ D = H̃_1(1 - P), P = Π Y_n の厳密な (平均, 分散)。
    独立性から E[P^k] = Π_n E[Y_n^k] なので
        E[D]   = H̃_1 (1 - Π E[Y_n]),
        Var[D] = H̃_1² (Π E[Y_n²] - (Π E[Y_n])²)。
    指数写像 D=H̃_1(1-e^S) を含んだ実効精度を MC 標本誤差ゼロで測る検証基準。"""
    EP = EP2 = 1.0
    for ymix in ymix_per_hit:
        EP *= y_raw_moment(ymix, 1)
        EP2 *= y_raw_moment(ymix, 2)
    mean = Htil * (1.0 - EP)
    var = Htil * Htil * (EP2 - EP * EP)
    return mean, var


# =============================================================================
# COS 法 (S = ln P の分布をフーリエ余弦級数で反転)
# =============================================================================

def cos_coeffs(ymix: list[Uniform], n_hits: int,
               a: float, b: float, n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """COS 係数 F_k = (2/(b-a)) Re[φ_S(u_k) e^{-i u_k a}] と角周波数 u_k を返す。"""
    L = b - a
    k = np.arange(n_terms)
    u = k * np.pi / L
    phi = sum_cf(ymix, n_hits, u)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))
    return u, Fk


def cos_coeffs_hits(ymix_per_hit: list[list[Uniform]],
                    a: float, b: float, n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Hit ごとに異なる Y 混合を許す一般版の COS 係数。"""
    L = b - a
    k = np.arange(n_terms)
    u = k * np.pi / L
    phi = cf_S_hits(ymix_per_hit, u)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))
    return u, Fk


def cos_cdf_S(u: np.ndarray, Fk: np.ndarray, a: float, s: np.ndarray) -> np.ndarray:
    """COS 係数から CDF F_S(s) = P(S ≤ s) を s 上で返す (項別積分)。"""
    s = np.asarray(s, dtype=float)
    ds = s - a
    cdf = 0.5 * Fk[0] * ds                         # k=0 項
    arg = np.outer(ds, u[1:])
    cdf += (Fk[1:][None, :] * np.sin(arg) / u[1:][None, :]).sum(axis=1)
    return np.clip(cdf, 0.0, 1.0)


def cos_pdf_S(u: np.ndarray, Fk: np.ndarray, a: float, s: np.ndarray) -> np.ndarray:
    """COS 係数から密度 f_S(s) を s 上で返す。"""
    s = np.asarray(s, dtype=float)
    Fk0 = Fk.copy()
    Fk0[0] *= 0.5                                  # Σ' は k=0 を半分に
    arg = np.outer(s - a, u)
    return (Fk0[None, :] * np.cos(arg)).sum(axis=1)


# =============================================================================
# S = ln P の分布 → D = H̃_1(1 - e^S) の分布 (単調変換 + ヤコビアン)
# =============================================================================

def damage_dist(
    ymix_per_hit: list[list[Uniform]],
    Htil: float,
    d_grid: np.ndarray,
    *,
    dp: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Hit ごとに異なる Y 混合 (= 基礎ダメージ x の混合をアフィン変換したもの) から
    COS 法で D = H̃_1(1 - e^S) の (密度 f_D, 分布関数 F_D) を d_grid 上で返す。

    D は S について単調「減少」(S↑ ⇔ P↑ ⇔ D↓) なので S(D)=ln(1-D/H̃_1) として
        F_D(D) = P(D'≤D) = P(S≥S(D)) = 1 - F_S(S(D)),
        f_D(D) = f_S(S(D)) · |dS/dD| = f_S(S(D)) / (H̃_1 - D)。

    dp=True (DPあり): 純原子部 (miss など全 Hit が点質量を取る組合せ) を厳密に
        分離する。連続部 CF φ_cont = φ_S − φ_S^atom (u→∞ で減衰) を COS 反転し、
        F_S に原子の階段 CDF を加える。離散・混合での Gibbs 振動を回避する。
        純原子部が無ければ通常 COS と一致する。
    dp=False: 全 CF をそのまま反転 (比較用)。"""
    a, b = support_bounds_hits(ymix_per_hit)

    mean_S, var_S = moments_hits(ymix_per_hit)
    std_S = math.sqrt(var_S) if var_S > 0 else (b - a)
    width_sigma = (b - a) / std_S if std_S > 0 else 1.0
    n_terms = int(math.ceil(_COS_TERMS_PER_SIGMA * width_sigma))
    n_terms = max(_COS_N_MIN, min(_COS_N_MAX, n_terms))

    L = b - a
    k = np.arange(n_terms)
    u = k * np.pi / L
    phi = cf_S_hits(ymix_per_hit, u)

    # 原子分離 (DP): 「引く」(atom_cf) と「足す」(階段 CDF) を必ず同条件でそろえる。
    av = ap = None
    if dp:
        av, ap = atom_part_distribution(ymix_per_hit)
        if av.size == 0:
            av = ap = None
        else:
            phi = phi - atom_cf_hits(ymix_per_hit, u)   # 連続部のみ
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))

    def cdf_S(s: np.ndarray) -> np.ndarray:
        cont = cos_cdf_S(u, Fk, a, s)
        if av is not None:
            cont = cont + _atom_step_cdf(av, ap, s)
        return np.clip(cont, 0.0, 1.0)

    if verbose:
        # 健全性チェック: F_S(B)=1, COS 平均 ≈ 解析平均 (原子部込み)
        F_at_b = float(cdf_S(np.array([b]))[0])
        s_dense = np.linspace(a, b, 4096)
        f_dense = cos_pdf_S(u, Fk, a, s_dense)
        mean_cont = float(np.trapezoid(s_dense * f_dense, s_dense))
        mean_atom = float((av * ap).sum()) if av is not None else 0.0
        atom_mass = float(ap.sum()) if av is not None else 0.0
        print(f"  COS{'+DP' if av is not None else ''}: 台S=[{a:.4f}, {b:.4f}], "
              f"項数={n_terms}, F_S(B)={F_at_b:.6f} (→1), "
              f"平均S COS={mean_cont + mean_atom:.5f} vs 解析={mean_S:.5f}"
              + (f", 原子質量={atom_mass:.4f}" if av is not None else ""))

        # 指数写像込みの実効精度を MC 非依存で測る。E[e^S]=E[P]=ΠE[Y_n] が厳密に
        # 出るので、COS 再構成の ∫e^s f_S ds (= 指数の肩を含む) を厳密値と比べる。
        mean_D_exact, var_D_exact = damage_moments(ymix_per_hit, Htil)
        eS_cont = float(np.trapezoid(np.exp(s_dense) * f_dense, s_dense))
        eS_atom = float((np.exp(av) * ap).sum()) if av is not None else 0.0
        mean_D_cos = Htil * (1.0 - (eS_cont + eS_atom))
        d_max = -Htil * math.expm1(a)                # D_max = H̃_1(1-e^A); 高ダメージ端
        rel = abs(mean_D_cos - mean_D_exact) / abs(mean_D_exact) if mean_D_exact else 0.0
        print(f"       E[D] COS={mean_D_cos:,.2f} vs 厳密={mean_D_exact:,.2f} "
              f"(相対誤差 {rel:.2e}), SD[D]厳密={math.sqrt(max(var_D_exact, 0.0)):,.2f}, "
              f"D_max=H̃_1(1-e^A)={d_max:,.2f}")

    # D → S 写像。D = H̃_1(1 - e^S) より S(D) = ln(1 - D/H̃_1)、e^{S} = 1 - D/H̃_1。
    #   β>0 (R1>R0): H̃_1>0、Y<1、S≤0。D は S の単調「減少」(S↑⇔D↓) で
    #       F_D(D) = P(S ≥ S(D)) = 1 - F_S(S(D))。
    #   β<0 (R1<R0): H̃_1<0、Y>1、S≥0。D は S の単調「増加」(S↑⇔D↑) で
    #       F_D(D) = P(S ≤ S(D)) = F_S(S(D))。
    #   いずれも密度は f_D = f_S(S(D))·|dS/dD| = f_S(S(D)) / |H̃_1 - D|。
    d = np.asarray(d_grid, dtype=float)
    arg = 1.0 - d / Htil                              # = e^{S(D)} (台の内側で正)
    s_of_d = np.full_like(d, np.nan)
    pos = (d >= 0.0) & (arg > 0.0)
    # S(D)=ln(1-D/H̃_1)。低ダメージ側 (D/H̃_1≪1) で arg≈1 となり log(arg) は桁落ち
    # するため log1p(-D/H̃_1) で直接評価し小値域の相対精度を保つ。
    s_of_d[pos] = np.log1p(-d[pos] / Htil)
    decreasing = Htil > 0.0                           # D が S の減少関数か (β>0)

    f_D = np.zeros_like(d)
    F_D = np.zeros_like(d)
    in_range = pos & (s_of_d >= a) & (s_of_d <= b)
    if in_range.any():
        cd = cdf_S(s_of_d[in_range])
        F_D[in_range] = (1.0 - cd) if decreasing else cd
        # 密度は連続部のみ (純原子部はデルタなので f_D には現れない)
        f_D[in_range] = cos_pdf_S(u, Fk, a, s_of_d[in_range]) / np.abs(Htil - d[in_range])
    # 台の外側で F=1 になる側 (最大ダメージ超):
    if decreasing:                                    # β>0: S(D)<a (D 大) または D≥H̃_1
        F_D[pos & (s_of_d < a)] = 1.0
        F_D[~pos] = 1.0
    else:                                             # β<0: S(D)>b (D 大)
        F_D[pos & (s_of_d > b)] = 1.0
    return f_D, F_D


def damage_pdf_cdf(sc: HPScenario, d_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同分布 n_hits の HPScenario 版。damage_dist に委譲する。"""
    ymix = y_mixture(sc.base_mixture, sc.beta)
    ymix_per_hit = [ymix] * sc.n_hits
    return damage_dist(ymix_per_hit, sc.Htil, d_grid, verbose=True)


# =============================================================================
# Monte Carlo (真値): HP依存の漸化式を直接回す
# =============================================================================

def mc_damage(sc: HPScenario, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """累積ダメージ D = H_1 - H_{N+1} を MC で返す。
    各Hit: 倍率 = β H_n + R_0、d = 倍率 × x、H_{n+1} = H_n - d (フロアなし=overkill許容)。"""
    Hn = np.full(n_samples, sc.H1, dtype=float)
    weights = np.array([u.weight for u in sc.base_mixture])
    weights = weights / weights.sum()
    for _ in range(sc.n_hits):
        comp = rng.choice(len(sc.base_mixture), size=n_samples, p=weights)
        x = np.empty(n_samples)
        for j, u in enumerate(sc.base_mixture):
            m = comp == j
            cnt = int(m.sum())
            if cnt == 0:
                continue
            x[m] = rng.uniform(u.lo, u.hi, size=cnt) if u.hi > u.lo else u.lo
        mult = sc.beta * Hn + sc.R0
        Hn = Hn - mult * x
    return sc.H1 - Hn


def mc_damage_hits(
    base_per_hit: list[list[Uniform]],
    H: float, H1: float, R0: float, R1: float,
    n_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """Hit ごとに異なる基礎ダメージ x 混合を許す MC。累積ダメージ D = H_1 - H_{N+1}。
    各Hit: 倍率 = β H_n + R_0、d = 倍率 × x、H_{n+1} = H_n - d (フロアなし=overkill許容)。"""
    beta = (R1 - R0) / H
    Hn = np.full(n_samples, H1, dtype=float)
    for base in base_per_hit:
        weights = np.array([c.weight for c in base], dtype=float)
        weights = weights / weights.sum()
        comp = rng.choice(len(base), size=n_samples, p=weights)
        x = np.empty(n_samples)
        for j, c in enumerate(base):
            m = comp == j
            cnt = int(m.sum())
            if cnt == 0:
                continue
            x[m] = rng.uniform(c.lo, c.hi, size=cnt) if c.hi > c.lo else c.lo
        mult = beta * Hn + R0
        Hn = Hn - mult * x
    return H1 - Hn


# =============================================================================
# 比較プロット
# =============================================================================

def make_plot(sc: HPScenario, output_path: str,
              n_mc: int = 2_000_000, n_grid: int = 1200) -> None:
    print(f"[{sc.name}] N={sc.n_hits} hits, H={sc.H:,.0f}, H1={sc.H1:,.0f}, "
          f"R0={sc.R0}, R1={sc.R1}, H̃1={sc.Htil:,.0f}")
    rng = np.random.default_rng(20260530)
    samples = mc_damage(sc, n_mc, rng)
    d_lo, d_hi = float(samples.min()), float(samples.max())
    pad = 0.04 * (d_hi - d_lo)
    d_grid = np.linspace(max(0.0, d_lo - pad), d_hi + pad, n_grid)

    f_D, F_D = damage_pdf_cdf(sc, d_grid)

    # 健全性: 平均ダメージ。厳密値 E[D]=H̃_1(1-ΠE[Y_n]) を MC 非依存の基準に置き、
    # MC (標本ノイズあり) と COS (指数写像込みの数値積分) の双方を突き合わせる。
    ymix = y_mixture(sc.base_mixture, sc.beta)
    mean_exact, var_exact = damage_moments([ymix] * sc.n_hits, sc.Htil)
    std_exact = float(np.sqrt(var_exact))
    mean_mc = float(samples.mean())
    mean_cos = float(np.trapezoid(d_grid * f_D, d_grid))
    print(f"  平均D: 厳密={mean_exact:,.1f}  MC={mean_mc:,.1f} "
          f"(相対 {abs(mean_mc-mean_exact)/mean_exact*100:.3f}%)  "
          f"COS={mean_cos:,.1f} (相対 {abs(mean_cos-mean_exact)/mean_exact*100:.3f}%)")

    fig, (ax_pdf, ax_tail) = plt.subplots(1, 2, figsize=(13, 5))

    # --- 密度 (期待値・標準偏差で標準化: z = (D − E[D]) / σ) ---
    # 標準化により密度は f_Z(z) = f_D(D) · σ。MC 標本も z に変換してから集計する。
    n_bins = 160
    z_samples = (samples - mean_exact) / std_exact
    counts, edges = np.histogram(z_samples, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_pdf.bar(centers, counts, width=edges[1] - edges[0], color="0.8",
               edgecolor="none", label="MC 経験密度 (真値)")
    z_grid = (d_grid - mean_exact) / std_exact
    ax_pdf.plot(z_grid, f_D * std_exact, color="magenta", lw=2.0, label="COS 法")
    ax_pdf.set_title("標準化ダメージ密度 f(z)")
    ax_pdf.set_xlabel("標準化ダメージ z = (D − E[D]) / σ")
    ax_pdf.set_ylabel("密度")
    ax_pdf.legend()
    ax_pdf.grid(alpha=0.3)

    # --- 両側裾確率 min(P(D<=x), P(D>x)) (片対数) ---
    # 中央値を境に左側は下側裾 P(D≤x)、右側は上側裾 P(D>x) を表す折れ線になる。
    # x 軸は期待値・標準偏差で標準化した z = (D − E[D]) / σ (裾確率自体は不変)。
    xs_tail = np.linspace(d_grid[0], d_grid[-1], 600)
    zs_tail = (xs_tail - mean_exact) / std_exact
    sorted_s = np.sort(samples)
    mc_cdf = np.searchsorted(sorted_s, xs_tail, side="right") / n_mc
    mc_tail = np.minimum(mc_cdf, 1.0 - mc_cdf)
    _, F_tail = damage_pdf_cdf(sc, xs_tail)
    cos_tail = np.minimum(F_tail, 1.0 - F_tail)
    ax_tail.semilogy(zs_tail, np.where(mc_tail > 0, mc_tail, np.nan),
                     color="0.4", lw=2.5, label="MC 裾確率 (真値)")
    ax_tail.semilogy(zs_tail, np.where(cos_tail > 0, cos_tail, np.nan),
                     color="magenta", lw=1.6, ls="--", label="COS 法")
    ax_tail.set_title("両側裾確率 min(P(D<=x), P(D>x))")
    ax_tail.set_xlabel("標準化ダメージ z = (D − E[D]) / σ")
    ax_tail.set_ylabel("min(P(D<=x), P(D>x))")
    ax_tail.set_ylim(1e-6, 1.0)
    ax_tail.legend()
    ax_tail.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"積で表される分布の COS 法 — {sc.name}\n"
        f"D = H~1(1 − Π(1 − β x)),  S = ln P = Σ ln Y を COS 反転",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {output_path}")


def main() -> None:
    # ミカ: R_1=2, R_0=1。満タン(H1=H)から殴り始める想定。
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
    make_plot(sc, os.path.join(OUT_DIR, "product_cos_mika.png"))


if __name__ == "__main__":
    main()
