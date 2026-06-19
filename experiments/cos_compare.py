"""COS法 と Edgeworth 展開の「裾確率」比較実験。

docs/edge.md の理論に基づき、1Hit のダメージを一様分布の混合とみなす点は
edgeworth_animation.py と共通。本スクリプトでは閾値 x を掃引し、合計ダメージ
S_n の裾確率 P(S_n > x) / P(S_n <= x) を以下の手法で比較する:

    - モンテカルロ 経験裾確率 (多数サンプルの点。誤差・密度比較の基準)
    - Edgeworth 展開 2次  (既存 edgeworth_pdf_cdf の CDF から 1-F)
    - COS法 (特性関数の数値反転; Fang & Oosterlee 2008) — 準厳密な基準線
      DPあり (原子分離)。DPなし (素の全CF) は show_cos_nodp で任意に併記

Edgeworth は中心付近で高精度だが裾で劣化する (密度が負・CDF が [0,1] を逸脱)。
COS法は漸近近似ではなく正確な特性関数を反転するため全域で準厳密で、MC と並ぶ
基準線になる。離散・混合では DPなし (素の全CF) が原子の周りで Gibbs 振動を出し、
DPあり (原子分離) はこれを回避する — show_cos_nodp=True のときその差も可視化する。

実行例:
    uv run python -m experiments.cos_compare
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
# edgeworth_animation の import で japanize_matplotlib / フォント設定も済む
from experiments.edgeworth_animation import (  # noqa: E402
    Uniform,
    build_all_hits,
    edgeworth_pdf_cdf,
    hit_cumulants,
    simulate_per_hit_samples,
    _build_cards_and_breakpoints,
)

# COS 法 (特性関数の数値反転) の設定
#   積分区間はキュムラント基準窓 m_n ± _COS_L·√(s_n²+√|κ4|) をサポートで
#   クリップして取る (両者の狭い方)。サポートが狭ければサポート (有界分布の
#   真の端)、広大なら窓 (σ スケールの幅) になる。これにより n が大きく幅が
#   大きい場合でも区間幅が σ スケールに収まり、項数 N が n に依存しない。
#   捨てる裾質量は ~Φ(-_COS_L) で機械精度以下。項数 N はクリップ後の区間幅
#   (σ 単位) に比例させる。
_COS_L = 12.0
_COS_TERMS_PER_SIGMA = 256
_COS_N_MIN = 2048
_COS_N_MAX = 1 << 18


# =============================================================================
# Hit 混合のグルーピングとサポート (台) の評価
# =============================================================================

GroupedHits = list[tuple[int, list[Uniform]]]


def group_hits(hit_mixtures: list[list[Uniform]]) -> GroupedHits:
    """同一 (オブジェクト同一) の混合分布を (個数, 混合) に畳んで CGF 評価を速くする。"""
    grouped: GroupedHits = []
    for mix in hit_mixtures:
        if grouped and grouped[-1][1] is mix:
            cnt, m = grouped[-1]
            grouped[-1] = (cnt + 1, m)
        else:
            grouped.append((1, mix))
    return grouped


def support_bounds(grouped: GroupedHits) -> tuple[float, float]:
    """S_n の取りうる値の下限・上限 (各成分の min/max を Hit 数だけ加算)。"""
    lo = 0.0
    hi = 0.0
    for cnt, mix in grouped:
        comp_lo = min(c.center - c.half_width for c in mix)
        comp_hi = max(c.center + c.half_width for c in mix)
        lo += cnt * comp_lo
        hi += cnt * comp_hi
    return lo, hi


# =============================================================================
# 特性関数の数値反転 (COS 法, Fang & Oosterlee 2008)
#   Edgeworth が漸近近似なのに対し、こちらは「正確な特性関数」をフーリエ余弦
#   級数で反転する準厳密法。一様 U(a,b) の特性関数は e^{iuc}·sin(uh)/(uh):
#       φ_U(u) = E[e^{iuY}] = e^{iuc} · sin(uh)/(uh)
#   混合は各成分の和、Hit の和は積 φ_S(u) = Π_hit φ_hit(u)。密度が滑らか
#   (Hit 数が多いほど高次の連続性を持つ) ため級数は速やかに収束する。
# =============================================================================

def mixture_cf(mixture: list[Uniform], u: np.ndarray) -> np.ndarray:
    """1Hit の混合分布の特性関数 φ(u) = Σ_j w_j e^{i u c_j} sinc(u h_j)。"""
    phi = np.zeros_like(u, dtype=complex)
    for comp in mixture:
        w = comp.weight
        c = comp.center
        h = comp.half_width
        if h == 0.0:
            phi += w * np.exp(1j * u * c)
        else:
            # np.sinc(x) = sin(πx)/(πx) なので sin(uh)/(uh) = sinc(uh/π)
            phi += w * np.exp(1j * u * c) * np.sinc((u * h) / np.pi)
    return phi


def sum_cf(grouped: GroupedHits, u: np.ndarray) -> np.ndarray:
    """Hit 和 S_n の特性関数 φ_S(u) = Π_hit φ_hit(u)。独立和なので積。"""
    phi = np.ones_like(u, dtype=complex)
    for cnt, mix in grouped:
        phi *= mixture_cf(mix, u) ** cnt
    return phi


# =============================================================================
# 原子分離ハイブリッド: 離散(原子)部を厳密に、連続部のみ COS で扱い Gibbs を回避
#   和 S の「純粋な原子部」は全 Hit が点質量成分 (h=0) を取るときだけ生じる
#   (質量 P_atom = Π_i α_i, α_i = Hit i の点質量成分の重み和)。連続成分を1つでも
#   含む項はすべて絶対連続。よって連続部 CF
#       φ_S^cont(u) = φ_S(u) − φ_S^atom(u),   φ_S^atom(u) = Π_group (Σ_pts w e^{iuc})^cnt
#   は u→∞ で 0 に減衰し、COS で Gibbs なく反転できる。原子部は DP で厳密に
#   階段 CDF として加える。点質量成分が無ければ P_atom=0 で従来の COS に一致。
# =============================================================================

_ATOM_MERGE_TOL = 1e-6      # 同一原子値とみなす許容差
_ATOM_MAX = 5_000_000       # 原子数がこれを超えたら列挙を諦める (準連続とみなす)


def _atom_components(grouped: GroupedHits) -> list[tuple[int, list[tuple[float, float]]]] | None:
    """各 group の点質量成分 [(weight, center), ...] を返す。どれか1群でも点質量
    成分を持たなければ純原子部は存在しないので None。"""
    per_group: list[tuple[int, list[tuple[float, float]]]] = []
    for cnt, mix in grouped:
        pts = [(c.weight, c.center) for c in mix if c.half_width == 0.0]
        if not pts:
            return None
        per_group.append((cnt, pts))
    return per_group


def atom_cf(grouped: GroupedHits, u: np.ndarray) -> np.ndarray:
    """純原子部の特性関数 φ_S^atom(u) = Π_group (Σ_pts w e^{iuc})^cnt。
    純原子部が無ければ 0。これは u→∞ で減衰せず (離散性)、連続部から差し引く。"""
    per_group = _atom_components(grouped)
    if per_group is None:
        return np.zeros_like(u, dtype=complex)
    phi = np.ones_like(u, dtype=complex)
    for cnt, pts in per_group:
        g = np.zeros_like(u, dtype=complex)
        for w, c in pts:
            g += w * np.exp(1j * u * c)
        phi *= g ** cnt
    return phi


def _group_atom_dist(pts: list[tuple[float, float]], cnt: int) -> tuple[np.ndarray, np.ndarray]:
    """cnt 個の同一原子分布 pts=[(w,c),...] の和の (値, 確率)。同一値はマージする
    (群内は同一 Hit なので値が格子上で衝突し、サイズは cnt·(J−1)+1 に収まる)。"""
    vals = np.array([0.0])
    probs = np.array([1.0])
    pw = np.array([w for w, _ in pts])
    pc = np.array([c for _, c in pts])
    for _ in range(cnt):
        v = (vals[:, None] + pc[None, :]).ravel()
        p = (probs[:, None] * pw[None, :]).ravel()
        order = np.argsort(v)
        v = v[order]
        p = p[order]
        keep = np.concatenate([[True], np.diff(v) > _ATOM_MERGE_TOL])
        grp = np.cumsum(keep) - 1
        vals = v[keep]
        probs = np.zeros(vals.size)
        np.add.at(probs, grp, p)
    return vals, probs


def _float_gcd(values: list[float], tol: float) -> float | None:
    """値集合の (近似) 最大公約数を Euclid 互除で求める。スプレッドが無い (空) なら
    None。通約不能な実数では g が tol 付近まで縮み、後段の格子検証で弾かれる。"""
    g = 0.0
    for v in values:
        a, b = max(g, abs(v)), min(g, abs(v))
        while b > tol:
            a, b = b, a - math.floor(a / b) * b
        g = a
    return g if g > tol else None


def _atom_part_fft(
    per_group: list[tuple[int, list[tuple[float, float]]]],
    merge_tol: float,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """純原子部 (値, 確率) を FFT で求める。分布は Π_group (Σ_pts w·x^c)^cnt の係数。

    格子設定: 刻み δ は「群内の中心差」の gcd だけで決める。各群の最小中心 c_min,g
    は定数なのでオフセット O = Σ cnt·c_min,g にまとめ、FFT は相対指数 (c−c_min,g)/δ
    のみを解像する。よってダメージの絶対値が大きくても (O が大きくなるだけで) FFT
    サイズは相対スパン D = Σ cnt·d_g で決まる。非格子 (通約不能) または D+1 が
    max_size 超なら None を返す (→ 呼び出し側で疎 DP / COS に fallback)。"""
    diffs: list[float] = []
    for _cnt, pts in per_group:
        cs = sorted(c for _w, c in pts)
        diffs += [cs[i] - cs[i - 1] for i in range(1, len(cs))
                  if cs[i] - cs[i - 1] > merge_tol]

    # 全群が単一点質量 → 原子は 1 値 (オフセットに集約)
    if not diffs:
        val = sum(cnt * pts[0][1] for cnt, pts in per_group)
        mass = 1.0
        for cnt, pts in per_group:
            mass *= sum(w for w, _c in pts) ** cnt
        return np.array([val]), np.array([mass])

    maxabs = max(abs(c) for _cnt, pts in per_group for _w, c in pts)
    gtol = merge_tol + 1e-9 * maxabs           # 大ダメージでの丸めに比例した許容差
    delta = _float_gcd(diffs, gtol)
    if delta is None:
        return None

    groups: list[tuple[int, np.ndarray]] = []
    offset = 0.0
    span = 0                                    # 相対スパン D (格子単位)
    for cnt, pts in per_group:
        cmin = min(c for _w, c in pts)
        emax = 0
        contrib: list[tuple[int, float]] = []
        for w, c in pts:
            e = round((c - cmin) / delta)
            if abs((c - cmin) - e * delta) > gtol:
                return None                    # 非格子 → fallback
            contrib.append((e, w))
            emax = max(emax, e)
        poly = np.zeros(emax + 1)
        for e, w in contrib:
            poly[e] += w                       # 同一指数の重みを併合
        groups.append((cnt, poly))
        offset += cnt * cmin
        span += cnt * emax

    if span + 1 > max_size:
        return None                            # 大きすぎ (準連続) → fallback

    size = 1
    while size < span + 1:
        size <<= 1                             # 周期畳み込みの巻き込み回避
    acc = np.ones(size // 2 + 1, dtype=complex)
    for cnt, poly in groups:
        acc *= np.fft.rfft(poly, n=size) ** cnt
    probs = np.clip(np.fft.irfft(acc, n=size)[: span + 1], 0.0, None)
    vals = offset + delta * np.arange(span + 1)
    mask = probs > probs.max() * 1e-14         # 丸め雑音・構造ゼロを除去
    return vals[mask], probs[mask]


def atom_part_distribution(grouped: GroupedHits) -> tuple[np.ndarray, np.ndarray]:
    """純原子部の (値, 確率)。総質量 = Π_i α_i。純原子部が無い、または FFT・疎 DP
    のいずれでも _ATOM_MAX を超える (準連続) 場合は空配列 (→ 呼び出し側は通常 COS)。

    中心が共通格子に乗る通常ケースは FFT (_atom_part_fft) で O(D log D)。通約不能な
    実数中心など格子化できない場合のみ、従来の疎な逐次畳み込み DP に fallback する。"""
    per_group = _atom_components(grouped)
    if per_group is None:
        return np.empty(0), np.empty(0)

    fft_res = _atom_part_fft(per_group, _ATOM_MERGE_TOL, _ATOM_MAX)
    if fft_res is not None:
        return fft_res

    vals = np.array([0.0])                      # fallback: 疎な逐次畳み込み
    probs = np.array([1.0])
    for cnt, pts in per_group:
        gv, gp = _group_atom_dist(pts, cnt)
        if vals.size * gv.size > _ATOM_MAX:
            return np.empty(0), np.empty(0)
        vals = (vals[:, None] + gv[None, :]).ravel()
        probs = (probs[:, None] * gp[None, :]).ravel()
    return vals, probs


def _atom_step_cdf(av: np.ndarray, ap: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """純原子部の厳密な階段 CDF Σ_{a_k ≤ x} p_k を xs 上で返す (av, ap は事前計算)。"""
    order = np.argsort(av, kind="mergesort")
    cum = np.concatenate([[0.0], np.cumsum(ap[order])])
    idx = np.searchsorted(av[order], np.asarray(xs, dtype=float), side="right")
    return cum[idx]


def cos_cdf(
    grouped: GroupedHits, xs: np.ndarray, a: float, b: float, n_terms: int,
    hybrid: bool = True,
) -> np.ndarray:
    """COS 法による CDF F(x) = P(S_n ≤ x) を xs 上で返す。

    hybrid=True (DPあり): 原子分離。連続部 CF φ_cont = φ_S − φ_S^atom (u→∞ で減衰)
        をフーリエ余弦級数 F_k = (2/(b−a)) Re[φ_cont(u_k) e^{−i u_k a}] で展開・
        項別積分し、純原子部の厳密な階段 CDF を加える (Gibbs なし)。
    hybrid=False (DPなし): 全 CF φ_S をそのまま反転 (離散だと原子で Gibbs が出る。
        比較用)。原子が無ければ両者は一致する。
    """
    L = b - a
    k = np.arange(n_terms)
    u = k * np.pi / L
    # 原子分離は「列挙できた」ときだけ行う。列挙過大 (ガード作動) なら従来 COS に
    # フォールバックする。atom_cf を引くのに階段を足せないと質量が失われるため、
    # 「引く」と「足す」を必ず同じ条件 (av is not None) でそろえる。
    av = ap = None
    if hybrid:
        av, ap = atom_part_distribution(grouped)
        if av.size == 0:
            av = ap = None
    phi = sum_cf(grouped, u)
    if av is not None:
        phi = phi - atom_cf(grouped, u)          # 連続部のみ (純原子部を除く)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))

    xs = np.asarray(xs, dtype=float)
    dx = xs - a
    cdf = 0.5 * Fk[0] * dx                       # k = 0 項
    arg = np.outer(dx, u[1:])                    # (len(xs), n_terms−1)
    cdf += (Fk[1:][None, :] * np.sin(arg) / u[1:][None, :]).sum(axis=1)
    if av is not None:
        cdf += _atom_step_cdf(av, ap, xs)        # 純原子部 (厳密・Gibbs なし)
    return np.clip(cdf, 0.0, 1.0)


def cos_pdf(
    grouped: GroupedHits, xs: np.ndarray, a: float, b: float, n_terms: int,
    hybrid: bool = True,
) -> np.ndarray:
    """COS 法による密度を xs 上で返す。

    hybrid=True (DPあり): 連続部 CF φ_cont = φ_S − φ_S^atom (u→∞ で減衰) の余弦
        係数を用いるため Gibbs が出ない。純原子部はデルタなので密度には現れない
        (確率は atom_part_distribution で別途取得)。完全離散なら f_cont ≡ 0。
    hybrid=False (DPなし): 全 CF をそのまま用いる。離散だと原子の周りで Gibbs
        振動 (負の値) が出る。比較用。原子が無ければ両者は一致。"""
    L = b - a
    k = np.arange(n_terms)
    u = k * np.pi / L
    phi = sum_cf(grouped, u)
    if hybrid:
        av, _ap = atom_part_distribution(grouped)
        if av.size > 0:                          # 列挙できた原子部だけ分離 (一貫性)
            phi = phi - atom_cf(grouped, u)      # 連続部のみ
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))
    Fk = Fk.copy()
    Fk[0] *= 0.5                                 # Σ' は k=0 を半分に
    xs = np.asarray(xs, dtype=float)
    arg = np.outer(xs - a, u)                    # (len(xs), n_terms)
    return (Fk[None, :] * np.cos(arg)).sum(axis=1)


# =============================================================================
# シナリオ (Hit 混合のリスト) → 比較データ
# =============================================================================

class Scenario:
    def __init__(self, name: str, hit_mixtures: list[list[Uniform]]):
        self.name = name
        self.hit_mixtures = hit_mixtures
        self.grouped = group_hits(hit_mixtures)
        mean = var = k3 = k4 = k5 = k6 = 0.0
        for mix in hit_mixtures:
            m, v, a, b, e, f = hit_cumulants(mix)
            mean += m
            var += v
            k3 += a
            k4 += b
            k5 += e
            k6 += f
        self.mean = mean
        self.var = var
        self.std = math.sqrt(var)
        self.lam3 = k3 / self.std**3
        self.lam4 = k4 / self.std**4
        self.lam5 = k5 / self.std**5
        self.lam6 = k6 / self.std**6
        self.n_hits = len(hit_mixtures)
        lo, hi = support_bounds(self.grouped)
        self.support_lo = lo
        self.support_hi = hi
        # COS 法の積分区間: キュムラント基準窓をサポートでクリップ (狭い方)。
        # サポートが狭ければサポート (有界分布の真の端)、広大なら窓 (n 非依存の
        # σ スケール幅)。捨てる裾質量は ~Φ(-_COS_L) で機械精度以下。項数 N は
        # クリップ後の区間幅 (σ 単位) に比例させ、無駄な解像度を避ける。
        c4 = abs(self.lam4) * self.std**4
        half = _COS_L * math.sqrt(self.var + math.sqrt(c4)) if self.std > 0 else 0.0
        self.cos_a = max(self.support_lo, self.mean - half)
        self.cos_b = min(self.support_hi, self.mean + half)
        width_sigma = (self.cos_b - self.cos_a) / self.std if self.std > 0 else 1.0
        n_cos = int(math.ceil(_COS_TERMS_PER_SIGMA * width_sigma))
        self.cos_n = max(_COS_N_MIN, min(_COS_N_MAX, n_cos))


def mc_survival_samples(
    hit_mixtures: list[list[Uniform]], n_samples: int, seed: int,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """合計ダメージ S_n の MC サンプル (長さ n_samples) を返す。Hit 配列は保持しない。"""
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples, dtype=float)
    processed = 0
    while processed < n_samples:
        chunk = min(chunk_size, n_samples - processed)
        samples = simulate_per_hit_samples(hit_mixtures, chunk, rng)  # (n_hits, chunk)
        out[processed:processed + chunk] = samples.sum(axis=0)
        del samples
        processed += chunk
    return out


# =============================================================================
# プロット
# =============================================================================

# Edgeworth 系の全手法 (order, 色, ラベル)。表示するものは _SHOWN_ORDERS で選ぶ。
_ALL_METHODS = [
    ("正規分布近似 (CLT)", "tab:blue", 0),
    ("Edgeworth 1次", "tab:orange", 1),
    ("Edgeworth 2次", "tab:red", 2),
    ("Edgeworth 3次", "tab:olive", 3),
    ("Edgeworth 4次", "tab:brown", 4),
]
# 既定では Edgeworth 2次のみを表示し、MC・COS法 (DPあり/なし) と比較する。
# 他の次数を加えたいときは _SHOWN_ORDERS に番号を入れる。
_SHOWN_ORDERS = {2}
_METHODS = [m for m in _ALL_METHODS if m[2] in _SHOWN_ORDERS]


def _two_sided_tail(cdf: np.ndarray) -> np.ndarray:
    """近い側の裾確率 min(F, 1-F)。CDF が [0,1] を逸脱する場合も含めそのまま返す。"""
    return np.minimum(cdf, 1.0 - cdf)


def _name_suffix(name: str) -> str:
    """プロット見出し用: シナリオ名があれば ' — 名前'、無ければ空文字。"""
    return f" — {name}" if name else ""


def make_comparison_plot(
    sc: Scenario, n_mc: int, seed: int, n_grid: int,
    output_path: str, density_output_path: str | None = None,
    show_cos_nodp: bool = True, show_edgeworth: bool = True,
) -> None:
    print(f"\n=== {sc.name} ===")
    print(
        f"  Hit数={sc.n_hits}  平均={sc.mean:,.1f}  標準偏差={sc.std:,.1f}  "
        f"λ3={sc.lam3:.4f}  λ4={sc.lam4:.4f}"
    )

    # Edgeworth 系手法 (_METHODS) は show_edgeworth のときだけ含める
    methods = _METHODS if show_edgeworth else []

    # MC
    print(f"  MC サンプリング {n_mc:,} 件 ...")
    samples = mc_survival_samples(sc.hit_mixtures, n_mc, seed)
    samples.sort()

    # x グリッド (両裾を広めに、サポート内にクリップ)
    x_lo = max(sc.support_lo + 1e-6 * (sc.support_hi - sc.support_lo),
               sc.mean - 6.0 * sc.std)
    x_hi = min(sc.support_hi - 1e-6 * (sc.support_hi - sc.support_lo),
               sc.mean + 8.0 * sc.std)
    xs = np.linspace(x_lo, x_hi, n_grid)
    z = (xs - sc.mean) / sc.std

    # 各手法の CDF → 両側裾確率
    method_tail: dict[str, np.ndarray] = {}
    for label, _color, order in methods:
        _pdf, cdf = edgeworth_pdf_cdf(
            xs, sc.mean, sc.std, sc.lam3, sc.lam4, order, sc.lam5, sc.lam6,
        )
        method_tail[label] = _two_sided_tail(cdf)

    # COS 法 (特性関数の数値反転): DP あり (原子分離) を基準線とする。DP なし
    # (素の全CF) は離散で Gibbs が出るため、比較用に show_cos_nodp のときだけ重ねる。
    cos_F_dp = cos_cdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)
    method_tail["COS法 (DPあり)"] = np.minimum(cos_F_dp, 1.0 - cos_F_dp)
    if show_cos_nodp:
        cos_F_nodp = cos_cdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=False)
        method_tail["COS法 (DPなし)"] = np.minimum(cos_F_nodp, 1.0 - cos_F_nodp)
    print(f"  COS: 区間=[{sc.cos_a:,.0f}, {sc.cos_b:,.0f}] (キュムラント窓∩サポート), 項数={sc.cos_n}")

    # MC 経験裾確率: searchsorted で F(x)=P(S<=x) を求める
    n = samples.size
    cdf_mc = np.searchsorted(samples, xs, side="right") / n
    tail_mc = np.minimum(cdf_mc, 1.0 - cdf_mc)
    # 信頼できるのは「近い側」の件数が十分ある領域だけ
    near_count = np.minimum(
        np.searchsorted(samples, xs, side="right"),
        n - np.searchsorted(samples, xs, side="right"),
    )
    MIN_COUNT = 50
    reliable = near_count >= MIN_COUNT

    # ---- 描画 ----
    fig, (ax_tail, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        f"裾確率 比較: COS法{' vs Edgeworth' if show_edgeworth else ''}"
        f"{_name_suffix(sc.name)}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
        fontsize=13,
    )

    # 上段: 両側裾確率 (log y)
    cos_dp_color = "magenta"
    cos_nodp_color = "tab:cyan"
    for label, color, _order in methods:
        tail = method_tail[label]
        ax_tail.plot(z, np.where(tail > 0, tail, np.nan), color=color, lw=1.5,
                     label=label)
    ax_tail.plot(z, np.where(method_tail["COS法 (DPあり)"] > 0,
                             method_tail["COS法 (DPあり)"], np.nan),
                 color=cos_dp_color, lw=1.8, ls="--", label="COS法 (DPあり)")
    if show_cos_nodp:
        ax_tail.plot(z, np.where(method_tail["COS法 (DPなし)"] > 0,
                                 method_tail["COS法 (DPなし)"], np.nan),
                     color=cos_nodp_color, lw=1.2, label="COS法 (DPなし)")
    ax_tail.plot(z[reliable], tail_mc[reliable], "k.", ms=4,
                 label=f"モンテカルロ ({n:,} 件)")

    ax_tail.set_yscale("log")
    ax_tail.set_ylabel("両側裾確率  min(F, 1−F)")
    ax_tail.set_ylim(max(1.0 / n / 10, 1e-8), 1.0)
    ax_tail.axvline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax_tail.grid(True, which="both", alpha=0.3)
    ax_tail.legend(loc="lower center", fontsize=9, ncol=2)

    # 下段: MC に対する相対誤差 (信頼領域のみ)
    eps = 1e-300
    all_methods = list(methods) + [("COS法 (DPあり)", cos_dp_color, -1)]
    if show_cos_nodp:
        all_methods = all_methods + [("COS法 (DPなし)", cos_nodp_color, -1)]
    for label, color, _order in all_methods:
        tail = method_tail[label]
        rel = (tail - tail_mc) / (tail_mc + eps)
        rel_plot = np.where(reliable & (tail_mc > 0), rel, np.nan)
        ax_err.plot(z, rel_plot, color=color, lw=1.4, label=label)
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.set_ylabel("相対誤差 (手法 − MC) / MC")
    ax_err.set_ylim(-1.0, 1.0)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper center", fontsize=9, ncol=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {output_path}")

    _print_tail_table(sc, samples)

    # 別画像: 密度関数の比較 (同じ MC サンプルを再利用)
    if density_output_path:
        _save_density_plot(sc, samples, n_grid, density_output_path,
                           show_cos_nodp, show_edgeworth)


def _save_density_plot(
    sc: Scenario, samples: np.ndarray, n_grid: int, output_path: str,
    show_cos_nodp: bool = True, show_edgeworth: bool = True,
) -> None:
    """密度関数 f(x) の比較を別画像として保存する。

    上段: 標準化密度 g(z)=σ·f(x) を MC 経験密度 (灰バー) / Edgeworth / COS法
          DPあり (原子分離) / COS法 DPなし (素の全CF, show_cos_nodp 時のみ) で
          重ねる。離散・混合では DPなしが原子の周りで Gibbs 振動 (負の値) を出し、
          DPありは連続部だけを滑らかに与える (純原子部はデルタなので密度には現れない)。
    下段: MC 経験密度を基準にした各手法の密度差 (手法 − MC)。MC のビン中心で評価する。"""
    std = sc.std
    z_lo = max((sc.support_lo - sc.mean) / std, -4.5)
    z_hi = min((sc.support_hi - sc.mean) / std, 4.5)
    zs = np.linspace(z_lo, z_hi, n_grid)
    xs = sc.mean + zs * std

    # MC 経験密度 (標準化スケール)
    z_samp = (samples - sc.mean) / std
    n_bins = 120
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    counts, _ = np.histogram(z_samp, bins=edges)
    bin_w = edges[1] - edges[0]
    mc_g = counts / (samples.size * bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xs_centers = sc.mean + centers * std

    methods = _METHODS if show_edgeworth else []

    # 各手法の標準化密度 g(z) = σ·f(x)。滑らかな線用 (zs) と、MC 差分用 (ビン中心)。
    method_g: dict[str, np.ndarray] = {}
    method_g_centers: dict[str, np.ndarray] = {}
    for label, _color, order in methods:
        pdf, _cdf = edgeworth_pdf_cdf(
            xs, sc.mean, sc.std, sc.lam3, sc.lam4, order, sc.lam5, sc.lam6,
        )
        method_g[label] = std * pdf
        pdf_c, _ = edgeworth_pdf_cdf(
            xs_centers, sc.mean, sc.std, sc.lam3, sc.lam4, order, sc.lam5, sc.lam6,
        )
        method_g_centers[label] = std * pdf_c
    g_cos_dp = std * cos_pdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)
    g_cos_dp_centers = std * cos_pdf(
        sc.grouped, xs_centers, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True
    )
    if show_cos_nodp:
        g_cos_nodp = std * cos_pdf(
            sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=False
        )
        g_cos_nodp_centers = std * cos_pdf(
            sc.grouped, xs_centers, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=False
        )

    fig, (ax_pdf, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        f"密度関数 比較: COS法{' vs Edgeworth' if show_edgeworth else ''}"
        f"{_name_suffix(sc.name)}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
        fontsize=13,
    )

    # 上段: 標準化密度
    ax_pdf.bar(centers, mc_g, width=bin_w * 0.95, alpha=0.3, color="gray",
               label=f"MC 経験密度 ({samples.size:,} 件)")
    for label, color, _order in methods:
        ax_pdf.plot(zs, method_g[label], color=color, lw=1.6, label=label)
    if show_cos_nodp:
        ax_pdf.plot(zs, g_cos_nodp, color="tab:cyan", lw=1.2, label="COS法 (DPなし)")
    ax_pdf.plot(zs, g_cos_dp, color="magenta", lw=1.8, ls="--", label="COS法 (DPあり)")
    ax_pdf.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax_pdf.set_ylabel("密度 (標準化スケール)")
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.legend(loc="upper right", fontsize=9)

    # 下段: MC 経験密度を基準にした密度差 (手法 − MC, MC ビン中心で評価)
    for label, color, _order in methods:
        ax_err.plot(centers, method_g_centers[label] - mc_g, color=color, lw=1.3,
                    label=f"{label} − MC")
    ax_err.plot(centers, g_cos_dp_centers - mc_g, color="magenta", lw=1.4, ls="--",
                label="COS(DPあり) − MC")
    if show_cos_nodp:
        ax_err.plot(centers, g_cos_nodp_centers - mc_g, color="tab:cyan", lw=1.4,
                    label="COS(DPなし) − MC")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.set_ylabel("密度差 (手法 − MC, 標準化)")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=9, ncol=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  密度プロット保存: {output_path}")


def _print_tail_table(
    sc: Scenario, samples: np.ndarray,
) -> None:
    """代表的な右裾分位点 x = MC の (1-p) 分位点 において、各手法の裾確率
    P(S>x) を「厳密に同じ x」で評価し、真値 p に対する相対誤差を表示する。"""
    targets = [0.1, 0.05, 0.01, 1e-3, 1e-4]
    print("  右裾 P(S>x) 比較 (各手法を MC 分位点 x で評価, 真値 p に対する相対誤差):")
    print("    P目標(=p)   z      Edge2    COS(DP)  COS(DPなし)")
    n = samples.size
    for p in targets:
        if p * n < 30:
            continue
        x = float(np.quantile(samples, 1.0 - p))
        zz = (x - sc.mean) / sc.std
        xa = np.array([x])
        cells = []
        for order in (2,):
            _pdf, cdf = edgeworth_pdf_cdf(
                xa, sc.mean, sc.std, sc.lam3, sc.lam4, order,
                sc.lam5, sc.lam6,
            )
            sf = float(1.0 - cdf[0])
            cells.append(f"{(sf - p) / p:+8.2%}")
        sf_dp = 1.0 - float(cos_cdf(sc.grouped, xa, sc.cos_a, sc.cos_b,
                                    sc.cos_n, hybrid=True)[0])
        cells.append(f"{(sf_dp - p) / p:+8.2%}")
        sf_nodp = 1.0 - float(cos_cdf(sc.grouped, xa, sc.cos_a, sc.cos_b,
                                      sc.cos_n, hybrid=False)[0])
        cells.append(f"{(sf_nodp - p) / p:+8.2%}")
        print(f"    {p:8.0e}  {zz:5.2f} " + " ".join(f"{c:>9}" for c in cells))


# =============================================================================
# シナリオ定義
# =============================================================================

def real_scenario() -> Scenario:
    """edgeworth_animation の実データ (時系列順 132 Hit)。"""
    cards, _bp = _build_cards_and_breakpoints()
    hit_mixtures, _cb = build_all_hits(
        cards, global_crit=61.35, global_evade=0.0, damage_mode="post_decay"
    )
    return Scenario("実データ (ブルアカ 全 Hit)", hit_mixtures)


def toy_scenario() -> Scenario:
    """CLT が効きにくい、少数・中程度の右歪みの設定。

    会心 30% (U(200k,500k)) / 非会心 70% (U(20k,220k)) の Hit を 8 回。
    レンジを重ねて一峰性・滑らかに保ちつつ右歪み (λ3≈0.29) を持たせることで、
    「COS法は裾全域でほぼ厳密、Edgeworth は深い裾で系統的に劣化」
    という対比をクリーンに見せる。"""
    mix = [
        Uniform(0.65, 4452673, 5479693),
        Uniform(0.35, 1033533, 1324131),
    ]
    hits = [mix for _ in range(8)]
    return Scenario("歪んだ少数 Hit (会心65%×8発)", hits)


def discrete_scenario() -> Scenario:
    """完全離散の例: 各 Hit が「会心(大)/非会心(小)」の2点分布 (幅0=点質量)。
    値の異なる4種を重ねるので和の原子が密に分布し (相異なる原子 ≈ 2352 個)、
    CDF は巨視的に滑らかで裾確率の比較が意味を持つ。一方で和は離散なので、
    DPなし COS は原子の周りで Gibbs 振動 (負の密度) を出し、DPあり COS は原子
    分離でこれを回避する — その差を可視化するためのシナリオ。"""
    groups = [
        ([Uniform(0.58, 5_010_000, 5_010_000), Uniform(0.42, 1_230_000, 1_230_000)], 6),
        ([Uniform(0.58, 4_330_000, 4_330_000), Uniform(0.42, 1_070_000, 1_070_000)], 7),
        ([Uniform(0.58, 5_790_000, 5_790_000), Uniform(0.42, 1_450_000, 1_450_000)], 5),
        ([Uniform(0.58, 4_660_000, 4_660_000), Uniform(0.42, 1_310_000, 1_310_000)], 6),
    ]
    hits: list[list[Uniform]] = []
    for mix, cnt in groups:
        hits += [mix] * cnt
    return Scenario("完全離散 (点入力・4種×24発)", hits)


# =============================================================================
# 設定 & main
# =============================================================================

N_MC = 20_000_000
N_GRID = 300
SEED = 42
OUT_DIR = "experiments/output"


def main() -> None:
    scenarios = [
        (toy_scenario(), os.path.join(OUT_DIR, "cos_tail_toy.png"),
         os.path.join(OUT_DIR, "cos_density_toy.png")),
        (real_scenario(), os.path.join(OUT_DIR, "cos_tail_real.png"),
         os.path.join(OUT_DIR, "cos_density_real.png")),
        (discrete_scenario(), os.path.join(OUT_DIR, "cos_tail_discrete.png"),
         os.path.join(OUT_DIR, "cos_density_discrete.png")),
    ]
    for sc, tail_path, density_path in scenarios:
        make_comparison_plot(sc, N_MC, SEED, N_GRID, tail_path,
                             density_output_path=density_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
