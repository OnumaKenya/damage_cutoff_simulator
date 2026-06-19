"""一般化 Irwin–Hall (Bates) による合計ダメージ S_n の厳密解。

このプロジェクトの 1Hit は「一様分布の混合」(experiments.edgeworth_animation.Uniform
のリスト) で、合計 S_n = Σ_i X_i は独立和。各 Hit が混合なので、各成分を 1 つ選ぶ
「組合せ」で条件付けると、条件付き分布は **幅の異なる独立一様和** になる。これは
一般化 Irwin–Hall 分布であり、切断べき乗 (truncated power) の包除公式で厳密に書ける:

    Y = Σ_{k=1}^n U(0, L_k)  (L_k = 各一様の幅) について
        F_Y(y) = 1/(n! ΠL_k) · Σ_{T⊆{1..n}} (-1)^{|T|} (y − σ_T)_+^n
        f_Y(y) = 1/((n−1)! ΠL_k) · Σ_{T} (-1)^{|T|} (y − σ_T)_+^{n−1}
            σ_T = Σ_{k∈T} L_k

各 Hit の最小値 a_k を A=Σa_k にまとめると S = A + Y。混合全体では、組合せ c に
重み W_c = Π_i w_{i,c_i} を付けて

    F_S(x) = Σ_c W_c · F_{Y_c}(x − A_c),   f_S(x) = Σ_c W_c · f_{Y_c}(x − A_c)

これは漸近近似 (Edgeworth) でも数値反転 (COS) でもない **閉形式の厳密解** で、
区分多項式 (次数 n−1) になる。退化成分 (幅 0 = 点質量, 例: 回避) は幅リストから
外し a_k だけ A に足す (次元が 1 つ減る)。ある組合せの Hit が全て点質量なら純粋な
原子 (デルタ) で、CDF には階段、PDF には寄与しない。

メイドアリス TL シナリオは 7Hit・全成分が連続一様 (回避 0%)・組合せ 192 通りなので、
24576 (=192×2^7) 項で厳密に評価できる。COS 法 (準厳密) の検証基準になり、Edgeworth
が裾で乖離することを定量化できる。

実行例:
    uv run python -m experiments.irwinhall_exact
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from experiments.edgeworth_animation import Uniform

# 数値安定化のためのスケール (この単位で幅・x を扱う)。包除公式は (y−σ)^n の
# 巨大値どうしの相殺を含むため、~O(1) に正規化して float64 の有効桁を温存する。
_SCALE = 1.0e6
_ATOM_TOL = 1.0e-9  # 幅がこれ以下なら点質量とみなす (スケール後)


def _component_list(hit_mixtures: list[list[Uniform]]) -> list[list[tuple[float, float, float]]]:
    """各 Hit を [(weight, lo, hi), ...] (スケール後) のリストに変換。重みは正規化済み前提。"""
    hits: list[list[tuple[float, float, float]]] = []
    for mix in hit_mixtures:
        comps = [(c.weight, c.lo / _SCALE, c.hi / _SCALE) for c in mix if c.weight > 0.0]
        hits.append(comps)
    return hits


def _truncated_power_cdf(
    x_minus_A: np.ndarray, widths: list[float], n_full: int,
) -> np.ndarray:
    """Y=ΣU(0,L_k) の CDF を点 (x−A) で評価 (一般化 Irwin–Hall, 包除)。

    widths は幅 0 を除いた正の幅リスト (長さ m)。点質量は呼び出し側で A に吸収済み。
    n_full は元の Hit 数だが、CDF/PDF の次数・規格化は実効次元 m で決まる。
    """
    m = len(widths)
    if m == 0:
        # 連続成分なし: 純粋な原子 (x≥A で 1, それ未満で 0) の階段
        return (x_minus_A >= 0.0).astype(float)
    L = np.asarray(widths, dtype=float)
    prodL = float(np.prod(L))
    inv = 1.0 / (math.factorial(m) * prodL)
    acc = np.zeros_like(x_minus_A, dtype=float)
    # 2^m 個の部分集合を列挙 (m≤7 なので 128 以下)
    for r in range(m + 1):
        for subset in itertools.combinations(range(m), r):
            sign = -1.0 if (r & 1) else 1.0
            sigma = float(L[list(subset)].sum()) if subset else 0.0
            diff = x_minus_A - sigma
            np.maximum(diff, 0.0, out=diff)
            acc += sign * diff ** m
    return inv * acc


def _truncated_power_pdf(
    x_minus_A: np.ndarray, widths: list[float],
) -> np.ndarray:
    """Y=ΣU(0,L_k) の PDF を点 (x−A) で評価 (スケール単位の密度)。"""
    m = len(widths)
    if m == 0:
        return np.zeros_like(x_minus_A, dtype=float)  # 純原子は密度に寄与しない
    L = np.asarray(widths, dtype=float)
    prodL = float(np.prod(L))
    inv = 1.0 / (math.factorial(m - 1) * prodL)
    acc = np.zeros_like(x_minus_A, dtype=float)
    for r in range(m + 1):
        for subset in itertools.combinations(range(m), r):
            sign = -1.0 if (r & 1) else 1.0
            sigma = float(L[list(subset)].sum()) if subset else 0.0
            diff = x_minus_A - sigma
            np.maximum(diff, 0.0, out=diff)
            acc += sign * diff ** (m - 1)
    return inv * acc


def exact_cdf_sf_pdf(
    hit_mixtures: list[list[Uniform]], xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合計 S_n の厳密 CDF・生存関数・PDF を xs (元スケール) 上で返す。

    CDF F(x)=P(S_n≤x) は下端からの引数 (x−A) で、生存関数 SF(x)=P(S_n>x) は
    各組合せの **対称性 1−F_Y(y)=F_Y(ΣL−y)** を使い上端からの引数 (上端−x) で
    別々に評価する。一般化 Irwin–Hall の包除は引数が小さい (=有効部分集合が少ない)
    ほど桁落ちが軽いので、F は左半分・SF は右半分でそれぞれ機械精度を保つ。
    SF を 1−F で代用すると上側裾で F≈1 からの引き算により ~1e−7 のノイズ床
    (負の確率) が出るため、両側裾には min(F, SF) を使うこと。

    全組合せ (各 Hit から成分を 1 つ選ぶ) に一般化 Irwin–Hall を重み付き加算する。
    """
    hits = _component_list(hit_mixtures)
    n_full = len(hits)
    xs = np.asarray(xs, dtype=float)
    xs_s = xs / _SCALE

    cdf = np.zeros_like(xs_s, dtype=float)
    sf = np.zeros_like(xs_s, dtype=float)
    pdf = np.zeros_like(xs_s, dtype=float)

    # 各 Hit の成分を直積で回す。組合せ数 = Π (成分数)。
    for combo in itertools.product(*hits):
        weight = 1.0
        A = 0.0       # 下端 Σ lo (点質量込み)
        upper = 0.0   # 上端 Σ hi (点質量込み)
        widths: list[float] = []
        for (w, lo, hi) in combo:
            weight *= w
            A += lo
            upper += hi
            width = hi - lo
            if width > _ATOM_TOL:
                widths.append(width)
        if weight == 0.0:
            continue
        # 下側: F_Y(x−A) / 上側: SF=F_Y(上端−x) (Y の対称性で反射)。各々
        # 引数が小さい裾側で安定。pdf は対称なのでどちらの引数でも可。
        cdf += weight * _truncated_power_cdf(xs_s - A, widths, n_full)
        sf += weight * _truncated_power_cdf(upper - xs_s, widths, n_full)
        pdf += weight * _truncated_power_pdf(xs_s - A, widths)

    pdf = pdf / _SCALE  # スケール単位の密度 → 元スケールの密度
    return cdf, sf, pdf


def exact_cdf_pdf(
    hit_mixtures: list[list[Uniform]], xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """合計 S_n の厳密 CDF F(x)=P(S_n≤x) と PDF f(x) を返す (互換 API)。"""
    cdf, _sf, pdf = exact_cdf_sf_pdf(hit_mixtures, xs)
    return cdf, pdf


def exact_tail(hit_mixtures: list[list[Uniform]], xs: np.ndarray) -> np.ndarray:
    """両側裾確率 min(F, SF) を厳密に返す (上側は反射で評価し桁落ちを回避)。"""
    cdf, sf, _pdf = exact_cdf_sf_pdf(hit_mixtures, xs)
    return np.minimum(cdf, sf)


if __name__ == "__main__":
    # 自己検証: メイドアリス TL で MC・COS と一致するか
    from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS
    from experiments.cos_compare import (
        Scenario, cos_cdf, mc_survival_samples,
    )
    from experiments.edgeworth_animation import build_all_hits

    hm, _ = build_all_hits(
        _BLUEARCHIVE_7HIT_CARDS, global_crit=65.27, global_evade=0.0,
        damage_mode="post_decay",
    )
    sc = Scenario("メイドアリスTL", hm)
    print(f"Hit数={sc.n_hits}  平均={sc.mean:,.0f}  σ={sc.std:,.0f}")
    n_combo = 1
    for cnt, mix in sc.grouped:
        n_combo *= len(mix) ** cnt
    print(f"組合せ数={n_combo}  (= Π 成分数^Hit数)")

    xs = np.linspace(sc.support_lo, sc.support_hi, 21)
    cdf_exact, pdf_exact = exact_cdf_pdf(hm, xs)
    cdf_cos = cos_cdf(sc.grouped, xs, sc.cos_a, sc.cos_b, sc.cos_n, hybrid=True)

    print("\n  x            F_exact      F_COS        |diff|")
    for x, fe, fc in zip(xs, cdf_exact, cdf_cos):
        print(f"  {x:13,.0f}  {fe:.8f}  {fc:.8f}  {abs(fe-fc):.2e}")
    print(f"\nmax|F_exact − F_COS| = {np.max(np.abs(cdf_exact - cdf_cos)):.3e}")

    # 全質量チェック: F(support_hi) ≈ 1, F(support_lo) ≈ 0
    edge = exact_cdf_pdf(hm, np.array([sc.support_lo - 1, sc.support_hi + 1]))[0]
    print(f"F(下限−)={edge[0]:.2e}  F(上限+)={edge[1]:.10f}")
