"""特性関数の「直接 FFT 反転」と COS 法の比較実験。

docs/saddlepoint.md・docs/edge.md の理論に基づき、1Hit のダメージを一様分布の
混合とみなす点・閉形式の特性関数 φ_S(u) = Π_hit φ_hit(u) を使う点は
cos_compare.py と共通。本スクリプトでは「同じ特性関数を反転する」2 つの数値
スキームを並べる:

    - COS 法 (Fang & Oosterlee 2008): φ を周波数格子 u_k = kπ/(b−a) で評価し、
      フーリエ余弦級数で密度・CDF を再構成 (cos_compare.py の cos_cdf/cos_pdf)。
    - 直接 FFT 反転 (Gil-Pelaez / Carr-Madan 型): 反転積分
          f(x) = 1/(2π) ∫ e^{−iux} φ(u) du
      を周波数格子で台形離散化し、逆 FFT で空間「相反格子」上の密度を一括計算。
      CDF は密度の累積台形積分、任意点はその格子からの補間で得る。

両者とも MC 真値を基準線に比較する。要点は「特性関数があるなら直接 FFT すれば
いい」という素朴な発想と COS 法の差を可視化すること:

    (i) 格子の相反制約 Δx·Δu = 2π/N — 空間範囲・空間分解能・周波数カットオフを
        独立に選べず、近離散な (φ の減衰が遅い) 設定では N を大きくしないと裾が
        荒れる。COS は周波数 u_k を自分で選び、この制約が無い。
    (ii) 折り返し (wraparound) / Gibbs — FFT は周期性を課すため、窓端で密度が
         0 でない・原子があると振動が出る。COS は有界台と区分多項式構造に整合。
    (iii) CDF・裾確率は密度を数値積分して作るので誤差が一段増える。COS は項別
          解析積分で CDF を閉形式で出す。

実行例:
    uv run python -m experiments.fft_compare
"""
from __future__ import annotations

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# cos_compare 経由で edgeworth_animation の import (フォント設定等) も済む
from experiments.cos_compare import (  # noqa: E402
    GroupedHits,
    Scenario,
    _METHODS,            # 表示する Edgeworth 系手法の選択 (cos_compare と共通)
    _name_suffix,
    cos_cdf,
    cos_pdf,
    discrete_scenario,
    edgeworth_pdf_cdf,
    mc_survival_samples,
    real_scenario,
    sum_cf,
    toy_scenario,
)

# 直接 FFT 反転の既定 FFT サイズ (2 の冪)。N を上げるほど周波数カットオフ
# U=(N/2)·Δu が上がり (近離散で有利) Δx が縮む (空間が細かくなる) が、CF 評価
# 点数も N に比例して増える。COS の項数 (sc.cos_n) と対比するための主ノブ。
_FFT_N_DEFAULT = 1 << 14


# =============================================================================
# 特性関数の直接 FFT 反転 (Gil-Pelaez / Carr-Madan 型)
#   空間窓 [a, b] は COS と同じものを使い (apples-to-apples)、唯一の差を
#   「反転スキーム (余弦級数 vs 逆 FFT)」に絞る。
#
#   反転積分を中心化周波数 u_j=(j−N/2)·Δu (j=0..N−1) で台形離散化:
#       f(x_k) = Δu/(2π) Σ_j φ(u_j) e^{−i u_j x_k},   x_k = a + k·Δx
#   ここで相反制約 Δx·Δu = 2π/N を課すと
#       e^{−i u_j x_k} = e^{−i u_j a} · e^{−i2π jk/N} · (−1)^k
#   と分解でき、括弧内 b_j=φ(u_j)e^{−i u_j a} の FFT 一発で全 x_k が出る。
#       Δx = (b−a)/N,   Δu = 2π/(b−a)  (← N に依らず窓幅だけで決まる)
#       周波数カットオフ U_cut = (N/2)·Δu = πN/(b−a)
# =============================================================================

def fft_invert_grid(
    grouped: GroupedHits, a: float, b: float, n_fft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """直接 FFT 反転で空間相反格子上の (x, pdf, cdf) と診断情報 dict を返す。

    pdf は逆 FFT の実部 (近離散だと Gibbs/折り返しで負になりうる — あえて素のまま)。
    cdf は pdf の累積台形積分 (正規化しない: 総質量 cdf[-1] の 1 からのずれを
    FFT の健全性として表に出すため)。"""
    N = int(n_fft)
    if N & (N - 1) != 0:                          # 2 の冪に丸める (FFT 効率)
        N = 1 << (N - 1).bit_length()
    dx = (b - a) / N
    du = 2.0 * math.pi / (b - a)                  # = 2π/(N·Δx); 周波数分解能
    j = np.arange(N)
    u = (j - N // 2) * du                          # 中心化周波数
    phi = sum_cf(grouped, u)

    # 周波数和の台形重み (端点を半分)。Carr-Madan は Simpson だが台形で十分。
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    seq = phi * np.exp(-1j * u * a) * w            # b_j = φ(u_j) e^{−i u_j a}
    transformed = np.fft.fft(seq)
    sign = np.where(j % 2 == 0, 1.0, -1.0)         # (−1)^k
    pdf = (du / (2.0 * math.pi)) * sign * transformed.real
    x = a + j * dx

    # CDF: 密度の累積台形積分 (COS のような項別解析積分は使えない — FFT の弱点)
    incr = 0.5 * (pdf[1:] + pdf[:-1]) * dx
    cdf = np.concatenate([[0.0], np.cumsum(incr)])  # 長さ N

    info = {
        "N": N,
        "dx": dx,
        "du": du,
        "u_cut": (N // 2) * du,                     # 周波数カットオフ U_cut
        "mass": float(cdf[-1]),                     # ∫f dx (健全性: 1 に近いか)
        "pdf_min": float(pdf.min()),                # 負値の深さ (Gibbs の目安)
    }
    return x, pdf, cdf, info


def fft_cdf(
    grouped: GroupedHits, xs: np.ndarray, a: float, b: float, n_fft: int,
) -> np.ndarray:
    """直接 FFT 反転による CDF を xs 上で返す (相反格子からの線形補間)。"""
    x, _pdf, cdf, _info = fft_invert_grid(grouped, a, b, n_fft)
    return np.clip(np.interp(xs, x, cdf), 0.0, 1.0)


def fft_pdf(
    grouped: GroupedHits, xs: np.ndarray, a: float, b: float, n_fft: int,
) -> np.ndarray:
    """直接 FFT 反転による密度を xs 上で返す (相反格子からの線形補間)。"""
    x, pdf, _cdf, _info = fft_invert_grid(grouped, a, b, n_fft)
    return np.interp(xs, x, pdf)


def _two_sided(cdf: np.ndarray) -> np.ndarray:
    return np.minimum(cdf, 1.0 - cdf)


# =============================================================================
# プロット: 直接 FFT 反転 vs COS (準厳密基準) vs MC 真値
# =============================================================================

_COS_COLOR = "magenta"        # COS法 (DPあり) — cos_compare と同色
_COS_NODP_COLOR = "tab:cyan"  # COS法 (DPなし) — cos_compare と同色
_FFT_COLOR = "tab:green"      # 直接 FFT 反転 (本アプリで追加する手法)

# 裾確率・密度プロットの線スタイル (cos_compare に合わせる)。Edgeworth 系は
# _METHODS 側の色を使い、ここに無い手法は既定 (実線・lw=1.5)。
_LINE_STYLE = {
    "COS法 (DPあり)": dict(lw=1.8, ls="--"),
    "COS法 (DPなし)": dict(lw=1.2, ls="-"),
    "直接 FFT 反転": dict(lw=1.3, ls="-"),
}


def make_comparison_plot(
    sc: Scenario, n_mc: int, n_mc_small: int, seed: int, n_grid: int,
    output_path: str, density_output_path: str | None = None,
    n_fft: int = _FFT_N_DEFAULT, convergence_output_path: str | None = None,
) -> None:
    print(f"\n=== {sc.name} ===")
    print(
        f"  Hit数={sc.n_hits}  平均={sc.mean:,.1f}  標準偏差={sc.std:,.1f}  "
        f"λ3={sc.lam3:.4f}  λ4={sc.lam4:.4f}"
    )

    # 窓は COS と共通 (差を反転スキームだけに絞る)
    a, b = sc.cos_a, sc.cos_b

    # MC 真値
    print(f"  MC サンプリング {n_mc:,} 件 ...")
    samples = mc_survival_samples(sc.hit_mixtures, n_mc, seed)
    samples.sort()
    n = samples.size

    # x グリッド (両裾を広めにサポート内へクリップ)
    x_lo = max(sc.support_lo + 1e-6 * (sc.support_hi - sc.support_lo),
               sc.mean - 6.0 * sc.std)
    x_hi = min(sc.support_hi - 1e-6 * (sc.support_hi - sc.support_lo),
               sc.mean + 8.0 * sc.std)
    xs = np.linspace(x_lo, x_hi, n_grid)
    z = (xs - sc.mean) / sc.std

    # --- 各手法の CDF を同じ窓で計算。表示手法は cos_compare (_METHODS) と共通:
    #     Edgeworth 系 + COS法 (DPあり/DPなし) に、本アプリの 直接 FFT 反転 を加える ---
    method_tail: dict[str, tuple[str, np.ndarray]] = {}  # label -> (color, tail)
    for label, color, order in _METHODS:
        _pdf, cdf = edgeworth_pdf_cdf(
            xs, sc.mean, sc.std, sc.lam3, sc.lam4, order, sc.lam5, sc.lam6,
        )
        method_tail[label] = (color, _two_sided(cdf))

    t0 = time.perf_counter()
    cos_F = cos_cdf(sc.grouped, xs, a, b, sc.cos_n, hybrid=True)
    t_cos = time.perf_counter() - t0
    cos_F_nodp = cos_cdf(sc.grouped, xs, a, b, sc.cos_n, hybrid=False)

    t0 = time.perf_counter()
    fx, fpdf, fcdf, finfo = fft_invert_grid(sc.grouped, a, b, n_fft)
    fft_F = np.clip(np.interp(xs, fx, fcdf), 0.0, 1.0)
    t_fft = time.perf_counter() - t0

    method_tail["COS法 (DPあり)"] = (_COS_COLOR, _two_sided(cos_F))
    method_tail["COS法 (DPなし)"] = (_COS_NODP_COLOR, _two_sided(cos_F_nodp))
    method_tail["直接 FFT 反転"] = (_FFT_COLOR, _two_sided(fft_F))

    cos_tail = method_tail["COS法 (DPあり)"][1]
    fft_tail = method_tail["直接 FFT 反転"][1]

    print(
        f"  窓=[{a:,.0f}, {b:,.0f}] (キュムラント窓∩サポート)\n"
        f"  COS : 項数={sc.cos_n:>6}  {t_cos*1e3:7.1f} ms\n"
        f"  FFT : N   ={finfo['N']:>6}  {t_fft*1e3:7.1f} ms  "
        f"Δx={finfo['dx']:,.0f}  Δu={finfo['du']:.3e}  "
        f"U_cut={finfo['u_cut']:.3e}\n"
        f"        ∫f dx={finfo['mass']:.6f} (=1 が理想)  "
        f"pdf_min={finfo['pdf_min']:.3e} (負なら Gibbs/折り返し)"
    )

    # MC 経験裾確率 (近い側の件数が十分な領域のみ信頼)
    cdf_mc = np.searchsorted(samples, xs, side="right") / n
    tail_mc = np.minimum(cdf_mc, 1.0 - cdf_mc)
    near_count = np.minimum(
        np.searchsorted(samples, xs, side="right"),
        n - np.searchsorted(samples, xs, side="right"),
    )
    reliable = near_count >= 50

    # 少数サンプル MC (線)
    samples_small = mc_survival_samples(sc.hit_mixtures, n_mc_small, seed + 1)
    samples_small.sort()
    cdf_small = np.searchsorted(samples_small, xs, side="right") / n_mc_small
    tail_small = np.minimum(cdf_small, 1.0 - cdf_small)

    # ---- 描画: 裾確率 ----
    fig, (ax_tail, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        f"裾確率 比較: 直接 FFT 反転 vs COS 法 vs Edgeworth/MC{_name_suffix(sc.name)}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f}, "
        f"COS項数={sc.cos_n}, FFT N={finfo['N']})",
        fontsize=13,
    )

    for label, (color, tail) in method_tail.items():
        st = _LINE_STYLE.get(label, dict(lw=1.5, ls="-"))
        ax_tail.plot(z, np.where(tail > 0, tail, np.nan), color=color,
                     label=label, **st)
    ax_tail.plot(z, np.where(tail_small > 0, tail_small, np.nan),
                 color="tab:purple", lw=1.1, alpha=0.85,
                 label=f"モンテカルロ ({n_mc_small:,} 件・線)")
    ax_tail.plot(z[reliable], tail_mc[reliable], "k.", ms=4,
                 label=f"モンテカルロ ({n:,} 件・真値)")
    ax_tail.set_yscale("log")
    ax_tail.set_ylabel("両側裾確率  min(F, 1−F)")
    ax_tail.set_ylim(max(1.0 / n / 10, 1e-8), 1.0)
    ax_tail.axvline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax_tail.grid(True, which="both", alpha=0.3)
    ax_tail.legend(loc="lower center", fontsize=9, ncol=2)

    # 下段: MC に対する相対誤差 (全手法)
    eps = 1e-300
    for label, (color, tail) in method_tail.items():
        rel = (tail - tail_mc) / (tail_mc + eps)
        ax_err.plot(z, np.where(reliable & (tail_mc > 0), rel, np.nan),
                    color=color, lw=1.4, label=label)
    rel_small = (tail_small - tail_mc) / (tail_mc + eps)
    ax_err.plot(z, np.where(reliable & (tail_mc > 0), rel_small, np.nan),
                color="tab:purple", lw=1.1, alpha=0.85,
                label=f"モンテカルロ ({n_mc_small:,} 件・線)")
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

    _print_tail_table(sc, samples, a, b, n_fft)

    if density_output_path:
        _save_density_plot(sc, samples, n_grid, a, b, n_fft, density_output_path)

    if convergence_output_path:
        _save_convergence_plot(sc, samples, a, b, n_fft, convergence_output_path)


def _save_density_plot(
    sc: Scenario, samples: np.ndarray, n_grid: int,
    a: float, b: float, n_fft: int, output_path: str,
) -> None:
    """密度 f(x) の比較。表示手法は cos_compare と共通 (Edgeworth 系 + COS法
    DPあり/DPなし) に 直接 FFT 反転 を加える。上段: MC 経験密度 (灰) と各手法の
    標準化密度 g(z)=σ·f(x)。下段: COS法 (DPあり) を基準にした密度差。FFT−COS
    と COS(DPなし)−COS(DP) がそれぞれの Gibbs/折り返し振動になる。"""
    std = sc.std
    z_lo = max((sc.support_lo - sc.mean) / std, -4.5)
    z_hi = min((sc.support_hi - sc.mean) / std, 4.5)
    zs = np.linspace(z_lo, z_hi, n_grid)
    xs = sc.mean + zs * std

    z_samp = (samples - sc.mean) / std
    n_bins = 120
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    counts, _ = np.histogram(z_samp, bins=edges)
    bin_w = edges[1] - edges[0]
    mc_g = counts / (samples.size * bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    method_g: dict[str, tuple[str, np.ndarray]] = {}  # label -> (color, g)
    for label, color, order in _METHODS:
        pdf, _cdf = edgeworth_pdf_cdf(
            xs, sc.mean, sc.std, sc.lam3, sc.lam4, order, sc.lam5, sc.lam6,
        )
        method_g[label] = (color, std * pdf)
    g_cos = std * cos_pdf(sc.grouped, xs, a, b, sc.cos_n, hybrid=True)
    g_cos_nodp = std * cos_pdf(sc.grouped, xs, a, b, sc.cos_n, hybrid=False)
    fx, fpdf, _fcdf, _info = fft_invert_grid(sc.grouped, a, b, n_fft)
    g_fft = std * np.interp(xs, fx, fpdf)
    method_g["COS法 (DPなし)"] = (_COS_NODP_COLOR, g_cos_nodp)
    method_g["直接 FFT 反転"] = (_FFT_COLOR, g_fft)
    g_cos_centers = std * cos_pdf(sc.grouped, sc.mean + centers * std,
                                  a, b, sc.cos_n, hybrid=True)

    fig, (ax_pdf, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        f"密度関数 比較: 直接 FFT 反転 vs COS 法 vs Edgeworth/MC{_name_suffix(sc.name)}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}, λ4={sc.lam4:.3f})",
        fontsize=13,
    )

    # 上段: MC 経験密度 + 各手法 + COS法 (DPあり, 基準) を最後に重ねる
    ax_pdf.bar(centers, mc_g, width=bin_w * 0.95, alpha=0.3, color="gray",
               label=f"MC 経験密度 ({samples.size:,} 件)")
    for label, (color, g) in method_g.items():
        st = _LINE_STYLE.get(label, dict(lw=1.5, ls="-"))
        ax_pdf.plot(zs, g, color=color, label=label, **st)
    ax_pdf.plot(zs, g_cos, color=_COS_COLOR, lw=1.8, ls="--",
                label="COS法 (DPあり)")
    ax_pdf.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax_pdf.set_ylabel("密度 (標準化スケール)")
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.legend(loc="upper right", fontsize=9)

    # 下段: COS法 (DPあり) を基準にした密度差
    for label, (color, g) in method_g.items():
        ax_err.plot(zs, g - g_cos, color=color, lw=1.3, label=f"{label} − COS(DP)")
    ax_err.plot(centers, mc_g - g_cos_centers, color="gray", lw=0.8, alpha=0.6,
                label="MC − COS(DP) (ノイズ確認)")
    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.set_ylabel("密度差 (− COS(DP), 標準化)")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=9, ncol=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  密度プロット保存: {output_path}")


# 収束曲線で掃く CF 評価点数 (COS 項数 M = FFT サイズ N)。基準 (機械精度の
# 真値) は COS を高項数で評価したもの。M は基準より十分粗い範囲に取る。
_CONV_MS = (128, 256, 512, 1024, 2048, 4096, 8192)
_CONV_GOLD_TERMS = 1 << 16
_CONV_ZMAX = 3.2          # 評価する z 範囲 (両側裾)
_CONV_TAIL_FLOOR = 1e-7   # この裾確率より浅い点だけで最大相対誤差を測る


def _save_convergence_plot(
    sc: Scenario, samples: np.ndarray, a: float, b: float, n_fft: int,
    output_path: str,
) -> None:
    """同じ CF 評価点数 (COS 項数 M = FFT サイズ N) での収束を 2 段で比較する。

    横軸 (共通): CF 評価点数 (= COS 項数 M / FFT サイズ N)。多 Hit ではこれが計算量
          (各点で全 Hit にわたる sinc 積) に比例する。

    上段 (基準比): 基準 (COS 高項数 = 機械精度の真値) に対する両側裾確率の
          最大相対誤差。COS は余弦級数の項別解析積分で少項で機械精度に達し、直接
          FFT 反転は密度の数値積分ぶん N^{-2} に律速される。

    下段 (MC 比): 独立基準である MC (本図では {n:,} 件) に対する RMS 相対誤差。
          基準は COS 自身なので循環の懸念があるが、MC は完全独立。両手法とも
          MC のサンプルノイズ床まで下がるとそれ以上は判別できず、床で頭打ちになる。
          床 (判別不能ライン) は MC 真値と基準の RMS 相対差で、サンプル数 n に
          対し ~1/√n で下がる。床より上で MC は手法差を検証でき、FFT の N^{-2} が
          MC とも整合することが確認できる。"""
    xs = sc.mean + np.linspace(-_CONV_ZMAX, _CONV_ZMAX, 400) * sc.std
    gold = cos_cdf(sc.grouped, xs, a, b, _CONV_GOLD_TERMS, hybrid=True)
    gtail = np.minimum(gold, 1.0 - gold)
    mask = gtail > _CONV_TAIL_FLOOR
    if not mask.any():
        return

    # MC 経験裾確率 (件数が十分な点だけ信頼)。下段・床はすべてこの領域で測る。
    n = samples.size
    cnt_hi = np.searchsorted(samples, xs, side="right")
    mc_tail = np.minimum(cnt_hi / n, 1.0 - cnt_hi / n)
    mc_ok = mask & (np.minimum(cnt_hi, n - cnt_hi) >= 50)

    def max_rel_gold(tail: np.ndarray) -> float:
        return float(np.max(np.abs((tail - gtail)[mask] / gtail[mask])))

    def rms_rel_mc(tail: np.ndarray) -> float:
        if not mc_ok.any():
            return float("nan")
        return float(np.sqrt(np.mean(((tail - mc_tail)[mc_ok] / mc_tail[mc_ok]) ** 2)))

    ms = [m for m in _CONV_MS if m < _CONV_GOLD_TERMS // 4]
    cos_g, fft_g, cos_m, fft_m = [], [], [], []
    for m in ms:
        ct = _two_sided(cos_cdf(sc.grouped, xs, a, b, m, hybrid=True))
        ft = _two_sided(fft_cdf(sc.grouped, xs, a, b, m))
        cos_g.append(max(max_rel_gold(ct), 1e-16))
        fft_g.append(max(max_rel_gold(ft), 1e-16))
        cos_m.append(rms_rel_mc(ct))
        fft_m.append(rms_rel_mc(ft))
    # 判別不能ライン: MC 真値と基準の RMS 相対差 (= MC 自身のノイズ床)。n に依存。
    mc_floor = rms_rel_mc(gtail)

    fig, (ax_g, ax_m) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    fig.suptitle(
        f"収束比較: 直接 FFT 反転 vs COS 法{_name_suffix(sc.name)}\n"
        f"(Hit数={sc.n_hits}, λ3={sc.lam3:.3f}; 上=基準比, 下=MC比 {n:,}件)",
        fontsize=12,
    )

    # 上段: 基準比 (機械精度の真値に対する最大相対誤差)
    ax_g.loglog(ms, cos_g, "o-", color=_COS_COLOR, lw=1.8, label="COS法")
    ax_g.loglog(ms, fft_g, "s-", color=_FFT_COLOR, lw=1.5, label="直接 FFT 反転")
    ax_g.axvline(sc.cos_n, color=_COS_COLOR, ls=":", lw=1.0, alpha=0.7,
                 label=f"COS 既定項数 = {sc.cos_n}")
    ax_g.axvline(n_fft, color=_FFT_COLOR, ls=":", lw=1.0, alpha=0.7,
                 label=f"FFT 既定 N = {n_fft}")
    ax_g.set_ylabel(f"基準比 最大相対誤差 (z∈±{_CONV_ZMAX:.1f})")
    ax_g.grid(True, which="both", alpha=0.3)
    ax_g.legend(fontsize=9)

    # 下段: MC 比 (独立基準) — 床まで下がると判別不能
    ax_m.loglog(ms, cos_m, "o-", color=_COS_COLOR, lw=1.8, label="COS法 (MC比)")
    ax_m.loglog(ms, fft_m, "s-", color=_FFT_COLOR, lw=1.5, label="直接 FFT 反転 (MC比)")
    if mc_floor == mc_floor:  # not NaN
        ax_m.axhline(mc_floor, color="tab:purple", lw=1.6, ls="-.",
                     label=f"MC ノイズ床 ({n:,} 件・判別不能ライン)")
        ax_m.axhspan(mc_floor * 1e-2, mc_floor, color="tab:purple", alpha=0.08)
    ax_m.axvline(sc.cos_n, color=_COS_COLOR, ls=":", lw=1.0, alpha=0.7)
    ax_m.axvline(n_fft, color=_FFT_COLOR, ls=":", lw=1.0, alpha=0.7)
    ax_m.set_xlabel("CF 評価点数  (COS 項数 M = FFT サイズ N)")
    ax_m.set_ylabel(f"MC比 RMS相対誤差 (z∈±{_CONV_ZMAX:.1f})")
    ax_m.grid(True, which="both", alpha=0.3)
    ax_m.legend(fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  収束プロット保存: {output_path}")


def _print_tail_table(
    sc: Scenario, samples: np.ndarray, a: float, b: float, n_fft: int,
) -> None:
    """右裾分位点 x で各手法の P(S>x) を評価し、真値 p に対する相対誤差を表示。"""
    targets = [0.1, 0.05, 0.01, 1e-3, 1e-4]
    print("  右裾 P(S>x) 比較 (MC 分位点 x で評価, 真値 p に対する相対誤差):")
    print("    P目標(=p)   z       COS      FFT直接")
    n = samples.size
    for p in targets:
        if p * n < 30:
            continue
        x = float(np.quantile(samples, 1.0 - p))
        zz = (x - sc.mean) / sc.std
        xa = np.array([x])
        sf_cos = 1.0 - float(cos_cdf(sc.grouped, xa, a, b, sc.cos_n,
                                     hybrid=True)[0])
        sf_fft = 1.0 - float(fft_cdf(sc.grouped, xa, a, b, n_fft)[0])
        print(f"    {p:8.0e}  {zz:5.2f}  {(sf_cos - p) / p:+8.2%} "
              f"{(sf_fft - p) / p:+8.2%}")


# =============================================================================
# 設定 & main
# =============================================================================

N_MC = 20_000_000
N_MC_SMALL = 200_000
N_GRID = 300
SEED = 42
N_FFT = _FFT_N_DEFAULT
OUT_DIR = "experiments/output"


def main() -> None:
    scenarios = [
        (toy_scenario(), "fft_tail_toy.png", "fft_density_toy.png",
         "fft_conv_toy.png"),
        (real_scenario(), "fft_tail_real.png", "fft_density_real.png",
         "fft_conv_real.png"),
        (discrete_scenario(), "fft_tail_discrete.png", "fft_density_discrete.png",
         "fft_conv_discrete.png"),
    ]
    for sc, tail_name, density_name, conv_name in scenarios:
        make_comparison_plot(
            sc, N_MC, N_MC_SMALL, SEED, N_GRID,
            os.path.join(OUT_DIR, tail_name),
            density_output_path=os.path.join(OUT_DIR, density_name),
            n_fft=N_FFT,
            convergence_output_path=os.path.join(OUT_DIR, conv_name),
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
