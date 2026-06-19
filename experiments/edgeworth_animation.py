"""Edgeworth 展開 vs Monte Carlo アニメーション実験。

docs/edge.md の理論に基づき、各 Hit (またはカード) を1つずつ累積していく過程で
正規分布近似 (CLT) / Edgeworth 展開 1次・2次 / モンテカルロ 経験分布 を比較する
アニメーション (GIF) を生成する。

入力はアプリと同じパラメータ構造 (carit_min/max, normal_min/max, hits, crit_rate,
evade_rate) を保持し、ファイル下部の CARDS を編集することで自由に設定できる。

実行例:
    uv run python -m experiments.edgeworth_animation
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import io

import matplotlib
matplotlib.use("Agg")  # GIF出力のためヘッドレスでも動作させる
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 日本語ラベルを表示するため japanize-matplotlib を読み込む
# (パッケージが見つからない場合のフォールバックを含む)
try:
    import japanize_matplotlib  # noqa: F401  (副作用で rcParams が書き換わる)
except ImportError:
    # 既知の CJK フォント名を試す
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic",
                  "Hiragino Sans", "Yu Gothic", "Meiryo", "MS Gothic"):
        if _name in available:
            plt.rcParams["font.family"] = _name
            break
    else:
        print(
            "[警告] 日本語フォントが見つかりません。ラベルが文字化けする可能性があります。\n"
            "       `uv add --dev japanize-matplotlib` を実行してください。"
        )

# プロジェクトルートを sys.path に追加して、既存実装 (decay 関数定義) を再利用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.simulation import DAMAGE_FUNC, inverse_decay  # noqa: E402

# =============================================================================
# 一様分布混合の表現
# =============================================================================

@dataclass
class Uniform:
    """1 つの一様分布成分。退化 (lo == hi) は1点分布を表す。"""
    weight: float
    lo: float
    hi: float

    @property
    def center(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)


def _decay_array(x: np.ndarray) -> np.ndarray:
    """生ダメージ → 減衰後ダメージ (np.ndarray対応)。"""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    for (x_lo, x_hi), (a, b) in DAMAGE_FUNC:
        mask = (x >= x_lo) & (x < x_hi)
        y[mask] = a * x[mask] + b
    # 最終段境界 x == x_hi のとき上のループから外れるため補完
    cap = DAMAGE_FUNC[-1][1][1]
    cap_mask = x >= DAMAGE_FUNC[-1][0][0]
    y[cap_mask & ~np.isfinite(y)] = cap
    return y


def split_uniform_through_decay(a: float, b: float) -> list[tuple[float, float, float]]:
    """生ダメージ上の U(a, b) を decay 関数を通して post-decay scale の混合に分解する。

    生ダメージ域の各 decay セグメントに該当する部分が、post-decay 域でそれぞれ
    1つの (場合により退化した) 一様分布になる。重みは元の幅に比例する。

    Returns:
        list of (weight, post_decay_lo, post_decay_hi) で重み合計は 1.0。
    """
    if b <= a:
        # 1点分布
        y = float(_decay_array(np.array([a]))[0])
        return [(1.0, y, y)]

    parts: list[tuple[float, float, float]] = []
    total_width = b - a
    for (x_lo, x_hi), (a_d, b_d) in DAMAGE_FUNC:
        lo = max(a, x_lo)
        hi = min(b, x_hi)
        if hi <= lo:
            continue
        w = (hi - lo) / total_width
        y_lo = a_d * lo + b_d
        y_hi = a_d * hi + b_d
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        parts.append((w, y_lo, y_hi))
    if not parts:
        # decay テーブル範囲外: 恒等
        return [(1.0, a, b)]
    return parts


# =============================================================================
# カード → 1Hitの混合分布
# =============================================================================

def expand_card_to_hit_mixture(
    card: dict, global_crit: float, global_evade: float, damage_mode: str
) -> list[Uniform]:
    """1枚のカードから、各Hit (同分布) に対応する1つの混合分布を作る。"""
    crit_min = float(card.get("crit_min") or 0)
    crit_max = float(card.get("crit_max") or 0)
    normal_min = float(card.get("normal_min") or 0)
    normal_max = float(card.get("normal_max") or 0)
    cr_raw = card.get("crit_rate")
    er_raw = card.get("evade_rate")
    cr = float(cr_raw if cr_raw is not None else (global_crit or 0)) / 100.0
    er = float(er_raw if er_raw is not None else (global_evade or 0)) / 100.0

    if damage_mode == "post_decay":
        # アプリ同様、入力は減衰後値。生ダメージ域へ戻して decay を再適用する
        raw_crit_lo = inverse_decay(crit_min)
        raw_crit_hi = inverse_decay(crit_max)
        raw_norm_lo = inverse_decay(normal_min)
        raw_norm_hi = inverse_decay(normal_max)
    else:
        raw_crit_lo, raw_crit_hi = crit_min, crit_max
        raw_norm_lo, raw_norm_hi = normal_min, normal_max
    raw_crit_hi = max(raw_crit_hi, raw_crit_lo)
    raw_norm_hi = max(raw_norm_hi, raw_norm_lo)

    crit_parts = split_uniform_through_decay(raw_crit_lo, raw_crit_hi)
    norm_parts = split_uniform_through_decay(raw_norm_lo, raw_norm_hi)

    mixture: list[Uniform] = []
    if (1 - er) * cr > 0:
        for (w, lo, hi) in crit_parts:
            mixture.append(Uniform((1 - er) * cr * w, lo, hi))
    if (1 - er) * (1 - cr) > 0:
        for (w, lo, hi) in norm_parts:
            mixture.append(Uniform((1 - er) * (1 - cr) * w, lo, hi))
    if er > 0:
        mixture.append(Uniform(er, 0.0, 0.0))

    # 重みを正規化
    total = sum(u.weight for u in mixture)
    if total > 0:
        for u in mixture:
            u.weight /= total
    return mixture


def build_all_hits(
    cards: list[dict], global_crit: float, global_evade: float, damage_mode: str
) -> tuple[list[list[Uniform]], list[int]]:
    """全カードを (各Hitの混合分布のリスト) に展開。

    Returns:
        hit_mixtures: 各Hitの混合分布 (長さ = 総Hit数)
        card_boundaries: 各カードの累積Hit数 (card-mode アニメ用)
    """
    hit_mixtures: list[list[Uniform]] = []
    card_boundaries: list[int] = []
    for c in cards:
        mix = expand_card_to_hit_mixture(c, global_crit, global_evade, damage_mode)
        hits = int(c.get("hits") or 1)
        for _ in range(hits):
            hit_mixtures.append(mix)
        card_boundaries.append(len(hit_mixtures))
    return hit_mixtures, card_boundaries


# =============================================================================
# 1Hitのキュムラント計算 (docs/edge.md 「一様分布の混合分布」)
# =============================================================================

def hit_cumulants(
    mixture: list[Uniform],
) -> tuple[float, float, float, float, float, float]:
    """1Hitの混合分布から (平均, 分散, 3次, 4次, 5次, 6次キュムラント) を返す。
    モーメント関係:
        κ3 = M3
        κ4 = M4 − 3 M2²
        κ5 = M5 − 10 M3 M2
        κ6 = M6 − 15 M4 M2 − 10 M3² + 30 M2³
    各次の中心モーメントは Σ_j w_j Σ_{k=0,2,4,..} C(r,k) δ_j^{r-k} h_j^k/(k+1)。"""
    if not mixture:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    w = np.array([u.weight for u in mixture])
    c = np.array([u.center for u in mixture])
    h = np.array([u.half_width for u in mixture])
    mu = float(np.sum(w * c))
    d = c - mu
    d2 = d ** 2
    d3 = d ** 3
    d4 = d ** 4
    d5 = d ** 5
    d6 = d ** 6
    h2 = h ** 2
    h4 = h ** 4
    h6 = h ** 6
    M2 = float(np.sum(w * (h2 / 3.0 + d2)))
    M3 = float(np.sum(w * (d * h2 + d3)))
    M4 = float(np.sum(w * (h4 / 5.0 + 2.0 * d2 * h2 + d4)))
    M5 = float(np.sum(w * (d5 + (10.0 / 3.0) * d3 * h2 + d * h4)))
    M6 = float(np.sum(w * (d6 + 5.0 * d4 * h2 + 3.0 * d2 * h4 + h6 / 7.0)))
    kappa3 = M3
    kappa4 = M4 - 3.0 * M2 ** 2
    kappa5 = M5 - 10.0 * M3 * M2
    kappa6 = M6 - 15.0 * M4 * M2 - 10.0 * M3 ** 2 + 30.0 * M2 ** 3
    return mu, M2, kappa3, kappa4, kappa5, kappa6


# =============================================================================
# Edgeworth 展開
# =============================================================================

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / _SQRT_2PI


def _Phi(z: np.ndarray) -> np.ndarray:
    # math.erf をベクトル化
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / _SQRT_2))


def _He2(z): return z ** 2 - 1.0
def _He3(z): return z ** 3 - 3.0 * z
def _He4(z): return z ** 4 - 6.0 * z ** 2 + 3.0
def _He5(z): return z ** 5 - 10.0 * z ** 3 + 15.0 * z
def _He6(z): return z ** 6 - 15.0 * z ** 4 + 45.0 * z ** 2 - 15.0
def _He7(z): return z ** 7 - 21.0 * z ** 5 + 105.0 * z ** 3 - 105.0 * z
def _He8(z): return z ** 8 - 28.0 * z ** 6 + 210.0 * z ** 4 - 420.0 * z ** 2 + 105.0
def _He9(z): return (z ** 9 - 36.0 * z ** 7 + 378.0 * z ** 5
                    - 1260.0 * z ** 3 + 945.0 * z)
def _He10(z): return (z ** 10 - 45.0 * z ** 8 + 630.0 * z ** 6
                     - 3150.0 * z ** 4 + 4725.0 * z ** 2 - 945.0)
def _He11(z): return (z ** 11 - 55.0 * z ** 9 + 990.0 * z ** 7
                     - 6930.0 * z ** 5 + 17325.0 * z ** 3 - 10395.0 * z)
def _He12(z): return (z ** 12 - 66.0 * z ** 10 + 1485.0 * z ** 8
                     - 13860.0 * z ** 6 + 51975.0 * z ** 4
                     - 62370.0 * z ** 2 + 10395.0)


def edgeworth_pdf_cdf(
    x: np.ndarray, mean: float, std: float, lam3: float, lam4: float, order: int,
    lam5: float = 0.0, lam6: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """ダメージスケール x に対する (pdf, cdf) を返す。

    order = 0: 正規分布近似 (CLT)
    order = 1: Edgeworth 1次  (λ3)
    order = 2: Edgeworth 2次  (+ λ4, λ3²)
    order = 3: Edgeworth 3次  (+ λ5, λ3·λ4, λ3³)
    order = 4: Edgeworth 4次  (+ λ6, λ4²+λ3·λ5, λ3²·λ4, λ3⁴)

    漸近展開なので次数を増やせば精度が上がる保証はなく、特に裾では高次の補正
    が振動・発散しがち。order≥3 では lam5, order=4 では lam6 も渡すこと。"""
    if std <= 0:
        pdf = np.zeros_like(x)
        cdf = (x >= mean).astype(float)
        return pdf, cdf

    z = (x - mean) / std
    phi = _phi(z)
    Phi = _Phi(z)

    # 補正項を次数ごとに累積。Σ_s P_s(z) で密度、Σ_s Q_s(z) (= P_s で He_k→He_{k-1})
    # で CDF を補正する。
    pdf_corr = np.zeros_like(z)
    cdf_corr = np.zeros_like(z)
    if order >= 1:
        pdf_corr = pdf_corr + lam3 / 6.0 * _He3(z)
        cdf_corr = cdf_corr + lam3 / 6.0 * _He2(z)
    if order >= 2:
        pdf_corr = (pdf_corr
                    + lam4 / 24.0 * _He4(z)
                    + lam3 ** 2 / 72.0 * _He6(z))
        cdf_corr = (cdf_corr
                    + lam4 / 24.0 * _He3(z)
                    + lam3 ** 2 / 72.0 * _He5(z))
    if order >= 3:
        pdf_corr = (pdf_corr
                    + lam5 / 120.0 * _He5(z)
                    + lam3 * lam4 / 144.0 * _He7(z)
                    + lam3 ** 3 / 1296.0 * _He9(z))
        cdf_corr = (cdf_corr
                    + lam5 / 120.0 * _He4(z)
                    + lam3 * lam4 / 144.0 * _He6(z)
                    + lam3 ** 3 / 1296.0 * _He8(z))
    if order >= 4:
        coeff_He8 = lam4 ** 2 / 1152.0 + lam3 * lam5 / 720.0
        pdf_corr = (pdf_corr
                    + lam6 / 720.0 * _He6(z)
                    + coeff_He8 * _He8(z)
                    + lam3 ** 2 * lam4 / 1728.0 * _He10(z)
                    + lam3 ** 4 / 31104.0 * _He12(z))
        cdf_corr = (cdf_corr
                    + lam6 / 720.0 * _He5(z)
                    + coeff_He8 * _He7(z)
                    + lam3 ** 2 * lam4 / 1728.0 * _He9(z)
                    + lam3 ** 4 / 31104.0 * _He11(z))

    pdf = phi * (1.0 + pdf_corr) / std
    cdf = Phi - phi * cdf_corr
    return pdf, cdf


# =============================================================================
# Monte Carlo: 各Hitのサンプルを生成し cumsum を取る
# =============================================================================

def simulate_per_hit_samples(
    hit_mixtures: list[list[Uniform]], n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """各Hit に n_samples 個のサンプルを生成し、形状 (n_hits, n_samples) で返す。"""
    n_hits = len(hit_mixtures)
    out = np.zeros((n_hits, n_samples))
    for i, mix in enumerate(hit_mixtures):
        if not mix:
            continue
        weights = np.array([u.weight for u in mix])
        weights = weights / weights.sum()
        comp_idx = rng.choice(len(mix), size=n_samples, p=weights)
        for j, u in enumerate(mix):
            mask = comp_idx == j
            n = int(mask.sum())
            if n == 0:
                continue
            if u.hi > u.lo:
                out[i, mask] = rng.uniform(u.lo, u.hi, size=n)
            else:
                out[i, mask] = u.lo
    return out


# =============================================================================
# シミュレーション結果データクラス + 共有ヘルパー
# =============================================================================

@dataclass
class SimResult:
    """1度の MC + キュムラント計算結果。複数の可視化に再利用できる。

    生サンプルは保持せず、各 Hit prefix の z 標準化ヒストグラムカウントだけを
    持つことでメモリを節約する (132 hits × 120 bins ≒ 16k 値)。"""
    n_hits: int
    card_boundaries: list[int]
    cum_mean: np.ndarray
    cum_var: np.ndarray
    cum_k3: np.ndarray
    cum_k4: np.ndarray
    # z 標準化ヒストグラムのカウント (n_hits × hist_bins)
    z_histogram_counts: np.ndarray
    z_bin_edges: np.ndarray
    n_mc_samples: int

    @property
    def z_bin_centers(self) -> np.ndarray:
        return 0.5 * (self.z_bin_edges[:-1] + self.z_bin_edges[1:])

    def mc_density_raw(self, k: int) -> np.ndarray:
        """Hit prefix k の MC 経験密度 (生・未平滑化)。"""
        bin_width = self.z_bin_edges[1] - self.z_bin_edges[0]
        return self.z_histogram_counts[k] / (self.n_mc_samples * bin_width)


def _make_gauss_kernel(sigma_bins: float) -> np.ndarray:
    """MC 密度を平滑化する Gauss kernel (σ は bin 数単位)。"""
    if sigma_bins <= 0:
        return np.array([1.0])
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return k / k.sum()


def _smooth(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size <= 1:
        return arr
    return np.convolve(arr, kernel, mode="same")


def _standardized_pdf(
    z: np.ndarray, lam3: float, lam4: float, order: int
) -> np.ndarray:
    """標準化変数 z における理論 PDF (order=0: 正規, 1: Edgeworth 1次, 2: 2次)。"""
    phi = _phi(z)
    if order == 0:
        return phi
    if order == 1:
        return phi * (1.0 + lam3 / 6.0 * _He3(z))
    return phi * (
        1.0
        + lam3 / 6.0 * _He3(z)
        + lam4 / 24.0 * _He4(z)
        + lam3 ** 2 / 72.0 * _He6(z)
    )


# 3手法の (order, 色, ラベル)
_APPROXIMATIONS = [
    (0, "tab:blue", "正規分布近似 (CLT)"),
    (1, "tab:orange", "Edgeworth 1次"),
    (2, "tab:red", "Edgeworth 2次"),
]

Z_LIM = 5.0


# =============================================================================
# メイン: シミュレーション → アニメーション / sup 誤差プロット
# =============================================================================

def prepare_simulation(
    cards: list[dict],
    global_crit: float,
    global_evade: float,
    damage_mode: str,
    n_mc_samples: int,
    hist_bins: int,
    seed: int,
    chunk_size: int = 500_000,
) -> SimResult:
    """カード列からキュムラント (prefix) と各 Hit prefix の z 標準化ヒストグラム
    カウントを返す。MC は chunk_size 単位で生成し、サンプル本体は保持しない。"""
    rng = np.random.default_rng(seed)
    hit_mixtures, card_boundaries = build_all_hits(
        cards, global_crit, global_evade, damage_mode
    )
    n_hits = len(hit_mixtures)
    if n_hits == 0:
        raise ValueError("Hit がありません。CARDS を確認してください。")
    print(f"Total hits: {n_hits}, Cards: {len(card_boundaries)}")

    # キュムラントの prefix sum
    means = np.zeros(n_hits)
    vars_ = np.zeros(n_hits)
    k3s = np.zeros(n_hits)
    k4s = np.zeros(n_hits)
    for i, mix in enumerate(hit_mixtures):
        m, v, k3, k4, _k5, _k6 = hit_cumulants(mix)
        means[i], vars_[i], k3s[i], k4s[i] = m, v, k3, k4
    cum_mean = np.cumsum(means)
    cum_var = np.cumsum(vars_)
    cum_k3 = np.cumsum(k3s)
    cum_k4 = np.cumsum(k4s)
    cum_std = np.sqrt(np.where(cum_var > 0, cum_var, 1.0))

    # MC をチャンク処理し、各 Hit prefix の z 標準化ヒストグラム counts を蓄積
    z_bin_edges = np.linspace(-Z_LIM, Z_LIM, hist_bins + 1)
    z_histogram_counts = np.zeros((n_hits, hist_bins), dtype=np.int64)

    print(
        f"Simulating {n_hits} hits x {n_mc_samples:,} samples "
        f"(chunk={chunk_size:,}) ..."
    )
    processed = 0
    while processed < n_mc_samples:
        chunk = min(chunk_size, n_mc_samples - processed)
        samples = simulate_per_hit_samples(hit_mixtures, chunk, rng)
        np.cumsum(samples, axis=0, out=samples)
        for k in range(n_hits):
            if cum_var[k] <= 0:
                continue
            z = (samples[k] - cum_mean[k]) / cum_std[k]
            counts, _ = np.histogram(z, bins=z_bin_edges)
            z_histogram_counts[k] += counts
        del samples
        processed += chunk
        print(f"  {processed:,}/{n_mc_samples:,} processed")

    return SimResult(
        n_hits=n_hits,
        card_boundaries=card_boundaries,
        cum_mean=cum_mean,
        cum_var=cum_var,
        cum_k3=cum_k3,
        cum_k4=cum_k4,
        z_histogram_counts=z_histogram_counts,
        z_bin_edges=z_bin_edges,
        n_mc_samples=n_mc_samples,
    )


def _draw_frame_on_axes(
    ax_pdf,
    ax_err,
    sim: SimResult,
    k: int,
    label: str,
    z_grid: np.ndarray,
    smooth_kernel: np.ndarray,
) -> None:
    """既存の (ax_pdf, ax_err) 上に Hit prefix k のフレームを描画する。
    save_animation と save_single_frame_png の両方から呼ばれる共通描画ロジック。"""
    z_bin_edges = sim.z_bin_edges
    z_bin_centers = sim.z_bin_centers
    m = sim.cum_mean[k]
    var = sim.cum_var[k]
    s = math.sqrt(var) if var > 0 else 0.0
    lam3 = (sim.cum_k3[k] / s ** 3) if s > 0 else 0.0
    lam4 = (sim.cum_k4[k] / s ** 4) if s > 0 else 0.0

    ax_pdf.clear()
    ax_err.clear()

    if s <= 0:
        ax_pdf.set_title(f"{label} 分散=0 (描画不能)", fontsize=10)
        return

    mc_density_raw = sim.mc_density_raw(k)
    mc_density_smooth = _smooth(mc_density_raw, smooth_kernel)

    # 上段の密度プロットは平滑化なしの生 MC 密度を表示する
    ax_pdf.bar(
        z_bin_centers, mc_density_raw,
        width=(z_bin_edges[1] - z_bin_edges[0]) * 0.95,
        alpha=0.3, color="gray", label="MC 経験密度",
    )

    # 下段の (理論 − MC) 誤差は平滑化した MC 密度との差をとる
    max_abs_err = 0.0
    for order, color, lbl in _APPROXIMATIONS:
        pdf_curve = _standardized_pdf(z_grid, lam3, lam4, order)
        ax_pdf.plot(z_grid, pdf_curve, color=color, lw=1.6, label=lbl)
        pdf_at_bins = _standardized_pdf(z_bin_centers, lam3, lam4, order)
        err = pdf_at_bins - mc_density_smooth
        ax_err.plot(z_bin_centers, err, color=color, lw=1.3, label=lbl)
        max_abs_err = max(max_abs_err, float(np.max(np.abs(err))))

    ax_err.axhline(0.0, color="gray", lw=0.8, alpha=0.6)

    ax_pdf.set_title(
        f"{label}   "
        f"平均={m:,.0f}  標準偏差={s:,.0f}  "
        f"λ3={lam3:.3f}  λ4={lam4:.3f}  "
        f"sup|誤差|={max_abs_err:.4f}",
        fontsize=10,
    )
    ax_pdf.set_ylabel("密度 (標準化スケール)")
    ax_pdf.set_xlim(-Z_LIM, Z_LIM)
    ax_pdf.set_ylim(0.0, 0.55)
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.legend(loc="upper right", fontsize=8)

    ax_err.set_xlabel("z = (x − 平均) / 標準偏差")
    ax_err.set_ylabel("理論密度 − MC密度")
    lim = max(5e-4, max_abs_err * 1.15)
    ax_err.set_ylim(-lim, lim)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=8)


def _new_pdf_err_figure(title_suffix: str):
    fig, (ax_pdf, ax_err) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    fig.suptitle(
        f"Edgeworth 展開 vs モンテカルロ ({title_suffix}、標準化スケール)",
        fontsize=13,
    )
    return fig, ax_pdf, ax_err


def save_single_frame_png(
    sim: SimResult,
    k: int,
    label: str,
    title_suffix: str,
    output_path: str,
    smooth_sigma_bins: float,
    n_grid: int = 600,
    dpi: int = 120,
) -> None:
    """Hit prefix k の単独フレームを PNG として保存する。"""
    z_grid = np.linspace(-Z_LIM, Z_LIM, n_grid)
    smooth_kernel = _make_gauss_kernel(smooth_sigma_bins)

    fig, ax_pdf, ax_err = _new_pdf_err_figure(title_suffix)
    _draw_frame_on_axes(ax_pdf, ax_err, sim, k, label, z_grid, smooth_kernel)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving single-frame PNG to {output_path} (k={k}: {label}) ...")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_animation(
    sim: SimResult,
    frame_indices: list[int],
    frame_labels: list[str],
    title_suffix: str,
    output_path: str,
    fps: int,
    smooth_sigma_bins: float,
    n_grid: int = 600,
    hold_first_sec: float = 0.0,
    hold_last_sec: float = 0.0,
    dpi: int = 100,
) -> None:
    """sim 結果から指定フレームの GIF アニメーションを保存する。

    `hold_first_sec` / `hold_last_sec` で先頭/末尾フレームの表示時間 (秒) を
    延ばせる。GIF の Graphics Control Extension の delay を直接設定するため、
    余分なフレーム複製は行わない。"""
    z_grid = np.linspace(-Z_LIM, Z_LIM, n_grid)
    smooth_kernel = _make_gauss_kernel(smooth_sigma_bins)

    fig, ax_pdf, ax_err = _new_pdf_err_figure(title_suffix)

    def draw_frame(frame_i: int) -> None:
        _draw_frame_on_axes(
            ax_pdf, ax_err, sim,
            frame_indices[frame_i], frame_labels[frame_i],
            z_grid, smooth_kernel,
        )

    n_frames = len(frame_indices)
    base_dur_ms = max(1, int(round(1000.0 / fps)))
    durations = [base_dur_ms] * n_frames
    if n_frames > 0:
        durations[0] = max(durations[0], int(round(hold_first_sec * 1000.0)))
        durations[-1] = max(durations[-1], int(round(hold_last_sec * 1000.0)))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        f"Rendering {n_frames} frames for {output_path} "
        f"(fps={fps}, hold first/last={hold_first_sec:.1f}/{hold_last_sec:.1f}s) ..."
    )
    pil_frames: list[Image.Image] = []
    for i in range(n_frames):
        draw_frame(i)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        # convert で PNG バッファを完全に読み込ませる
        pil_frames.append(Image.open(buf).convert("RGB"))

    print(f"Saving animation to {output_path} ...")
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    plt.close(fig)


def compute_sup_errors(
    sim: SimResult, smooth_sigma_bins: float
) -> np.ndarray:
    """全 Hit 1..n_hits に対して 3 手法の sup|f_theory - f_MC| を返す。

    Returns: shape (3, n_hits) — 行: 正規 / Edgeworth1 / Edgeworth2、列: Hit prefix。
    """
    z_bin_centers = sim.z_bin_centers
    smooth_kernel = _make_gauss_kernel(smooth_sigma_bins)

    out = np.full((3, sim.n_hits), np.nan)
    for k in range(sim.n_hits):
        var = sim.cum_var[k]
        if var <= 0:
            continue
        s = math.sqrt(var)
        lam3 = sim.cum_k3[k] / s ** 3
        lam4 = sim.cum_k4[k] / s ** 4
        mc_density = _smooth(sim.mc_density_raw(k), smooth_kernel)
        for order, _color, _label in _APPROXIMATIONS:
            pdf_at_bins = _standardized_pdf(z_bin_centers, lam3, lam4, order)
            out[order, k] = float(np.max(np.abs(pdf_at_bins - mc_density)))
    return out


def save_sup_error_plot(
    sim: SimResult,
    smooth_sigma_bins: float,
    output_path: str,
    breakpoints: list[tuple[int, str]] | None = None,
) -> None:
    """全 Hit に対する 3 手法の sup 誤差を折れ線でプロットして PNG 保存する。"""
    errors = compute_sup_errors(sim, smooth_sigma_bins)
    hits_axis = np.arange(1, sim.n_hits + 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for order, color, label in _APPROXIMATIONS:
        ax.plot(hits_axis, errors[order], color=color, lw=1.4, label=label)

    # 発目区切りの縦線
    if breakpoints:
        for idx, label in breakpoints:
            ax.axvline(idx, color="gray", lw=0.5, alpha=0.4)
        # x 軸 2 段目に breakpoint ラベルを表示
        ax2 = ax.secondary_xaxis("top")
        ax2.set_xticks([idx for idx, _ in breakpoints])
        ax2.set_xticklabels([lab for _, lab in breakpoints], rotation=75,
                            ha="left", fontsize=7)

    ax.set_xlabel("累積 Hit 数")
    ax.set_ylabel("sup|理論密度 − MC密度|")
    ax.set_title("各手法の最大誤差 (Hit ごと累積、標準化スケール)")
    ax.set_yscale("log")
    ax.set_xlim(1, sim.n_hits)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sup error plot to {output_path}")


def _frame_indices_for_anim_mode(
    sim: SimResult, anim_mode: str, cards: list[dict]
) -> tuple[list[int], list[str], str]:
    """anim_mode から frame_indices / frame_labels / タイトル接尾辞を導出する。"""
    if anim_mode == "card":
        frame_indices = [b - 1 for b in sim.card_boundaries]
        frame_labels: list[str] = []
        for i, k in enumerate(frame_indices):
            memo = str((cards[i] or {}).get("memo") or "").strip()
            if memo:
                frame_labels.append(f"{memo} まで累積 ({k + 1} Hits)")
            else:
                frame_labels.append(f"累積 {k + 1} Hits")
        return frame_indices, frame_labels, "カードごと累積"
    # default: "hit"
    frame_indices = list(range(sim.n_hits))
    frame_labels = [f"Hit {i + 1}/{sim.n_hits}" for i in range(sim.n_hits)]
    return frame_indices, frame_labels, "Hit ごと累積"


def run_experiment(
    cards: list[dict],
    global_crit: float,
    global_evade: float,
    damage_mode: str,
    anim_mode: str,
    n_mc_samples: int,
    hist_bins: int,
    output_path: str,
    fps: int,
    seed: int,
    smooth_sigma_bins: float = 0.0,
    n_grid: int = 600,
) -> None:
    """既存 API: 1 つのアニメーション GIF を生成する。"""
    sim = prepare_simulation(
        cards, global_crit, global_evade, damage_mode,
        n_mc_samples, hist_bins, seed,
    )
    frame_indices, frame_labels, suffix = _frame_indices_for_anim_mode(
        sim, anim_mode, cards
    )
    save_animation(
        sim, frame_indices, frame_labels, suffix,
        output_path, fps, smooth_sigma_bins, n_grid,
    )


# =============================================================================
# 設定 (アプリのカード入力と同じ構造で記述する)
# =============================================================================

# アプリのダメージ生成モードと同じ意味: "post_decay" (減衰考慮済み入力) or "pre_decay"
DAMAGE_MODE = "post_decay"
GLOBAL_CRIT_RATE = 61.35  # %
GLOBAL_EVADE_RATE = 0     # %

# ─── ダメージプロファイル (タプル: crit_min, crit_max, normal_min, normal_max) ───
# 1, 2射目 子
_DMG_12_FULL_INI = (229_088, 292_713, 48_024, 61_362)
_DMG_12_FULL_REM = (228_540, 292_012, 47_909, 61_215)
_DMG_12_NOB_INI  = (106_217, 135_717, 22_266, 28_450)
_DMG_12_NOB_REM  = (105_963, 135_392, 22_213, 28_382)
# 2射目 親
_DMG_2_PAR_INI   = (136_389, 174_268, 28_591, 36_532)
_DMG_2_PAR_REM   = (136_062, 173_851, 28_523, 36_444)
# 3, 4射目 子
_DMG_34_FULL_INI = (251_035, 320_755, 52_624, 67_240)
_DMG_34_FULL_REM = (250_433, 319_986, 52_498, 67_079)
_DMG_34_NOB_INI  = (116_393, 148_719, 24_399, 31_176)
_DMG_34_NOB_REM  = (116_114, 148_362, 24_341, 31_101)
# 3, 4射目 親
_DMG_34_PAR_INI  = (149_455, 190_963, 31_330, 40_032)
_DMG_34_PAR_REM  = (149_097, 190_505, 31_255, 39_936)


def _card(dmg: tuple, hits: int, memo: str) -> dict:
    cm, cx, nm, nx = dmg
    return {
        "crit_min": cm, "crit_max": cx, "normal_min": nm, "normal_max": nx,
        "hits": hits, "crit_rate": None, "evade_rate": None, "memo": memo,
    }


def _slot_label(slot: int) -> str:
    return "初弾" if slot == 1 else f"{slot}発目"


def _build_cards_and_breakpoints() -> tuple[list[dict], list[tuple[int, str]]]:
    """時系列順 (1→2→3→4射目 × 初弾→2発目→…→6発目) でカードを展開し、
    各発目終了時点の (累積 Hit 数, ラベル) を `breakpoints` に記録する。"""
    cards: list[dict] = []
    breakpoints: list[tuple[int, str]] = []
    cum = 0
    for volley in (1, 2, 3, 4):
        for slot in range(1, 7):
            is_ini = slot == 1
            sname = _slot_label(slot)
            new: list[dict] = []

            # 子 (フルデバフ + デバフなし) — 1,2射目 と 3,4射目 でダメージ値が異なる
            if volley in (1, 2):
                dmg_full = _DMG_12_FULL_INI if is_ini else _DMG_12_FULL_REM
                dmg_nob  = _DMG_12_NOB_INI  if is_ini else _DMG_12_NOB_REM
            else:
                dmg_full = _DMG_34_FULL_INI if is_ini else _DMG_34_FULL_REM
                dmg_nob  = _DMG_34_NOB_INI  if is_ini else _DMG_34_NOB_REM
            new.append(_card(dmg_full, 3, f"{volley}射目{sname}・フルデバフ子(3体)"))

            # デバフなし子は 1射目=1体、2~4射目=2体
            nob_count = 1 if volley == 1 else 2
            new.append(_card(
                dmg_nob, nob_count,
                f"{volley}射目{sname}・デバフなし子({nob_count}体)",
            ))

            # 親は 2射目以降のみ
            if volley == 2:
                dmg_par = _DMG_2_PAR_INI if is_ini else _DMG_2_PAR_REM
                new.append(_card(dmg_par, 1, f"{volley}射目親{sname}"))
            elif volley in (3, 4):
                dmg_par = _DMG_34_PAR_INI if is_ini else _DMG_34_PAR_REM
                new.append(_card(dmg_par, 1, f"{volley}射目親{sname}"))

            cards.extend(new)
            cum += sum(c["hits"] for c in new)
            breakpoints.append((cum, f"{volley}射目 {sname}"))
    return cards, breakpoints


# 時系列順 (1→2→3→4射目 × 初弾→2発目→…→6発目) に展開されたカード列と、
# 各発目終了時点の (累積 Hit 数, ラベル)。アニメーションの key frame として使える。
CARDS, HIT_BREAKPOINTS = _build_cards_and_breakpoints()


# アニメ進行: "hit" = 1Hit ずつ、"card" = 1カードずつ
ANIM_MODE = "hit"
N_MC_SAMPLES = 10_000_000
HIST_BINS = 120
# MC 密度の平滑化幅 (bin 数単位)。0 で平滑化なし
SMOOTH_SIGMA_BINS = 1.5
OUTPUT_PATH = "experiments/output/edgeworth_animation.gif"
BREAKPOINT_OUTPUT_PATH = "experiments/output/edgeworth_breakpoints.gif"
SUP_ERROR_OUTPUT_PATH = "experiments/output/edgeworth_sup_error.png"
VOLLEY1_SLOT3_OUTPUT_PATH = "experiments/output/edgeworth_volley1_slot3.png"
FPS = 2
# GIF 先頭/末尾フレームを停止表示する秒数
HOLD_FIRST_SEC = 2.0
HOLD_LAST_SEC = 3.0
SEED = 42


def main() -> None:
    # MC は 1 回だけ (チャンク処理) 。3 つの出力で再利用する。
    sim = prepare_simulation(
        cards=CARDS,
        global_crit=GLOBAL_CRIT_RATE,
        global_evade=GLOBAL_EVADE_RATE,
        damage_mode=DAMAGE_MODE,
        n_mc_samples=N_MC_SAMPLES,
        hist_bins=HIST_BINS,
        seed=SEED,
    )

    # 1) Hit ごと累積アニメーション
    hit_frame_indices, hit_frame_labels, hit_suffix = _frame_indices_for_anim_mode(
        sim, ANIM_MODE, CARDS
    )
    save_animation(
        sim, hit_frame_indices, hit_frame_labels, hit_suffix,
        OUTPUT_PATH, FPS, SMOOTH_SIGMA_BINS,
        hold_first_sec=HOLD_FIRST_SEC, hold_last_sec=HOLD_LAST_SEC,
    )

    # 2) 発目 (HIT_BREAKPOINTS) ごと累積アニメーション
    bp_frame_indices = [idx - 1 for idx, _ in HIT_BREAKPOINTS]
    bp_frame_labels = [
        f"{label} まで累積 ({idx} Hits)" for idx, label in HIT_BREAKPOINTS
    ]
    save_animation(
        sim, bp_frame_indices, bp_frame_labels, "発目ごと累積",
        BREAKPOINT_OUTPUT_PATH, FPS, SMOOTH_SIGMA_BINS,
        hold_first_sec=HOLD_FIRST_SEC, hold_last_sec=HOLD_LAST_SEC,
    )

    # 3) 全 Hit にわたる sup 誤差の折れ線プロット
    save_sup_error_plot(
        sim, SMOOTH_SIGMA_BINS,
        SUP_ERROR_OUTPUT_PATH, breakpoints=HIT_BREAKPOINTS,
    )

    # 4) 1射目 3発目 (HIT_BREAKPOINTS[2] = (12, ...)) の単独 PNG
    bp_cum, bp_label = HIT_BREAKPOINTS[2]
    save_single_frame_png(
        sim, k=bp_cum - 1,
        label=f"{bp_label} まで累積 ({bp_cum} Hits)",
        title_suffix="発目ごと累積",
        output_path=VOLLEY1_SLOT3_OUTPUT_PATH,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )

    print("Done.")


if __name__ == "__main__":
    main()
