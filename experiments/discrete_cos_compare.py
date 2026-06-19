"""離散一様ダメージに対する連続 COS 近似の「幅 +1/2 広げ」効果の検証。

docs/discrete.md の解析を数値で確かめる。1Hit のダメージが整数格子上の
**離散一様分布** U_disc{a,…,b} (N=b−a+1 点, 各 1/N) であるとき、これを連続
一様で近似して COS 法を回す。比較するのは 2 つの連続化:

    - 素の幅   U(a, b)        : 既存文書の素直な連続化 (幅 N−1)
    - +1/2 広げ U(a−½, b+½)   : 連続性補正した連続化 (幅 N)

真値は和の **厳密な離散分布** (各 Hit の整数 PMF を FFT 畳み込み = docs/discrete.md
の「1 周期上の逆 DFT」と数学的に同じ、標本ノイズ無し)。さらに **MC は整数一様
分布** (rng.integers, 端点込み) で回し、厳密離散と一致することを確認する。

期待される結果 (docs/discrete.md より):
    - 3 者は **平均が完全一致** (対称な広げ) し、違いは **分散だけ**。
        σ²_disc = (N²−1)/12,  σ²_素 = (N−1)²/12,  σ²_広げ = N²/12
      素は分散を (N−1)/6 (= O(N)) 過小評価、+1/2 広げは +1/12 (定数) 過大評価。
    - よって素の幅は裾が細く (裾確率を過小評価)、+1/2 広げは厳密離散にほぼ重なる。
    - 相対的な改善率は ~2/(N+1)。格子が細かい (N 大) ほど両者の差は縮む
      = ブルアカスケール (N~10³–10⁴) では +1/2 補正の恩恵もごく小さい。

    uv run python -m experiments.discrete_cos_compare
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.cos_compare import Uniform, Scenario, cos_cdf  # noqa: E402

OUT_DIR = "experiments/output"
N_GRID = 700
N_MC = 10_000_000
SEED = 42

# 離散シナリオ = (名前, hits_spec)。
#   hits_spec: list[hit]、hit = list[(weight, a, b)] で a,b は **整数** 端点
#   (端点込み離散一様 {a,…,b})。a==b は真の 1 点 (原子) で広げない。
DiscreteHit = list  # list[tuple[float, int, int]]


def _scaled_hits(width: int, n_hits: int, base: int = 800) -> list[DiscreteHit]:
    """純粋な離散一様 U_disc{base,…,base+width} (N=width+1 点) を n_hits 回。

    本筋 (離散一様分布そのもの) を最もクリーンに見るため 1 成分・対称にする。和は
    対称な離散 Irwin–Hall になり、ハードエッジは ±√(n)·(width/2)/σ₁ ≈ 数 σ 先まで
    退くので、エッジに汚されない素直なガウス的裾で比較できる。width が小さいほど
    点数 N が小さく (格子が粗く) +1/2 補正が効く。"""
    mix: DiscreteHit = [(1.0, base, base + width)]   # 1 成分の離散一様
    return [mix for _ in range(n_hits)]


SCENARIOS = [
    ("粗い格子 (N=41)", _scaled_hits(width=40, n_hits=12)),
    ("細かい格子 (N=401)", _scaled_hits(width=400, n_hits=12)),
]


# =============================================================================
# 連続化 (素 / +1/2 広げ) と厳密離散分布
# =============================================================================

def narrow_mixtures(spec: list[DiscreteHit]) -> list[list[Uniform]]:
    """素の連続化 U(a, b) (幅 N−1)。"""
    return [[Uniform(w, float(a), float(b)) for (w, a, b) in hit] for hit in spec]


def widened_mixtures(spec: list[DiscreteHit]) -> list[list[Uniform]]:
    """+1/2 広げの連続化 U(a−½, b+½) (幅 N)。a==b の真の原子は広げない。"""
    out: list[list[Uniform]] = []
    for hit in spec:
        comps = []
        for (w, a, b) in hit:
            if a == b:
                comps.append(Uniform(w, float(a), float(b)))      # 原子はそのまま
            else:
                comps.append(Uniform(w, a - 0.5, b + 0.5))
        out.append(comps)
    return out


def exact_discrete_pmf(spec: list[DiscreteHit]) -> tuple[np.ndarray, np.ndarray]:
    """和 S_n の厳密な整数 PMF。各 Hit の整数 PMF を FFT 畳み込み (= 逆 DFT)。

    Returns: (values, pmf) で values は連続する整数、pmf は P(S_n = value)。"""
    dist = np.array([1.0])          # δ at 0
    base = 0
    for hit in spec:
        hlo = min(a for (_w, a, _b) in hit)
        hhi = max(b for (_w, _a, b) in hit)
        hp = np.zeros(hhi - hlo + 1)
        for (w, a, b) in hit:
            hp[a - hlo:b - hlo + 1] += w / (b - a + 1)   # 離散一様 1/N を端点込みで
        dist = np.convolve(dist, hp)
        base += hlo
    values = np.arange(base, base + len(dist))
    return values, dist


def mc_discrete_samples(spec: list[DiscreteHit], n: int, seed: int) -> np.ndarray:
    """MC: **整数一様分布** で和 S_n をサンプリング (rng.integers, 端点込み)。"""
    rng = np.random.default_rng(seed)
    total = np.zeros(n, dtype=np.int64)
    for hit in spec:
        ws = np.array([w for (w, _a, _b) in hit], dtype=float)
        ws /= ws.sum()
        idx = rng.choice(len(hit), size=n, p=ws)
        for j, (_w, a, b) in enumerate(hit):
            m = idx == j
            cnt = int(m.sum())
            if cnt:
                total[m] += rng.integers(a, b + 1, size=cnt)   # {a,…,b} 一様
    return total


# =============================================================================
# CDF 評価ヘルパー
# =============================================================================

def discrete_cdf(values: np.ndarray, pmf: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """厳密離散の階段 CDF P(S_n ≤ x) を xs 上に。"""
    cum = np.cumsum(pmf)
    idx = np.searchsorted(values, xs, side="right") - 1
    return np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)


def mc_cdf(samples_sorted: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """MC 経験 CDF P(S_n ≤ x)。"""
    return np.searchsorted(samples_sorted, xs, side="right") / len(samples_sorted)


# =============================================================================
# 1 シナリオの計算 + プロット
# =============================================================================

def run(name: str, spec: list[DiscreteHit], out_path: str,
        n_mc: int, seed: int) -> None:
    print(f"\n=== {name} ===")
    sc_narrow = Scenario(f"{name} 素", narrow_mixtures(spec))
    sc_wide = Scenario(f"{name} +1/2", widened_mixtures(spec))

    # 厳密離散 (真値)
    values, pmf = exact_discrete_pmf(spec)
    mean_d = float(np.sum(values * pmf))
    var_d = float(np.sum((values - mean_d) ** 2 * pmf))
    std_d = np.sqrt(var_d)

    # 代表成分で離散 vs 連続の分散公式を確認
    _w, a, b = spec[0][0]
    Nc = b - a + 1
    print(f"  Hit数={len(spec)}  平均={mean_d:,.1f}  σ(厳密離散)={std_d:,.3f}")
    print(f"  代表成分 U_disc{{{a},…,{b}}}  N={Nc}")
    print(f"    σ²: 厳密離散(N²−1)/12={ (Nc**2-1)/12:.4f}  "
          f"素(N−1)²/12={ (Nc-1)**2/12:.4f}  +1/2 N²/12={ Nc**2/12:.4f}")
    print(f"    素の過小={ (Nc-1)/6:.4f}(=O(N))  +1/2 の過大=1/12={1/12:.4f}  "
          f"相対改善率 2/(N+1)={2/(Nc+1):.3%}")
    print(f"  和の σ²: 厳密離散={var_d:,.2f}  素={sc_narrow.var:,.2f}  "
          f"+1/2={sc_wide.var:,.2f}")
    print(f"    平均一致: |素−厳密|={abs(sc_narrow.mean-mean_d):.2e}  "
          f"|+1/2−厳密|={abs(sc_wide.mean-mean_d):.2e}")

    # x グリッド (厳密離散のサポート内)
    lo, hi = float(values[0]), float(values[-1])
    pad = 1e-6 * (hi - lo)
    xs = np.linspace(lo + pad, hi - pad, N_GRID)
    z = (xs - mean_d) / std_d

    # 各手法の CDF → 両側裾 min(F, 1−F)。
    #   連続COS は **連続性補正** として半整数点で評価する (docs/discrete.md):
    #   離散の階段 P(S≤m) は連続 F(m+½) に最もよく対応するので、xs+½ で評価して
    #   先頭の半ステップずれ (O(½)) を除き、残差を「幅の効果」だけにする。
    cdf_ex = discrete_cdf(values, pmf, xs)
    cdf_nw = cos_cdf(sc_narrow.grouped, xs + 0.5, sc_narrow.cos_a, sc_narrow.cos_b,
                     sc_narrow.cos_n, hybrid=True)
    cdf_wd = cos_cdf(sc_wide.grouped, xs + 0.5, sc_wide.cos_a, sc_wide.cos_b,
                     sc_wide.cos_n, hybrid=True)
    print(f"  MC (整数一様) {n_mc:,} 件 ...")
    mc = mc_discrete_samples(spec, n_mc, seed)
    mc.sort()
    print(f"    MC 平均={mc.mean():,.1f} (厳密 {mean_d:,.1f})  "
          f"MC σ={mc.std():,.3f} (厳密 {std_d:,.3f})")
    cdf_mc = mc_cdf(mc, xs)

    tail_ex = np.minimum(cdf_ex, 1.0 - cdf_ex)
    tail_nw = np.minimum(cdf_nw, 1.0 - cdf_nw)
    tail_wd = np.minimum(cdf_wd, 1.0 - cdf_wd)
    tail_mc = np.minimum(cdf_mc, 1.0 - cdf_mc)

    # 厳密離散基準の相対誤差
    EPS = 1e-300
    rel_nw = (tail_nw - tail_ex) / (tail_ex + EPS)
    rel_wd = (tail_wd - tail_ex) / (tail_ex + EPS)

    EX_C, NW_C, WD_C, MC_C = "0.7", "tab:red", "tab:blue", "0.35"
    fig, (ax_t, ax_e) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )
    fig.suptitle(
        f"{name}: 厳密離散 vs 連続COS(素 / +1/2 広げ)  — MC は整数一様分布\n"
        f"(Hit数={len(spec)}, 平均は3者一致・違いは分散のみ)",
        fontsize=12,
    )

    ax_t.plot(z, np.where(tail_ex > 0, tail_ex, np.nan), color=EX_C, lw=5.0,
              solid_capstyle="round", label="厳密離散 (真値)", zorder=1)
    ax_t.plot(z, np.where(tail_mc > 0, tail_mc, np.nan), color=MC_C, lw=0.0,
              marker=".", ms=2.5, alpha=0.5, label="MC (整数一様)", zorder=2)
    ax_t.plot(z, np.where(tail_nw > 0, tail_nw, np.nan), color=NW_C, lw=1.6,
              label="連続COS 素 U(a,b)", zorder=3)
    ax_t.plot(z, np.where(tail_wd > 0, tail_wd, np.nan), color=WD_C, lw=1.3,
              ls="--", label="連続COS +1/2 U(a−½,b+½)", zorder=4)
    ax_t.set_yscale("log")
    ax_t.set_ylim(1e-8, 1.0)
    ax_t.set_ylabel("両側裾確率 min(F, 1−F)")
    ax_t.set_title("裾確率 (厳密離散が真値)")
    ax_t.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(loc="lower center", fontsize=9, ncol=2)

    ax_e.plot(z, rel_nw, color=NW_C, lw=1.5, label="素 − 厳密離散")
    ax_e.plot(z, rel_wd, color=WD_C, lw=1.5, ls="--", label="+1/2 − 厳密離散")
    ax_e.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_e.set_ylim(-0.6, 0.6)
    ax_e.set_ylabel("裾の相対誤差\n(手法 − 厳密離散) / 厳密離散")
    ax_e.set_xlabel("z = (x − 平均) / σ")
    ax_e.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="upper center", fontsize=9, ncol=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")

    # 数値サマリ: 中腹 (分散効果が支配) と深裾 (格子粒度の床) を分けて見る
    def _maxrel(rel, lo_z, hi_z):
        m = (np.abs(z) > lo_z) & (np.abs(z) < hi_z) & (tail_ex > 1e-9)
        return np.max(np.abs(rel[m])) if m.any() else float("nan")
    for label, lo_z, hi_z in [("中腹 1<|z|<2", 1.0, 2.0), ("深裾 2.5<|z|<4", 2.5, 4.0)]:
        nw, wd = _maxrel(rel_nw, lo_z, hi_z), _maxrel(rel_wd, lo_z, hi_z)
        print(f"  最大相対誤差 {label}:  素={nw:.2e}  +1/2={wd:.2e}  "
              f"→ 改善 ×{nw/max(wd,1e-300):.0f}")


def main() -> None:
    for name, spec in SCENARIOS:
        slug = "coarse" if "粗い" in name else "fine"
        run(name, spec, os.path.join(OUT_DIR, f"discrete_cos_{slug}.png"),
            N_MC, SEED)


if __name__ == "__main__":
    main()
