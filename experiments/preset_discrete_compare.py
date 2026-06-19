"""プリセット群 (和 + product) で「離散一様 → 連続化 (素 / +1/2 広げ)」を比較。

docs/discrete.md の解析を、合成シナリオ (discrete_cos_compare.py) ではなく **実プリセット**
で確かめる。ダメージは本来整数値なので、各一様成分を整数格子上の離散一様
U_disc{a,…,b} と理想化し (端点を丸める)、それを連続化して COS 法を回す:

    - 素の幅   U(a, b)        (幅 N−1)
    - +1/2 広げ U(a−½, b+½)   (幅 N, 連続性補正)

真値は **整数一様分布での MC** (rng.integers, 端点込み)。和は cos_compare の COS、
product (HP依存) は product_cos の COS を使う。

期待される結果 (docs/discrete.md): プリセットは全て **BAスケール** で 1Hit の点数
N ~ 10³–10⁵ と巨大なので、離散性の効果も +1/2 補正の恩恵も ~2/(N+1) で機械精度級に
小さい。素・+1/2・整数MC は全て重なり、素↔+1/2 の決定論的な差は MC ノイズ以下になる
— 合成の「粗い格子」で +1/2 が効いたのと対照的で、効果が 1/N で消えることを示す。

    uv run python -m experiments.preset_discrete_compare
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.cos_compare import (  # noqa: E402
    Uniform, Scenario, cos_cdf, group_hits,
    toy_scenario, real_scenario,
)
from experiments.edgeworth_animation import build_all_hits  # noqa: E402
from experiments.cos_app import _BLUEARCHIVE_7HIT_CARDS  # noqa: E402
from experiments.discrete_cos_compare import (  # noqa: E402
    narrow_mixtures, widened_mixtures, mc_discrete_samples, mc_cdf,
)
from experiments.product_cos import (  # noqa: E402
    HPScenario, base_mixture, y_mixture, damage_dist, damage_moments,
)

OUT_DIR = "experiments/output"
N_GRID = 600
N_MC = 10_000_000
SEED = 20260603


# =============================================================================
# 整数格子への理想化: 各成分の端点を丸めて (weight, a, b) の整数スペックに
# =============================================================================

def int_spec_from_mixtures(hit_mixtures: list[list[Uniform]]) -> list[list]:
    """post-decay 混合の各成分 U(lo,hi) を離散一様 U_disc{round(lo),…,round(hi)} に。"""
    spec: list[list] = []
    for mix in hit_mixtures:
        hit = []
        for u in mix:
            a, b = int(round(u.lo)), int(round(u.hi))
            if b < a:
                a, b = b, a
            hit.append((u.weight, a, b))
        spec.append(hit)
    return spec


def _two_sided(cdf: np.ndarray) -> np.ndarray:
    return np.minimum(cdf, 1.0 - cdf)


# =============================================================================
# 和プリセット: 整数MC (真値) vs 連続COS (素 / +1/2)
# =============================================================================

def compare_sum(name: str, hit_mixtures: list[list[Uniform]], out_path: str,
                n_mc: int, seed: int) -> dict:
    spec = int_spec_from_mixtures(hit_mixtures)
    sc_nw = Scenario(f"{name} 素", narrow_mixtures(spec))
    sc_wd = Scenario(f"{name} +1/2", widened_mixtures(spec))

    # 代表 1Hit の点数 N (最も広い連続成分)
    Ns = [b - a + 1 for hit in spec for (w, a, b) in hit if b > a]
    N_rep = int(np.median(Ns)) if Ns else 1

    lo, hi = sc_wd.support_lo, sc_wd.support_hi
    pad = 1e-6 * (hi - lo)
    xs = np.linspace(lo + pad, hi - pad, N_GRID)
    z = (xs - sc_nw.mean) / sc_nw.std

    # 連続COS は連続性補正で半整数点 xs+½ で評価 (docs/discrete.md)
    cdf_nw = cos_cdf(sc_nw.grouped, xs + 0.5, sc_nw.cos_a, sc_nw.cos_b, sc_nw.cos_n)
    cdf_wd = cos_cdf(sc_wd.grouped, xs + 0.5, sc_wd.cos_a, sc_wd.cos_b, sc_wd.cos_n)

    mc = mc_discrete_samples(spec, n_mc, seed)
    mc.sort()
    cdf_mc = mc_cdf(mc, xs)

    t_nw, t_wd, t_mc = _two_sided(cdf_nw), _two_sided(cdf_wd), _two_sided(cdf_mc)

    # 決定論的な離散性シグナル: 素↔+1/2 の差 (MC ノイズに依らない)
    d_nw_wd = float(np.max(np.abs(cdf_nw - cdf_wd)))
    rel_var = (sc_nw.var - sc_wd.var) / sc_wd.var
    # 中腹で MC と各 COS の相対差 (MC ノイズ込み)
    mid = (np.abs(z) > 1.0) & (np.abs(z) < 2.5) & (t_mc > 1e-5)
    def _relmc(t):
        return float(np.max(np.abs((t[mid] - t_mc[mid]) / t_mc[mid]))) if mid.any() else float("nan")

    row = dict(name=name, kind="和", n_hits=len(hit_mixtures), N_rep=N_rep,
               mean_eq=abs(sc_nw.mean - sc_wd.mean), rel_var=rel_var,
               d_nw_wd=d_nw_wd, relmc_nw=_relmc(t_nw), relmc_wd=_relmc(t_wd))
    _plot(name, z, t_mc, t_nw, t_wd, cdf_nw, cdf_wd, out_path, row, kind="和")
    return row


# =============================================================================
# product プリセット: 整数MC (真値) vs product-COS (素 / +1/2)
# =============================================================================

def mc_damage_hits_int(base_per_hit: list[list[Uniform]],
                       H: float, H1: float, R0: float, R1: float,
                       n: int, rng: np.random.Generator) -> np.ndarray:
    """基礎ダメージ x を **整数一様** で引いて HP依存漸化式を回し D を返す。"""
    beta = (R1 - R0) / H
    Hn = np.full(n, H1, dtype=float)
    for base in base_per_hit:
        ws = np.array([c.weight for c in base], dtype=float)
        ws /= ws.sum()
        comp = rng.choice(len(base), size=n, p=ws)
        x = np.empty(n)
        for j, c in enumerate(base):
            m = comp == j
            cnt = int(m.sum())
            if cnt == 0:
                continue
            a, b = int(round(c.lo)), int(round(c.hi))
            x[m] = rng.integers(a, b + 1, size=cnt) if b > a else a
        Hn = Hn - (beta * Hn + R0) * x
    return H1 - Hn


def _base_narrow_widened(base: list[Uniform]) -> tuple[list[Uniform], list[Uniform]]:
    """基礎ダメージ x 混合を、整数化した素 / +1/2 広げの 2 版にする (atom は据置)。"""
    nw, wd = [], []
    for u in base:
        a, b = int(round(u.lo)), int(round(u.hi))
        if b < a:
            a, b = b, a
        nw.append(Uniform(u.weight, float(a), float(b)))
        wd.append(Uniform(u.weight, float(a), float(b)) if a == b
                  else Uniform(u.weight, a - 0.5, b + 0.5))
    return nw, wd


def compare_product(sc: HPScenario, out_path: str, n_mc: int, seed: int) -> dict:
    beta = sc.beta
    nw_base, wd_base = _base_narrow_widened(sc.base_mixture)
    ymix_nw = [y_mixture(nw_base, beta)] * sc.n_hits
    ymix_wd = [y_mixture(wd_base, beta)] * sc.n_hits

    Ns = [int(round(u.hi)) - int(round(u.lo)) + 1 for u in sc.base_mixture if u.hi > u.lo]
    N_rep = int(np.median(Ns)) if Ns else 1

    # 厳密モーメント (基準) と D グリッド
    mean_D, var_D = damage_moments(ymix_wd, sc.Htil)
    std_D = math.sqrt(max(var_D, 0.0))

    rng = np.random.default_rng(seed)
    mc = mc_damage_hits_int([nw_base] * sc.n_hits, sc.H, sc.H1, sc.R0, sc.R1, n_mc, rng)
    mc.sort()
    d_lo, d_hi = float(mc[0]), float(mc[-1])
    pad = 0.01 * (d_hi - d_lo)
    xs = np.linspace(max(0.0, d_lo - pad), d_hi + pad, N_GRID)
    z = (xs - mean_D) / std_D

    _f_nw, F_nw = damage_dist(ymix_nw, sc.Htil, xs)
    _f_wd, F_wd = damage_dist(ymix_wd, sc.Htil, xs)
    cdf_mc = np.searchsorted(mc, xs, side="right") / n_mc

    t_nw, t_wd, t_mc = _two_sided(F_nw), _two_sided(F_wd), _two_sided(cdf_mc)
    d_nw_wd = float(np.max(np.abs(F_nw - F_wd)))
    mean_nw, var_nw = damage_moments(ymix_nw, sc.Htil)
    rel_var = (var_nw - var_D) / var_D
    mid = (np.abs(z) > 1.0) & (np.abs(z) < 2.5) & (t_mc > 1e-5)
    def _relmc(t):
        return float(np.max(np.abs((t[mid] - t_mc[mid]) / t_mc[mid]))) if mid.any() else float("nan")

    row = dict(name=sc.name, kind="product", n_hits=sc.n_hits, N_rep=N_rep,
               mean_eq=abs(mean_nw - mean_D), rel_var=rel_var,
               d_nw_wd=d_nw_wd, relmc_nw=_relmc(t_nw), relmc_wd=_relmc(t_wd))
    _plot(sc.name, z, t_mc, t_nw, t_wd, F_nw, F_wd, out_path, row, kind="product")
    return row


# =============================================================================
# 共通プロット & サマリ
# =============================================================================

def _plot(name, z, t_mc, t_nw, t_wd, cdf_nw, cdf_wd, out_path, row, kind):
    fig, (ax_t, ax_d) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]})
    fig.suptitle(
        f"{name} [{kind}]: 整数MC vs 連続COS(素 / +1/2 広げ)\n"
        f"Hit数={row['n_hits']}, 代表 N≈{row['N_rep']:,}, "
        f"相対σ²差(素−+½)={row['rel_var']:+.2e}, max|F素−F+½|={row['d_nw_wd']:.2e}",
        fontsize=11)

    ax_t.plot(z, np.where(t_mc > 0, t_mc, np.nan), color="0.35", lw=0.0,
              marker=".", ms=2.5, alpha=0.5, label="整数MC (真値)")
    ax_t.plot(z, np.where(t_nw > 0, t_nw, np.nan), color="tab:red", lw=1.6,
              label="連続COS 素 U(a,b)")
    ax_t.plot(z, np.where(t_wd > 0, t_wd, np.nan), color="tab:blue", lw=1.2,
              ls="--", label="連続COS +1/2 U(a−½,b+½)")
    ax_t.set_yscale("log")
    ax_t.set_ylim(1e-7, 1.0)
    ax_t.set_ylabel("両側裾確率 min(F, 1−F)")
    ax_t.set_title("裾確率 (整数MC が真値)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(loc="lower center", fontsize=9, ncol=3)

    ax_d.plot(z, cdf_nw - cdf_wd, color="purple", lw=1.4)
    ax_d.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax_d.set_ylabel("F素 − F+½\n(決定論的な離散性シグナル)")
    ax_d.set_xlabel("z = (x − 平均) / σ")
    ax_d.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def main() -> None:
    rows = []

    # --- 和プリセット ---
    sum_presets = [
        ("歪んだ少数Hit (8発)", toy_scenario().hit_mixtures, "discrete_preset_toy.png"),
        ("メイドアリスTL (7発カード)",
         build_all_hits(_BLUEARCHIVE_7HIT_CARDS, 65.27, 0.0, "post_decay")[0],
         "discrete_preset_maidalice.png"),
        ("実データ (132 Hit)", real_scenario().hit_mixtures, "discrete_preset_real.png"),
    ]
    for name, hm, fn in sum_presets:
        print(f"\n=== {name} (和) ===")
        rows.append(compare_sum(name, hm, os.path.join(OUT_DIR, fn), N_MC, SEED))

    # --- product プリセット (ミカ) ---
    print("\n=== ミカ (R1=2, R0=1) [product] ===")
    mika = HPScenario(
        name="ミカ (R1=2, R0=1)", H=1_000_000.0, H1=1_000_000.0,
        R0=1.0, R1=2.0, n_hits=20,
        base_mixture=base_mixture(8_000.0, 12_000.0, 16_000.0, 24_000.0,
                                  crit_rate=0.5, evade_rate=0.0),
    )
    rows.append(compare_product(mika, os.path.join(OUT_DIR, "discrete_preset_mika.png"),
                                2_000_000, SEED))

    # --- サマリ表 ---
    print("\n" + "=" * 92)
    print("サマリ: 整数離散の連続化 (素 vs +1/2 広げ)。BAスケールでは N 大 → 効果 ~2/(N+1) で微小")
    print("=" * 92)
    print(f"{'プリセット':<24}{'種別':<8}{'Hit':>5}{'代表N':>9}{'2/(N+1)':>10}"
          f"{'相対σ²差':>11}{'max|F素−F+½|':>13}{'素 vs MC':>10}{'+½ vs MC':>10}")
    for r in rows:
        print(f"{r['name']:<24}{r['kind']:<8}{r['n_hits']:>5}{r['N_rep']:>9,}"
              f"{2/(r['N_rep']+1):>10.2e}{r['rel_var']:>+11.2e}{r['d_nw_wd']:>13.2e}"
              f"{r['relmc_nw']:>10.1%}{r['relmc_wd']:>10.1%}")
    print("\n相対σ²差・max|F素−F+½| が 2/(N+1) 級に小さく、素/+½ とも整数MC と中腹で誤差数%以内")
    print("(= MC 標本ノイズ水準) なら、BAスケールで離散性も +1/2 補正も無視できることの確認。")


if __name__ == "__main__":
    main()
