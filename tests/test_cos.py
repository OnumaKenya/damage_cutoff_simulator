"""app/backend/cos.py (COS 法 + DP の参照実装) の単体テスト。

JS 移植 (assets/cos.js) の正解基準。和モデル・積モデル両方で、COS 反転が
解析モーメント・MC と一致し、CDF が台の両端で 0/1 に収束することを検証する。
"""
import math

import numpy as np

from app.backend import cos


# =============================================================================
# 和モデル (HP非依存)
# =============================================================================

def _sum_cards():
    """会心/非会心 + 一部 miss を含む 2 カード (合計 18 Hit)。"""
    return [
        {"crit_min": 100000, "crit_max": 120000, "normal_min": 50000,
         "normal_max": 60000, "hits": 10, "crit_rate": 60, "evade_rate": 0},
        {"crit_min": 80000, "crit_max": 80000, "normal_min": 40000,
         "normal_max": 45000, "hits": 8, "crit_rate": 50, "evade_rate": 10},
    ]


def test_sum_cdf_converges_at_support_ends():
    mixes = cos.build_hit_mixtures(_sum_cards(), 60, 0, "pre_decay")
    dist = cos.build_sum_dist(mixes)
    lo = np.array([dist.support_lo - 1.0])
    hi = np.array([dist.support_hi + 1.0])
    assert dist.cdf(lo)[0] == 0.0
    assert abs(dist.cdf(hi)[0] - 1.0) < 1e-6


def test_sum_mean_matches_analytic_and_mc():
    cards = _sum_cards()
    mixes = cos.build_hit_mixtures(cards, 60, 0, "pre_decay")
    dist = cos.build_sum_dist(mixes)

    # 解析平均 = Σ hit 平均
    analytic_mean = dist.mean

    # COS 平均 = ∫ x f(x) dx (台上の細グリッドで台形積分)
    grid = np.linspace(dist.support_lo, dist.support_hi, 20000)
    pdf = dist.pdf(grid)
    cos_mean = float(np.trapezoid(grid * pdf, grid))
    assert abs(cos_mean - analytic_mean) / analytic_mean < 1e-3

    # MC 平均
    rng = np.random.default_rng(12345)
    mc = cos.mc_sum(mixes, 2_000_000, rng)
    assert abs(float(mc.mean()) - analytic_mean) / analytic_mean < 5e-3


def test_sum_cdf_matches_mc_quantiles():
    mixes = cos.build_hit_mixtures(_sum_cards(), 60, 0, "pre_decay")
    dist = cos.build_sum_dist(mixes)
    rng = np.random.default_rng(999)
    mc = np.sort(cos.mc_sum(mixes, 2_000_000, rng))
    for p in (0.1, 0.5, 0.9, 0.99):
        x = float(np.quantile(mc, p))
        f_cos = float(dist.cdf(np.array([x]))[0])
        assert abs(f_cos - p) < 0.01, f"p={p}: COS CDF={f_cos}"


def test_sum_atom_separation_mass():
    """miss を含む場合、純原子部 (全 Hit miss = 値 0) の質量 = Π evade_i。"""
    # 全 Hit が evade=20% を持つ単一カード 3 Hit → 原子は 値0、質量 0.2^3
    cards = [{"crit_min": 100, "crit_max": 200, "normal_min": 50,
              "normal_max": 60, "hits": 3, "crit_rate": 50, "evade_rate": 20}]
    mixes = cos.build_hit_mixtures(cards, 50, 20, "pre_decay")
    av, ap = cos.atom_part_distribution(mixes)
    assert av.size == 1
    assert abs(av[0]) < 1e-9          # 値 0 (全 miss)
    assert abs(ap[0] - 0.2 ** 3) < 1e-9


# =============================================================================
# 安定値 (ダメージ上限張り付き時の最大ダメージ逆算)
# =============================================================================

def test_stability_min_ratio():
    # x=0 → 1 - 1/1 + 0.2 = 0.2
    assert abs(cos.stability_min_ratio(0) - 0.2) < 1e-12
    # x=2000 → 1 - 1/3 + 0.2
    assert abs(cos.stability_min_ratio(2000) - (1 - 1 / 3 + 0.2)) < 1e-12


def test_raw_bounds_uses_stability_when_capped():
    cap = cos.DAMAGE_CAP
    raw_lo_expected = cos.raw_damage_bounds(5_000_000, 6_000_000, None, "post_decay")[0]

    # 最大がキャップ & 安定値あり → 最小から逆算
    lo, hi = cos.raw_damage_bounds(5_000_000, cap, 2000, "post_decay")
    assert abs(lo - raw_lo_expected) < 1e-6
    assert abs(hi - raw_lo_expected / cos.stability_min_ratio(2000)) < 1e-3
    # 逆算した最大 > 最小 (安定値2000 → 比率<1)
    assert hi > lo


def test_raw_bounds_normal_paths():
    cap = cos.DAMAGE_CAP
    # キャップ未満 → 通常の逆変換 (安定値は無視)
    lo, hi = cos.raw_damage_bounds(5_000_000, 6_000_000, 2000, "post_decay")
    lo2, hi2 = cos.raw_damage_bounds(5_000_000, 6_000_000, None, "post_decay")
    assert abs(hi - hi2) < 1e-9
    # キャップだが安定値なし → 従来どおり逆変換 (キャップに張り付く)
    lo3, hi3 = cos.raw_damage_bounds(5_000_000, cap, None, "post_decay")
    _, hi_cap = cos.raw_damage_bounds(5_000_000, cap, None, "post_decay")
    assert hi3 == hi_cap
    # 減衰考慮前モードは恒等
    lo4, hi4 = cos.raw_damage_bounds(8000, 12000, 2000, "pre_decay")
    assert (lo4, hi4) == (8000, 12000)


# =============================================================================
# 積モデル (HP依存, ミカ型)
# =============================================================================

def _mika_base():
    """ミカ設定の 1Hit 基礎ダメージ混合 (非会心 U(8k,12k) / 会心 U(16k,24k), 会心率0.5)。"""
    return [cos.Uniform(0.5, 16000.0, 24000.0), cos.Uniform(0.5, 8000.0, 12000.0)]


def _mika_ymix(n_hits=20):
    hp = cos.HPParams(H=1_000_000.0, H1=1_000_000.0, R0=1.0, R1=2.0)
    base = _mika_base()
    ymix = cos.y_mixture(base, hp.beta)
    return [ymix] * n_hits, hp


def test_product_exact_mean_is_521727():
    """docs/product.md の厳密値 E[D] ≈ 521,727。"""
    ymix_per_hit, hp = _mika_ymix()
    mean, var = cos.damage_moments(ymix_per_hit, hp.Htil)
    assert abs(mean - 521_727) < 50, f"E[D]={mean}"
    assert var > 0


def test_product_cos_mean_matches_exact():
    """COS 再構成の D 平均が厳密値と一致 (台形積分の格子誤差以内)。"""
    ymix_per_hit, hp = _mika_ymix()
    dist = cos.build_product_dist(ymix_per_hit, hp)
    mean_exact, _ = cos.damage_moments(ymix_per_hit, hp.Htil)
    grid = np.linspace(0.0, dist.d_max, 40000)
    pdf = dist.pdf(grid)
    cos_mean = float(np.trapezoid(grid * pdf, grid))
    assert abs(cos_mean - mean_exact) / mean_exact < 2e-3


def test_product_cdf_full_mass_at_dmax():
    ymix_per_hit, hp = _mika_ymix()
    dist = cos.build_product_dist(ymix_per_hit, hp)
    f_at_max = float(dist.cdf(np.array([dist.d_max]))[0])
    assert abs(f_at_max - 1.0) < 1e-4
    assert dist.cdf(np.array([0.0]))[0] < 1e-6


def test_product_cos_matches_mc():
    ymix_per_hit, hp = _mika_ymix()
    base_per_hit = [_mika_base() for _ in range(20)]
    dist = cos.build_product_dist(ymix_per_hit, hp)
    rng = np.random.default_rng(2026)
    mc = np.sort(cos.mc_product(base_per_hit, hp, 1_000_000, rng))
    for p in (0.1, 0.5, 0.9):
        x = float(np.quantile(mc, p))
        f_cos = float(dist.cdf(np.array([x]))[0])
        assert abs(f_cos - p) < 0.015, f"p={p}: COS CDF={f_cos}"


def test_product_from_cards_matches_direct():
    """カード経由の構築が、基礎混合直接構築と一致する。"""
    hp = cos.HPParams(H=1_000_000.0, H1=1_000_000.0, R0=1.0, R1=2.0)
    cards = [{"crit_min": 16000, "crit_max": 24000, "normal_min": 8000,
              "normal_max": 12000, "hits": 20, "crit_rate": 50, "evade_rate": 0}]
    dist = cos.build_product_from_cards(cards, 50, 0, "pre_decay", hp)
    ymix_per_hit, _ = _mika_ymix()
    direct = cos.build_product_dist(ymix_per_hit, hp)
    xs = np.linspace(0.0, direct.d_max * 0.99, 50)
    assert np.allclose(dist.cdf(xs), direct.cdf(xs), atol=1e-6)
