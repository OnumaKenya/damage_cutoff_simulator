"""係数空間 (COS/F&O) 多段リスタ最適化 restart_cos の検証。

正解基準:
  - _corr_valid は np.correlate(..., 'valid') と一致。
  - 前向き通過率は、関門が単一・序盤0のとき厳密な周辺超過確率 1-F_S(d) と一致。
  - 関門・通過率・スループットは グリッド版 restart.py と整合(粗グリッド差を除く)。
  - 積モデル(HP依存)も同様。
"""
import numpy as np

from app.backend.cos import (
    HPParams,
    build_hit_mixtures,
    build_sum_dist,
    y_mixture,
)
from app.backend import restart, restart_cos


def _card(cmin, cmax, nmin, nmax, hits, cr, ev=0):
    return {"crit_min": cmin, "crit_max": cmax, "normal_min": nmin,
            "normal_max": nmax, "hits": hits, "crit_rate": cr,
            "evade_rate": ev, "enemies": 1}


# 実ユーザー入力相当 (8カード/60ヒット, post_decay)
_CARDS = [
    _card(15235, 19572, 4659, 5985, 4, 52.37),
    _card(786905, 1010907, 142967, 183664, 8, 57.96),
    _card(209841, 269575, 38125, 48977, 16, 57.96),
    _card(241318, 310011, 43843, 56324, 2, 57.96),
    _card(89183, 114569, 16203, 20815, 4, 57.96),
    _card(786905, 1010907, 142967, 183664, 8, 57.96),
    _card(209841, 269575, 38125, 48977, 16, 57.96),
    _card(241318, 310011, 43843, 56324, 2, 57.96),
]
_TIMES_CARD = [1, 1, 0, 20, 1, 1, 1, 5]
_CPS = [4, 12, 28, 30, 34, 42, 58]
_D = 19_400_000


def _hits_and_times():
    hits = build_hit_mixtures(_CARDS, 57.96, 0.0, "post_decay")
    ht = []
    for c, w in zip(_CARDS, _TIMES_CARD):
        h = int(c["hits"])
        ht += [w / h] * h
    return hits, ht


def test_corr_valid_matches_numpy():
    rng = np.random.default_rng(0)
    for nK, nx in [(9, 5), (31, 16), (64, 33), (129, 64)]:
        K = rng.standard_normal(nK)
        x = rng.standard_normal(nx)
        assert np.allclose(restart_cos._corr_valid(K, x),
                           np.correlate(K, x, "valid"), atol=1e-9)


def test_forward_passrate_matches_exact_marginal():
    """序盤の関門が0なら cp での通過率 = 厳密な周辺超過確率 1-F_{S_m}(d)。"""
    hits, ht = _hits_and_times()
    bounds = [0, *_CPS, len(hits)]
    times = [float(sum(ht[bounds[i]:bounds[i + 1]])) for i in range(len(bounds) - 1)]
    eng = restart_cos._CosEngine(restart_cos._segs_sum(hits, bounds))
    # cp4, cp12 を0, cp28 に関門。cp28 の通過率は P(S_28 >= d28) (joint=marginal)。
    d28 = 9_048_232
    gates = {1: 0.0, 2: 0.0, 3: float(d28), 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
    fwd = restart_cos.forward_metrics(eng, times, _D, gates)
    exact = 1.0 - float(build_sum_dist(hits[:28]).cdf(np.array([d28]))[0])
    assert abs(fwd["pass_rates"][2] - exact) < 2e-3      # cp28
    assert fwd["pass_rates"][0] == 1.0 or abs(fwd["pass_rates"][0] - 1.0) < 1e-3


def test_forward_passrate_le_100pct():
    """通過率・成功率は [0,1]。grid 版で 100% を超えていた入力で確認。"""
    hits, ht = _hits_and_times()
    rc = restart_cos.analyze(hits, _CPS, ht, _D)
    for r in rc["rows"]:
        assert 0.0 <= r["pass_rate"] <= 1.0
    assert 0.0 <= rc["success"] <= 1.0


def test_sum_model_matches_grid():
    """和モデル: 係数空間版とグリッド版で 関門・通過率・スループットが整合。"""
    hits, ht = _hits_and_times()
    rg = restart.analyze(hits, _CPS, ht, _D)
    rc = restart_cos.analyze(hits, _CPS, ht, _D)
    # スループット・成功率 (粗グリッド差を許容)
    assert abs(rc["throughput"] - rg["throughput"]) / rg["throughput"] < 0.02
    assert abs(rc["success"] - rg["success"]) < 5e-3
    assert abs(rc["g_star_dp"] - rg["g_star_dp"]) / rg["g_star_dp"] < 0.02
    # 拘束的な関門 (cp28 以降) はダメージ値が一致 (相対 1%)
    gg = {r["checkpoint"]: r["gate"] for r in rg["rows"]}
    gc = {r["checkpoint"]: r["gate"] for r in rc["rows"]}
    for cp in (28, 42, 58):
        assert abs(gc[cp] - gg[cp]) / gg[cp] < 0.01


def test_product_model_matches_grid():
    """積モデル(HP依存): 係数空間版とグリッド版で整合。"""
    cards = [
        _card(20000, 30000, 10000, 15000, 4, 40),
        _card(25000, 35000, 12000, 18000, 6, 50),
        _card(22000, 32000, 11000, 16000, 4, 50),
    ]
    hits = build_hit_mixtures(cards, 50.0, 0.0, "pre_decay")
    n = len(hits)
    hp = HPParams(H=100_000_000, H1=100_000_000, R0=1000, R1=2000)
    ymix = [y_mixture(m, hp.beta) for m in hits]
    cps = [4, 10]
    ht = [1.0, 1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0]
    D = 193_000_000
    rg = restart.analyze_product(ymix, hp, cps, ht, D)
    rc = restart_cos.analyze_product(ymix, hp, cps, ht, D)
    assert abs(rc["throughput"] - rg["throughput"]) / rg["throughput"] < 0.03
    assert abs(rc["success"] - rg["success"]) < 5e-3
    for r in rc["rows"]:
        assert 0.0 <= r["pass_rate"] <= 1.0


def test_forward_matches_montecarlo():
    """前向き通過率・成功率を素朴な MC と照合 (確定シード)。"""
    hits, ht = _hits_and_times()
    bounds = [0, *_CPS, len(hits)]
    times = [float(sum(ht[bounds[i]:bounds[i + 1]])) for i in range(len(bounds) - 1)]
    rg = restart.analyze(hits, _CPS, ht, _D)
    gates = {k + 1: r["gate"] for k, r in enumerate(rg["rows"])}

    eng = restart_cos._CosEngine(restart_cos._segs_sum(hits, bounds))
    fc = restart_cos.forward_metrics(eng, times, _D, gates)

    rng = np.random.default_rng(7)
    M = 300_000
    los = [np.array([u.lo for u in mix]) for mix in hits]
    his = [np.array([u.hi for u in mix]) for mix in hits]
    ws = [np.array([u.weight for u in mix]) / sum(u.weight for u in mix) for mix in hits]
    cum = np.zeros(M)
    alive = np.ones(M, bool)
    K = len(_CPS)
    pr = []
    for j in range(len(bounds) - 1):
        for h in range(bounds[j], bounds[j + 1]):
            comp = rng.choice(len(ws[h]), size=M, p=ws[h])
            x = los[h][comp] + rng.random(M) * (his[h][comp] - los[h][comp])
            cum = cum + np.where(alive, x, 0.0)
        if j < K:
            alive = alive & (cum >= gates[j + 1])
            pr.append(alive.mean())
    succ = (alive & (cum >= _D)).mean()

    for k in range(K):
        assert abs(fc["pass_rates"][k] - pr[k]) < 5e-3
    assert abs(fc["success"] - succ) < 5e-3
