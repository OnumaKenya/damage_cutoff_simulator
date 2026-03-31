from bisect import bisect_left

import numpy as np
import plotly.graph_objects as go

N_SAMPLES = 200_000
N_CUTOFF_SAMPLES = 200_000

# ((x_min, x_max), (a, b)) -> y = a * x + b if x_min <= x < x_max
DAMAGE_FUNC = [
    ((0, 4000000), (1.0, 0)),
    ((4000000, 6248000), (0.8, 4000000 - 0.8 * 4000000)),
    ((6248000, 8496000), (0.65, 5798400 - 0.65 * 6248000)),
    ((8496000, 10744000), (0.5, 7259600 - 0.5 * 8496000)),
    ((10744000, 12992000), (0.4, 8383600 - 0.4 * 10744000)),
    ((12992000, 15240000), (0.3, 9282800 - 0.3 * 12992000)),
    ((15240000, 17488000), (0.225, 9957200 - 0.225 * 15240000)),
    ((17488000, 19736000), (0.15, 10463000 - 0.15 * 17488000)),
    ((19736000, 22000000), (0.075, 10800200 - 0.075 * 19736000)),
    ((22000000, 10**20), (0.0, 10966999)),
]

# 逆変換テーブル: 減衰後の境界値 y = a * x_min + b を事前計算
_INVERSE_TABLE: list[tuple[tuple[float, float], float, float]] = []
for (x_lo, x_hi), (a, b) in DAMAGE_FUNC:
    y_lo = a * x_lo + b
    y_hi = a * x_hi + b
    _INVERSE_TABLE.append(((y_lo, y_hi), a, b))


def decay(x: np.ndarray) -> np.ndarray:
    """減衰関数: 生ダメージ x → 減衰後ダメージ y"""
    y = np.empty_like(x)
    for (x_lo, x_hi), (a, b) in DAMAGE_FUNC:
        mask = (x >= x_lo) & (x < x_hi)
        y[mask] = a * x[mask] + b
    return y


def inverse_decay(y: float) -> float:
    """逆変換: 減衰後ダメージ y → 生ダメージ x (スカラー)"""
    for (y_lo, y_hi), a, b in _INVERSE_TABLE:
        lo, hi = min(y_lo, y_hi), max(y_lo, y_hi)
        if lo <= y <= hi:
            if a == 0.0:
                return DAMAGE_FUNC[-1][0][0]  # 上限キャップ
            return (y - b) / a
    # テーブル範囲外は恒等
    return y


def _extract_hit_params(
    indices: list[int],
    params: dict[int, dict],
    global_crit: float,
    global_evade: float,
    damage_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """カードリストからヒットごとのパラメータを展開し、6本の配列として返す。"""
    crit_lows: list[float] = []
    crit_highs: list[float] = []
    normal_lows: list[float] = []
    normal_highs: list[float] = []
    crit_rates: list[float] = []
    evade_rates: list[float] = []

    for idx in indices:
        p = params.get(idx)
        if p is None:
            continue

        crit_min = float(p.get("crit_min") or 0)
        crit_max = float(p.get("crit_max") or 0)
        normal_min = float(p.get("normal_min") or 0)
        normal_max = float(p.get("normal_max") or 0)
        hits = int(p.get("hits") or 1)
        cr = p.get("crit_rate")
        er = p.get("evade_rate")
        cr = float(cr if cr is not None else global_crit or 0) / 100.0
        er = float(er if er is not None else global_evade or 0) / 100.0

        if damage_mode == "post_decay":
            raw_crit_lo = inverse_decay(crit_min)
            raw_crit_hi = inverse_decay(crit_max)
            raw_norm_lo = inverse_decay(normal_min)
            raw_norm_hi = inverse_decay(normal_max)
        else:
            raw_crit_lo, raw_crit_hi = crit_min, crit_max
            raw_norm_lo, raw_norm_hi = normal_min, normal_max

        raw_crit_hi = max(raw_crit_hi, raw_crit_lo)
        raw_norm_hi = max(raw_norm_hi, raw_norm_lo)

        for _ in range(hits):
            crit_lows.append(raw_crit_lo)
            crit_highs.append(raw_crit_hi)
            normal_lows.append(raw_norm_lo)
            normal_highs.append(raw_norm_hi)
            crit_rates.append(cr)
            evade_rates.append(er)

    return (
        np.asarray(crit_lows),
        np.asarray(crit_highs),
        np.asarray(normal_lows),
        np.asarray(normal_highs),
        np.asarray(crit_rates),
        np.asarray(evade_rates),
    )


def _simulate_vectorized(
    rng: np.random.Generator,
    crit_lows: np.ndarray,
    crit_highs: np.ndarray,
    normal_lows: np.ndarray,
    normal_highs: np.ndarray,
    crit_rates: np.ndarray,
    evade_rates: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """全ヒットをまとめてベクトル演算でシミュレーションし、合計ダメージを返す。

    Python ループを排除し、(n_hits, n_samples) の 2D 配列で一括処理する。
    """
    n_hits = len(crit_lows)
    if n_hits == 0:
        return np.zeros(n_samples)

    # (n_hits, n_samples) の乱数を一括生成
    hit_mask = rng.random((n_hits, n_samples)) >= evade_rates[:, None]
    is_crit = rng.random((n_hits, n_samples)) < crit_rates[:, None]

    # ヒットごとに異なる範囲の一様乱数を生成
    u_crit = rng.random((n_hits, n_samples))
    u_norm = rng.random((n_hits, n_samples))

    crit_raw = u_crit * (crit_highs - crit_lows)[:, None] + crit_lows[:, None]
    norm_raw = u_norm * (normal_highs - normal_lows)[:, None] + normal_lows[:, None]

    raw_samples = np.where(is_crit, crit_raw, norm_raw)

    # 減衰関数を一括適用 (1D に展開して処理し、元の形に戻す)
    dmg = decay(raw_samples.ravel()).reshape(n_hits, n_samples)

    return np.sum(dmg * hit_mask, axis=0)


def run_simulation(
    indices: list[int],
    params: dict[int, dict],
    global_crit: float,
    global_evade: float,
    target_damage: float,
    damage_mode: str = "post_decay",
) -> tuple[go.Figure, str]:
    """モンテカルロ法でダメージ分布をシミュレーションし、Figureと通過率テキストを返す。"""
    rng = np.random.default_rng()

    hit_params = _extract_hit_params(indices, params, global_crit, global_evade, damage_mode)
    total_damage = _simulate_vectorized(rng, *hit_params, N_SAMPLES)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=total_damage, nbinsx=200, name="ダメージ分布"))
    fig.update_layout(
        title="合計ダメージ分布",
        xaxis_title="合計ダメージ",
        yaxis_title="頻度",
        bargap=0.05,
    )

    mean_val = np.mean(total_damage)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"期待値: {mean_val:,.0f}")

    target = float(target_damage or 0)
    pass_rate = float(np.mean(total_damage >= target) * 100) if target > 0 else None
    pass_text = ""
    if target > 0:
        fig.add_vline(x=target, line_dash="solid", line_color="green", annotation_text=f"目標: {target:,.0f}")
        pass_text = f"目標ダメージ {target:,.0f} の通過確率: {pass_rate:.2f}%"

    return fig, pass_text


def _simulate_cards(
    indices: list[int],
    params: dict[int, dict],
    global_crit: float,
    global_evade: float,
    damage_mode: str,
    n_samples: int = N_CUTOFF_SAMPLES,
) -> np.ndarray:
    """指定カード群の合計ダメージをシミュレーションし、ソート済み配列を返す。"""
    rng = np.random.default_rng()

    hit_params = _extract_hit_params(indices, params, global_crit, global_evade, damage_mode)
    total = _simulate_vectorized(rng, *hit_params, n_samples)

    total.sort()
    return total


def _build_lookup_table(sorted_samples: np.ndarray, n_points: int = 2000) -> dict:
    """ソート済みサンプルから補間用ルックアップテーブルを構築する。"""
    n = len(sorted_samples)
    if n == 0:
        return {"values": [], "min": 0.0, "max": 0.0}
    step = max(1, n // n_points)
    values = sorted_samples[::step].tolist()
    return {
        "values": values,
        "min": float(sorted_samples[0]),
        "max": float(sorted_samples[-1]),
    }


def exceedance_prob(table: dict, threshold: float) -> float:
    """ルックアップテーブルから P(X >= threshold) を % で返す。"""
    values = table.get("values", [])
    if not values:
        return 0.0
    idx = bisect_left(values, threshold)
    cdf = idx / len(values)
    return round((1 - cdf) * 100, 2)


def value_at_exceedance(table: dict, exceedance_pct: float) -> float:
    """ルックアップテーブルから指定超過確率に対応するダメージ値を返す。"""
    values = table.get("values", [])
    if not values:
        return 0.0
    cdf = 1 - exceedance_pct / 100
    idx = int(cdf * (len(values) - 1))
    idx = max(0, min(idx, len(values) - 1))
    return round(values[idx], 0)


def compute_cutoff(
    order: list,
    params: dict[int, dict],
    global_crit: float,
    global_evade: float,
    target_damage: float,
    damage_mode: str,
) -> dict:
    """足切り位置で上側・下側の分布を計算し、初期値 (超過確率50%) で返す。"""
    # order 内の "cutoff_0" で上下に分割
    cutoff_pos = None
    for i, x in enumerate(order):
        if isinstance(x, str) and x.startswith("cutoff"):
            cutoff_pos = i
            break
    if cutoff_pos is None:
        return {}

    upper_indices = [x for x in order[:cutoff_pos] if isinstance(x, int)]
    lower_indices = [x for x in order[cutoff_pos + 1 :] if isinstance(x, int)]

    upper_table = _build_lookup_table(_simulate_cards(
        upper_indices, params, global_crit, global_evade, damage_mode
    ))
    lower_table = _build_lookup_table(_simulate_cards(
        lower_indices, params, global_crit, global_evade, damage_mode
    ))

    target = float(target_damage or 0)

    # 初期値: 超過確率 50%
    e2 = 50.0
    e1 = value_at_exceedance(upper_table, e2)
    e3 = max(target - e1, 0)
    e4 = exceedance_prob(lower_table, e3)

    return {
        "upper_table": upper_table,
        "lower_table": lower_table,
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
    }
