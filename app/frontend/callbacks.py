import json
import time

import dash
from dash import callback, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate

from app.backend.simulation import (
    run_simulation,
    _simulate_cards,
    _build_lookup_table,
    exceedance_prob,
    value_at_exceedance,
)
from app.frontend.layout import make_damage_card, make_cutoff_card, log_slider_to_pct, pct_to_log_slider


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _build_params(values: list, ids: list) -> dict[int, dict]:
    params: dict[int, dict] = {}
    for val, id_dict in zip(values, ids):
        idx = id_dict["index"]
        param = id_dict["param"]
        if idx not in params:
            params[idx] = {}
        params[idx][param] = val
    return params


def _split_order_at_cutoff(order: list, cutoff_index: int) -> tuple[list[int], list[int]]:
    marker = f"cutoff_{cutoff_index}"
    pos = order.index(marker) if marker in order else len(order)
    upper = [x for x in order[:pos] if isinstance(x, int)]
    lower = [x for x in order[pos + 1 :] if isinstance(x, int)]
    return upper, lower


def _order_from_children(children: list) -> list:
    """children の並び順から sorted-indices 用の順序リストを構築する。"""
    order = []
    for c in children:
        # Dash コンポーネントの場合は属性アクセス、辞書の場合はキーアクセス
        if isinstance(c, dict):
            cid = c.get("props", {}).get("id", {})
        else:
            cid = getattr(c, "id", {})
        if not isinstance(cid, dict):
            continue
        if cid.get("type") == "card":
            order.append(cid["index"])
        elif cid.get("type") == "cutoff":
            order.append(f"cutoff_{cid['index']}")
    return order


# ---------------------------------------------------------------------------
# ドラッグ順同期
# ---------------------------------------------------------------------------
@callback(
    Output("sorted-indices", "data"),
    Input("drag-order", "data"),
    State("card-indices", "data"),
    prevent_initial_call=True,
)
def sync_drag_order(drag_order, indices):
    if not drag_order:
        return indices
    try:
        return json.loads(drag_order)
    except (json.JSONDecodeError, TypeError):
        return indices


# ---------------------------------------------------------------------------
# カード追加・削除
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children"),
    Output("card-indices", "data"),
    Output("next-index", "data"),
    Output("cutoff-indices", "data"),
    Output("cutoff-next-index", "data"),
    Output("sorted-indices", "data", allow_duplicate=True),
    Input("add-btn", "n_clicks"),
    Input("add-cutoff-btn", "n_clicks"),
    Input({"type": "remove-btn", "index": ALL}, "n_clicks"),
    Input({"type": "cutoff-remove", "index": ALL}, "n_clicks"),
    Input({"type": "duplicate-btn", "index": ALL}, "n_clicks"),
    State("card-indices", "data"),
    State("next-index", "data"),
    State("cards-container", "children"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("cutoff-indices", "data"),
    State("cutoff-next-index", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_cards(
    add_clicks, add_cutoff_clicks, remove_clicks, cutoff_remove_clicks, duplicate_clicks,
    indices, next_idx, children, global_crit, global_evade,
    cutoff_indices, cutoff_next_idx,
    param_values, param_ids, memo_values, memo_ids,
):
    trigger = ctx.triggered_id

    if trigger == "add-btn":
        indices.append(next_idx)
        children.append(make_damage_card(next_idx, global_crit, global_evade))
        return children, indices, next_idx + 1, cutoff_indices, cutoff_next_idx, _order_from_children(children)

    if trigger == "add-cutoff-btn":
        children.append(make_cutoff_card(cutoff_next_idx))
        cutoff_indices.append(cutoff_next_idx)
        return children, indices, next_idx, cutoff_indices, cutoff_next_idx + 1, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "remove-btn":
        remove_idx = trigger["index"]
        if len(indices) <= 1:
            return children, indices, next_idx, cutoff_indices, cutoff_next_idx, dash.no_update
        indices = [i for i in indices if i != remove_idx]
        children = [
            c for c in children
            if not (c["props"]["id"].get("type") == "card" and c["props"]["id"].get("index") == remove_idx)
        ]
        return children, indices, next_idx, cutoff_indices, cutoff_next_idx, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "cutoff-remove":
        if not any(n for n in cutoff_remove_clicks if n):
            raise PreventUpdate
        remove_idx = trigger["index"]
        cutoff_indices = [i for i in cutoff_indices if i != remove_idx]
        children = [
            c for c in children
            if not (c["props"]["id"].get("type") == "cutoff" and c["props"]["id"].get("index") == remove_idx)
        ]
        return children, indices, next_idx, cutoff_indices, cutoff_next_idx, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "duplicate-btn":
        src_idx = trigger["index"]
        # 元カードのパラメータを収集
        src_params = {}
        for val, pid in zip(param_values, param_ids):
            if pid["index"] == src_idx:
                src_params[pid["param"]] = val
        # 元カードのメモを取得
        src_memo = ""
        for val, mid in zip(memo_values, memo_ids):
            if mid["index"] == src_idx:
                src_memo = val or ""
                break
        # 一番下に追加
        new_card = make_damage_card(next_idx, params=src_params, memo=src_memo)
        children.append(new_card)
        indices.append(next_idx)
        return children, indices, next_idx + 1, cutoff_indices, cutoff_next_idx, _order_from_children(children)

    raise PreventUpdate


# ---------------------------------------------------------------------------
# 世代カウンタ: ダメージ分布に影響する変更で世代を進める
# ---------------------------------------------------------------------------
@callback(
    Output("cutoff-generation", "data"),
    Input("card-indices", "data"),
    Input("sorted-indices", "data"),
    Input({"type": "param", "param": ALL, "index": ALL}, "value"),
    Input("global-crit-rate", "value"),
    Input("global-evade-rate", "value"),
    Input("damage-mode", "value"),
    State("cutoff-generation", "data"),
    prevent_initial_call=True,
)
def increment_generation(_ci, _si, _pv, _gc, _ge, _dm, current_gen):
    return current_gen + 1


# ---------------------------------------------------------------------------
# 一括適用
# ---------------------------------------------------------------------------
@callback(
    Output({"type": "param", "param": "crit_rate", "index": ALL}, "value"),
    Output({"type": "param", "param": "evade_rate", "index": ALL}, "value"),
    Input("apply-global-btn", "n_clicks"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State({"type": "param", "param": "crit_rate", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def apply_global(n_clicks, global_crit, global_evade, current_crit_values):
    n = len(current_crit_values)
    return [global_crit] * n, [global_evade] * n


# ---------------------------------------------------------------------------
# シミュレーション実行
# ---------------------------------------------------------------------------
@callback(
    Output("result-graph", "figure"),
    Output("pass-rate-text", "children"),
    Input("run-btn", "n_clicks"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("target-damage", "value"),
    State("damage-mode", "value"),
    prevent_initial_call=True,
)
def run_simulation_callback(
    n_clicks, values, ids, sorted_indices, card_indices,
    global_crit, global_evade, target_damage, damage_mode,
):
    indices = sorted_indices if sorted_indices else card_indices
    indices = [x for x in indices if isinstance(x, int)]
    params = _build_params(values, ids)
    return run_simulation(indices, params, global_crit, global_evade, target_damage, damage_mode)


# ---------------------------------------------------------------------------
# 足切り計算ボタン: フラグ(世代)が変わっていれば MC 再計算、変わっていなければスキップ
# ---------------------------------------------------------------------------
@callback(
    Output("cutoff-dist-store", "data"),
    Output("cutoff-values-store", "data"),
    Output({"type": "cutoff-status", "index": ALL}, "children"),
    Input({"type": "cutoff-compute", "index": ALL}, "n_clicks"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("target-damage", "value"),
    State("damage-mode", "value"),
    State("cutoff-dist-store", "data"),
    State("cutoff-values-store", "data"),
    State("cutoff-generation", "data"),
    State({"type": "cutoff-status", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def compute_cutoff_button(
    n_clicks_list, sorted_indices, card_indices, values, ids,
    global_crit, global_evade, target_damage, damage_mode,
    prev_dist, prev_values, generation, status_ids,
):
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict) or not any(n for n in n_clicks_list if n):
        raise PreventUpdate

    cutoff_index = trigger["index"]
    key = str(cutoff_index)
    order = sorted_indices if sorted_indices else card_indices

    dist_store = dict(prev_dist or {})
    values_store = dict(prev_values or {})

    cached = dist_store.get(key)
    need_recompute = not cached or cached.get("generation") != generation

    if need_recompute:
        # フラグON → MC 再計算
        params = _build_params(values, ids)
        upper_indices, lower_indices = _split_order_at_cutoff(order, cutoff_index)
        upper_table = _build_lookup_table(
            _simulate_cards(upper_indices, params, global_crit, global_evade, damage_mode)
        )
        lower_table = _build_lookup_table(
            _simulate_cards(lower_indices, params, global_crit, global_evade, damage_mode)
        )
        dist_store[key] = {
            "upper_table": upper_table,
            "lower_table": lower_table,
            "generation": generation,
        }
        status_action = "再計算完了"
    else:
        # フラグOFF → MC スキップ、キャッシュ利用
        upper_table = cached["upper_table"]
        lower_table = cached["lower_table"]
        status_action = "キャッシュ利用"

    target = float(target_damage or 0)
    e2 = 50.0
    e1 = value_at_exceedance(upper_table, e2)
    e3 = max(target - e1, 0)
    e4 = exceedance_prob(lower_table, e3)

    values_store[key] = {"e1": round(e1, 0), "e2": round(e2, 2), "e3": round(e3, 0), "e4": round(e4, 2)}

    upper_range = f'{upper_table["min"]:,.0f} ~ {upper_table["max"]:,.0f}'
    lower_range = f'{lower_table["min"]:,.0f} ~ {lower_table["max"]:,.0f}'
    status_msg = f"{status_action} | 上側: {upper_range} | 下側: {lower_range}"
    status_texts = [status_msg if sid["index"] == cutoff_index else dash.no_update for sid in status_ids]

    return dist_store, values_store, status_texts


# ---------------------------------------------------------------------------
# 足切りスライダー操作 → 軽量計算のみ（MC なし、キャッシュ必須）
# ---------------------------------------------------------------------------
@callback(
    Output("cutoff-values-store", "data", allow_duplicate=True),
    Input({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "value"),
    State("cutoff-values-store", "data"),
    State("cutoff-dist-store", "data"),
    State("target-damage", "value"),
    State({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def on_cutoff_slider_change(slider_vals, current_values, dist, target_damage, slider_ids):
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict):
        raise PreventUpdate

    cutoff_key = str(trigger["index"])
    elem = trigger["elem"]

    # キャッシュが無ければスライダーは無効（先に「足切り計算」を押す必要あり）
    if not dist or cutoff_key not in dist:
        raise PreventUpdate

    idx_in_list = next(
        (i for i, sid in enumerate(slider_ids) if sid["elem"] == elem and sid["index"] == trigger["index"]),
        None,
    )
    if idx_in_list is None:
        raise PreventUpdate
    val = slider_vals[idx_in_list]
    if val is None:
        raise PreventUpdate
    val = float(val)

    # パーセントスライダーは対数内部値 → 実パーセントへ変換
    is_pct = elem in ("e2", "e4")
    if is_pct:
        pct_val = log_slider_to_pct(val)
    else:
        pct_val = val

    # エコーバック防止（パーセント系は対数空間で比較）
    if current_values:
        card_values = current_values.get(cutoff_key)
        if card_values:
            current_stored = float(card_values.get(elem, float("inf")))
            if is_pct:
                current_log = pct_to_log_slider(current_stored)
                if abs(val - current_log) < 1.5:
                    raise PreventUpdate
            else:
                if abs(val - current_stored) < 0.5:
                    raise PreventUpdate

    target = float(target_damage or 0)
    upper = dist[cutoff_key]["upper_table"]
    lower = dist[cutoff_key]["lower_table"]

    # 軽量なルックアップ計算のみ
    if elem == "e1":
        e1 = pct_val
        e2 = exceedance_prob(upper, e1)
        e3 = max(target - e1, 0)
        e4 = exceedance_prob(lower, e3)
    elif elem == "e2":
        e2 = pct_val
        e1 = value_at_exceedance(upper, e2)
        e3 = max(target - e1, 0)
        e4 = exceedance_prob(lower, e3)
    elif elem == "e3":
        e3 = pct_val
        e1 = max(target - e3, 0)
        e2 = exceedance_prob(upper, e1)
        e4 = exceedance_prob(lower, e3)
    elif elem == "e4":
        e4 = pct_val
        e3 = value_at_exceedance(lower, e4)
        e1 = max(target - e3, 0)
        e2 = exceedance_prob(upper, e1)
    else:
        raise PreventUpdate

    new_values = dict(current_values or {})
    new_values[cutoff_key] = {"e1": round(e1, 0), "e2": round(e2, 2), "e3": round(e3, 0), "e4": round(e4, 2)}
    return new_values


# ---------------------------------------------------------------------------
# values Store → スライダー表示更新
# ---------------------------------------------------------------------------
@callback(
    Output({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "value"),
    Output({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "min"),
    Output({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "max"),
    Input("cutoff-values-store", "data"),
    State("cutoff-dist-store", "data"),
    State({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_cutoff_display(values, dist, slider_ids):
    if not values or not dist:
        raise PreventUpdate

    s_vals, s_mins, s_maxs = [], [], []
    for sid in slider_ids:
        key = str(sid["index"])
        elem = sid["elem"]
        card_vals = values.get(key)
        card_dist = dist.get(key)

        if card_vals:
            raw = card_vals.get(elem, 0)
            # パーセント系は実パーセント → 対数スライダー値に変換
            if elem in ("e2", "e4"):
                s_vals.append(pct_to_log_slider(float(raw)))
            else:
                s_vals.append(raw)
        else:
            s_vals.append(dash.no_update)

        if card_dist:
            if elem == "e1":
                s_mins.append(card_dist["upper_table"].get("min", 0))
                s_maxs.append(card_dist["upper_table"].get("max", 10_000_000))
            elif elem == "e3":
                s_mins.append(card_dist["lower_table"].get("min", 0))
                s_maxs.append(card_dist["lower_table"].get("max", 10_000_000))
            else:
                s_mins.append(dash.no_update)
                s_maxs.append(dash.no_update)
        else:
            s_mins.append(dash.no_update)
            s_maxs.append(dash.no_update)

    return s_vals, s_mins, s_maxs


# ---------------------------------------------------------------------------
# % Input 直接入力 → Store 更新
# ---------------------------------------------------------------------------
@callback(
    Output("cutoff-values-store", "data", allow_duplicate=True),
    Input({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "value"),
    State("cutoff-values-store", "data"),
    State("cutoff-dist-store", "data"),
    State("target-damage", "value"),
    State({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def on_cutoff_pct_input_change(input_vals, current_values, dist, target_damage, input_ids):
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict):
        raise PreventUpdate

    cutoff_key = str(trigger["index"])
    elem = trigger["elem"]

    if not dist or cutoff_key not in dist:
        raise PreventUpdate

    idx_in_list = next(
        (i for i, sid in enumerate(input_ids) if sid["elem"] == elem and sid["index"] == trigger["index"]),
        None,
    )
    if idx_in_list is None:
        raise PreventUpdate
    val = input_vals[idx_in_list]
    if val is None:
        raise PreventUpdate
    val = max(0.01, min(100, float(val)))

    # エコーバック防止
    if current_values:
        card_values = current_values.get(cutoff_key)
        if card_values:
            current_stored = float(card_values.get(elem, float("inf")))
            if abs(val - current_stored) < 0.005:
                raise PreventUpdate

    target = float(target_damage or 0)
    upper = dist[cutoff_key]["upper_table"]
    lower = dist[cutoff_key]["lower_table"]

    if elem == "e2":
        e2 = val
        e1 = value_at_exceedance(upper, e2)
        e3 = max(target - e1, 0)
        e4 = exceedance_prob(lower, e3)
    elif elem == "e4":
        e4 = val
        e3 = value_at_exceedance(lower, e4)
        e1 = max(target - e3, 0)
        e2 = exceedance_prob(upper, e1)
    else:
        raise PreventUpdate

    new_values = dict(current_values or {})
    new_values[cutoff_key] = {"e1": round(e1, 0), "e2": round(e2, 2), "e3": round(e3, 0), "e4": round(e4, 2)}
    return new_values


# ---------------------------------------------------------------------------
# Store → % Input 表示同期
# ---------------------------------------------------------------------------
@callback(
    Output({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "value", allow_duplicate=True),
    Input("cutoff-values-store", "data"),
    State({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_pct_input_display(values, input_ids):
    if not values:
        raise PreventUpdate

    out = []
    for sid in input_ids:
        key = str(sid["index"])
        elem = sid["elem"]
        card_vals = values.get(key)
        if card_vals:
            out.append(round(float(card_vals.get(elem, 0)), 2))
        else:
            out.append(dash.no_update)
    return out


# ---------------------------------------------------------------------------
# マニュアルモーダル開閉
# ---------------------------------------------------------------------------
@callback(
    Output("manual-modal", "style"),
    Input("open-manual-btn", "n_clicks"),
    Input("close-manual-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_manual_modal(open_clicks, close_clicks):
    if ctx.triggered_id == "open-manual-btn":
        return {"display": "flex"}
    return {"display": "none"}
