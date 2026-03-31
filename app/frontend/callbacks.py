import json

import dash
from dash import callback, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate

from app.frontend.layout import make_damage_card, make_cutoff_card


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _order_from_children(children: list) -> list:
    """children の並び順から sorted-indices 用の順序リストを構築する。"""
    order = []
    for c in children:
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
# カード追加・削除 (コンポーネント生成が必要なためサーバーサイド)
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
        src_params = {}
        for val, pid in zip(param_values, param_ids):
            if pid["index"] == src_idx:
                src_params[pid["param"]] = val
        src_memo = ""
        for val, mid in zip(memo_values, memo_ids):
            if mid["index"] == src_idx:
                src_memo = val or ""
                break
        new_card = make_damage_card(next_idx, params=src_params, memo=src_memo)
        children.append(new_card)
        indices.append(next_idx)
        return children, indices, next_idx + 1, cutoff_indices, cutoff_next_idx, _order_from_children(children)

    raise PreventUpdate
