import base64
import json

import dash
import plotly.graph_objects as go
from dash import callback, Input, Output, State, ALL, ctx, dcc, html
from dash.exceptions import PreventUpdate

from app.backend import ocr, restart_cos
from app.backend.cos import HPParams, build_hit_mixtures, y_mixture
from app.frontend.layout import make_damage_card

# エクスポート/インポートで扱うカードパラメータ項目とフォーマット版。
_CARD_PARAMS = ["crit_min", "crit_max", "normal_min", "normal_max",
                "hits", "crit_rate", "evade_rate", "enemies"]
_IO_VERSION = 2


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _triggered_clicked() -> bool:
    """今回のトリガー当人の値が真か (= 実際にクリックされたか) を返す。

    動的にカードが追加されると n_clicks=0 の新ボタンが ALL 入力に加わり
    本コールバックが再発火する。n_clicks は累積保持されるため集計では
    判定できず、トリガーされた当人の値 (ctx.triggered[0]) を見る。
    """
    trig = ctx.triggered
    return bool(trig and trig[0].get("value"))


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
    return order


# ---------------------------------------------------------------------------
# カード追加・削除 (コンポーネント生成が必要なためサーバーサイド)
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children"),
    Output("card-indices", "data"),
    Output("next-index", "data"),
    Output("sorted-indices", "data", allow_duplicate=True),
    Input("add-btn", "n_clicks"),
    Input({"type": "remove-btn", "index": ALL}, "n_clicks"),
    Input({"type": "duplicate-btn", "index": ALL}, "n_clicks"),
    State("card-indices", "data"),
    State("next-index", "data"),
    State("cards-container", "children"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_cards(
    add_clicks, remove_clicks, duplicate_clicks,
    indices, next_idx, children, global_crit, global_evade,
    param_values, param_ids, memo_values, memo_ids,
):
    trigger = ctx.triggered_id

    if trigger == "add-btn":
        indices.append(next_idx)
        children.append(make_damage_card(next_idx, global_crit, global_evade))
        return children, indices, next_idx + 1, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "remove-btn":
        # カード動的追加でボタンが増えると本コールバックが再発火するため、
        # 実際にクリックされた (トリガー当人の n_clicks が真) 時のみ処理する。
        # n_clicks は累積保持されるので集計ではなくトリガー値を見る。
        if not _triggered_clicked():
            raise PreventUpdate
        remove_idx = trigger["index"]
        if len(indices) <= 1:
            return children, indices, next_idx, dash.no_update
        indices = [i for i in indices if i != remove_idx]
        children = [
            c for c in children
            if not (c["props"]["id"].get("type") == "card" and c["props"]["id"].get("index") == remove_idx)
        ]
        return children, indices, next_idx, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "duplicate-btn":
        # 同上: 実際に複製ボタンが押された時のみ処理する。
        if not _triggered_clicked():
            raise PreventUpdate
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
        return children, indices, next_idx + 1, _order_from_children(children)

    raise PreventUpdate


# ---------------------------------------------------------------------------
# スクリーンショット OCR → カード自動生成 (サーバーサイド)
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children", allow_duplicate=True),
    Output("card-indices", "data", allow_duplicate=True),
    Output("next-index", "data", allow_duplicate=True),
    Output("sorted-indices", "data", allow_duplicate=True),
    Output("ocr-status", "children"),
    Output("hp-mode", "value", allow_duplicate=True),
    Input("ocr-upload", "contents"),
    Input("ocr-image-store", "data"),
    State("cards-container", "children"),
    State("card-indices", "data"),
    State("next-index", "data"),
    prevent_initial_call=True,
)
def ocr_add_cards(upload_contents, snip_data, children, indices, next_idx):
    """アップロード / スニップ画像を OCR し、抽出カードを追加する。"""
    trigger = ctx.triggered_id
    image = upload_contents if trigger == "ocr-upload" else snip_data
    if not image:
        raise PreventUpdate

    no_change = (dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    try:
        result = ocr.cards_from_image(image)
    except ocr.OcrError as exc:
        return (*no_change, f"⚠ {exc}", dash.no_update)
    except Exception as exc:  # noqa: BLE001 - 予期せぬ失敗もユーザーに表示
        return (*no_change, f"⚠ 解析に失敗しました: {exc}", dash.no_update)

    parsed = result["cards"]
    if not parsed:
        return (*no_change, "⚠ カードを検出できませんでした。画像を確認してください。", dash.no_update)

    for card in parsed:
        children.append(make_damage_card(next_idx, params=card["params"], memo=card["memo"]))
        indices.append(next_idx)
        next_idx += 1

    msgs = [f"✅ {len(parsed)} 枚のカードを追加しました。"]
    hp_mode = dash.no_update
    if result.get("hp_dependent"):
        hp_mode = "on"
        msgs.append("HP依存を検出 → サイドバーのHP依存モードをONにしました。")

    return (
        children,
        indices,
        next_idx,
        _order_from_children(children),
        " ".join(msgs),
        hp_mode,
    )


# ---------------------------------------------------------------------------
# 多段リスタ最適化 (サーバーサイド: COS + Bermudan 後ろ向き帰納)
# ---------------------------------------------------------------------------
def _assemble_cards_ordered(order, card_indices, param_values, param_ids):
    """param の State から、表示順の (カード index, カード dict) 列を組み立てる。"""
    by_index: dict = {}
    for val, pid in zip(param_values, param_ids):
        by_index.setdefault(pid["index"], {})[pid["param"]] = val
    seq = [i for i in (order or []) if isinstance(i, int)] or list(card_indices or [])
    return [(i, by_index[i]) for i in seq if i in by_index]


# ---------------------------------------------------------------------------
# 入力情報のエクスポート (カード + 全体設定 + 多段リスタ設定 → JSON ダウンロード)
# ---------------------------------------------------------------------------
@callback(
    Output("export-download", "data"),
    Input("export-btn", "n_clicks"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    State("restart-cp-store", "data"),
    State("restart-seg-time-store", "data"),
    State("restart-seg-success-store", "data"),
    State("target-damage", "value"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("global-stability", "value"),
    State("calc-method", "value"),
    State("damage-mode", "value"),
    State("hp-mode", "value"),
    State("hp-H", "value"), State("hp-H1", "value"),
    State("hp-R0", "value"), State("hp-R1", "value"),
    State("restart-D", "value"),
    prevent_initial_call=True,
)
def export_input(n_clicks, order, card_indices, param_values, param_ids,
                 memo_values, memo_ids, cp_store, seg_times, seg_success,
                 target_damage, gcrit, gevade, gstab, calc_method, damage_mode,
                 hp_mode, hp_H, hp_H1, hp_R0, hp_R1, restart_D):
    if not n_clicks:
        raise PreventUpdate

    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    memo_by = {mid["index"]: (v or "") for v, mid in zip(memo_values, memo_ids)}
    n = sum(int(params.get("hits") or 1) for _idx, params in ordered)
    cps = sorted({int(c) for c in (cp_store or []) if 0 < int(c) < n})
    # 区間開始境界 (0, cps...) の時間割合だけを書き出す
    seg_times = seg_times or {}
    seg_success = seg_success or {}
    boundaries = [0, *cps]
    segment_times = {str(b): float(seg_times.get(str(b), 1.0)) for b in boundaries}
    segment_success = {str(b): float(seg_success.get(str(b), 100.0)) for b in boundaries}

    cards = []
    for idx, params in ordered:
        cards.append({
            "params": {k: params.get(k) for k in _CARD_PARAMS},
            "memo": memo_by.get(idx, ""),
        })
    data = {
        "version": _IO_VERSION,
        "globals": {
            "target_damage": target_damage,
            "global_crit": gcrit, "global_evade": gevade, "global_stability": gstab,
            "calc_method": calc_method, "damage_mode": damage_mode, "hp_mode": hp_mode,
            "hp_H": hp_H, "hp_H1": hp_H1, "hp_R0": hp_R0, "hp_R1": hp_R1,
            "restart_D": restart_D,
        },
        "cards": cards,
        "restart": {"checkpoints": cps, "segment_times": segment_times,
                    "segment_success": segment_success},
    }
    return dict(content=json.dumps(data, ensure_ascii=False, indent=2),
                filename="damage_cutoff_input.json")


# ---------------------------------------------------------------------------
# 入力情報のインポート (JSON → カード再構築 + 全体設定 + 多段リスタ設定復元)
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children", allow_duplicate=True),
    Output("card-indices", "data", allow_duplicate=True),
    Output("next-index", "data", allow_duplicate=True),
    Output("sorted-indices", "data", allow_duplicate=True),
    Output("restart-cp-store", "data", allow_duplicate=True),
    Output("restart-seg-time-store", "data", allow_duplicate=True),
    Output("restart-seg-success-store", "data", allow_duplicate=True),
    Output("io-status", "children"),
    Output("target-damage", "value"),
    Output("global-crit-rate", "value"),
    Output("global-evade-rate", "value"),
    Output("global-stability", "value"),
    Output("calc-method", "value"),
    Output("damage-mode", "value"),
    Output("hp-mode", "value", allow_duplicate=True),
    Output("hp-H", "value"), Output("hp-H1", "value"),
    Output("hp-R0", "value"), Output("hp-R1", "value"),
    Output("restart-D", "value"),
    Input("import-upload", "contents"),
    prevent_initial_call=True,
)
def import_input(contents):
    if not contents:
        raise PreventUpdate
    nu = dash.no_update
    # globals 出力 13 個 (target..restart_D) の「変更なし」ベクトル
    globals_nu = (nu,) * 12

    try:
        _meta, b64 = contents.split(",", 1)
        data = json.loads(base64.b64decode(b64).decode("utf-8"))
        cards = data.get("cards", [])
        if not isinstance(cards, list) or not cards:
            raise ValueError("カードが空です。")
    except Exception as exc:  # noqa: BLE001 - 不正ファイルはユーザーに表示
        return (nu, nu, nu, nu, nu, nu, nu, f"⚠ インポート失敗: {exc}", *globals_nu)

    children = []
    total_hits = 0
    for i, c in enumerate(cards):
        params = {k: (c.get("params", {}) or {}).get(k) for k in _CARD_PARAMS}
        children.append(make_damage_card(i, params=params, memo=c.get("memo", "")))
        total_hits += int(params.get("hits") or 1)
    n = len(cards)
    indices = list(range(n))

    # 多段リスタ設定 (新フォーマット: restart.checkpoints / restart.segment_times)
    restart = data.get("restart", {}) or {}
    cps = sorted({int(c) for c in restart.get("checkpoints", [])
                  if 0 < int(c) < total_hits})
    seg_times = {str(k): float(v)
                 for k, v in (restart.get("segment_times", {}) or {}).items()}
    seg_times.setdefault("0", 1.0)
    seg_success = {str(k): float(v)
                   for k, v in (restart.get("segment_success", {}) or {}).items()}
    seg_success.setdefault("0", 100.0)

    g = data.get("globals", {}) or {}
    def gv(key):
        return g[key] if key in g else nu
    msg = f"✅ {n} 枚のカードと設定を読み込みました。足切りライン最適化の設定も復元済みです。"
    return (
        children, indices, n, indices, cps, seg_times, seg_success, msg,
        gv("target_damage"), gv("global_crit"), gv("global_evade"), gv("global_stability"),
        gv("calc_method"), gv("damage_mode"), gv("hp_mode"),
        gv("hp_H"), gv("hp_H1"), gv("hp_R0"), gv("hp_R1"), gv("restart_D"),
    )


# ---------------------------------------------------------------------------
# 多段リスタ: カード別 チェックポイント / 時間割合 テーブル
# ---------------------------------------------------------------------------
@callback(
    Output("restart-cards-table", "children"),
    Output("restart-cp-dropdown", "options"),
    Output("restart-nhits", "data"),
    Input("nav-restart", "n_clicks"),
    Input("restart-reload-btn", "n_clicks"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def populate_restart_table(_n1, _n2, order, card_indices, param_values, param_ids,
                           memo_values, memo_ids):
    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    if not ordered:
        return html.Div("カードがありません。", style={"color": "#d63031"}), [], 0

    memo_by = {mid["index"]: (v or "") for v, mid in zip(memo_values, memo_ids)}

    header = html.Tr([html.Th("カード"), html.Th("ヒット"), html.Th("累積")])
    rows = [header]
    options = []
    cum = 0
    last = len(ordered) - 1
    for pos, (idx, card) in enumerate(ordered):
        h = int(card.get("hits") or 1)
        cum += h
        memo = memo_by.get(idx, "")
        label = f"{pos + 1}: {memo}" if memo else f"カード{pos + 1}"
        # 最終セグメント以外を足切り候補としてプルダウンに出す
        if pos != last:
            options.append({"label": f"{label}(累積 {cum} ヒット目で足切り)",
                            "value": cum})
        rows.append(html.Tr([
            html.Td(label), html.Td(str(h)), html.Td(str(cum)),
        ]))
    table = html.Table(rows, className="restart-cards")
    return table, options, cum


# ---------------------------------------------------------------------------
# 多段リスタ: チェックポイントの追加 / 削除 (プルダウン + カード方式)
# ---------------------------------------------------------------------------
@callback(
    Output("restart-cp-store", "data", allow_duplicate=True),
    Output("restart-cp-dropdown", "value"),
    Input("restart-cp-add-btn", "n_clicks"),
    Input({"type": "restart-cp-remove", "index": ALL}, "n_clicks"),
    State("restart-cp-dropdown", "value"),
    State("restart-cp-store", "data"),
    prevent_initial_call=True,
)
def manage_restart_cp(_add, _removes, dropdown_value, store):
    if not _triggered_clicked():
        raise PreventUpdate
    store = list(store or [])
    trig = ctx.triggered_id
    if trig == "restart-cp-add-btn":
        if dropdown_value is None:
            raise PreventUpdate
        cum = int(dropdown_value)
        if cum not in store:
            store.append(cum)
            store.sort()
        return store, None
    if isinstance(trig, dict) and trig.get("type") == "restart-cp-remove":
        cum = int(trig["index"])
        return [c for c in store if c != cum], dash.no_update
    raise PreventUpdate


# ---------------------------------------------------------------------------
# 多段リスタ: 区間カード (足切りで区切られた各区間 = 1 カード)
#   各カードに「時間割合」入力を内蔵し、末尾以外は「✕」でその足切りを解除できる。
# ---------------------------------------------------------------------------
def _segments(cps, n):
    """足切り cps と総ヒット n から区間 [(start, end), ...] を作る。"""
    cps = sorted({int(c) for c in (cps or []) if 0 < int(c) < int(n or 0)})
    bounds = [0, *cps, int(n or 0)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _seg_card(idx, total, s, e, weight, end_label, success=100.0):
    """区間カード 1 枚 (横長) を生成する。末尾以外は ✕ で足切り (境界 e) を解除。"""
    is_last = idx == total - 1
    head = "完走(最終区間)" if is_last else f"足切り{idx + 1}"
    title = f"区間{idx + 1}・{head}"
    sub = f"ヒット {s + 1}–{e}"
    if not is_last and end_label:
        sub += f"  /  {end_label}"
    cells = [
        # 左: 区間名 + ヒット範囲
        html.Div(
            [
                html.Div(title, style={"fontWeight": "bold", "fontSize": "0.85rem"}),
                html.Div(sub, style={"fontSize": "0.78rem", "color": "#666"}),
            ],
            style={"flex": "1", "minWidth": "0"},
        ),
        # 中: 時間割合入力
        html.Div(
            [
                html.Label("時間割合 ", style={"fontSize": "0.8rem"}),
                dcc.Input(id={"type": "restart-seg-time", "index": s}, type="number",
                          value=weight, min=0, step=0.1,
                          style={"width": "90px", "marginLeft": "4px"}),
            ],
            style={"whiteSpace": "nowrap"},
        ),
        # 中2: ダメージと独立な成功率 (%) 入力。この区間を回しきって次へ進める確率。
        # 足切り(ダメージ)とは別要因であることが分かるよう、ラベル・色・注記で区別する。
        html.Div(
            [
                html.Label(
                    "🎲 ダメージ外 成功率% ",
                    title="ダメージ(足切り)とは無関係な成功要因。この区間を回しきって"
                          "次へ進める確率です。失敗するとリスタート(その区間の時間は消費)。"
                          "100%=この要因では失敗しない(=ダメージ足切りのみ)。",
                    style={"fontSize": "0.8rem", "color": "#0984e3",
                           "fontWeight": "bold"}),
                dcc.Input(id={"type": "restart-seg-success", "index": s}, type="number",
                          value=success, min=0, max=100, step=1,
                          style={"width": "64px", "marginLeft": "4px",
                                 "border": "1px solid #0984e3", "color": "#0984e3"}),
                html.Span("%", style={"fontSize": "0.8rem", "color": "#0984e3",
                                      "marginLeft": "2px"}),
            ],
            style={"whiteSpace": "nowrap",
                   "borderLeft": "1px solid #dfe6e9", "paddingLeft": "12px"},
        ),
    ]
    # 右: 足切り解除 (末尾区間以外)
    cells.append(html.Button(
        "✕", id={"type": "restart-cp-remove", "index": e if not is_last else -1},
        n_clicks=0, title="この足切りを解除",
        style={"border": "none", "background": "transparent",
               "cursor": "pointer" if not is_last else "default",
               "color": "#d63031" if not is_last else "transparent",
               "fontWeight": "bold", "marginLeft": "12px",
               "visibility": "visible" if not is_last else "hidden"}))
    return html.Div(
        cells,
        style={"display": "flex", "alignItems": "center", "gap": "12px",
               "border": "1px solid #d63031", "borderRadius": "8px",
               "padding": "8px 14px", "marginBottom": "8px",
               "background": "#fff", "width": "100%", "boxSizing": "border-box"},
    )


@callback(
    Output("restart-cp-cards", "children"),
    Input("restart-cp-store", "data"),
    Input("restart-nhits", "data"),
    Input("restart-cp-dropdown", "options"),
    State("restart-seg-time-store", "data"),
    State("restart-seg-success-store", "data"),
    prevent_initial_call=True,
)
def render_restart_cards(cp_store, n, options, seg_times, seg_success):
    segs = _segments(cp_store, n)
    if not segs:
        return html.Div("攻撃列が未読込です。「カード読込 / 更新」を押してください。",
                        style={"fontSize": "0.82rem", "color": "#d63031"})
    label_by = {opt["value"]: opt["label"] for opt in (options or [])}
    seg_times = seg_times or {}
    seg_success = seg_success or {}
    cards = []
    for i, (s, e) in enumerate(segs):
        cards.append(_seg_card(i, len(segs), s, e,
                               seg_times.get(str(s), 1.0), label_by.get(e, ""),
                               seg_success.get(str(s), 100.0)))
    return html.Div(cards, style={"display": "flex", "flexDirection": "column"})


@callback(
    Output("restart-seg-time-store", "data", allow_duplicate=True),
    Input({"type": "restart-seg-time", "index": ALL}, "value"),
    State({"type": "restart-seg-time", "index": ALL}, "id"),
    State("restart-seg-time-store", "data"),
    prevent_initial_call=True,
)
def update_restart_seg_time(values, ids, store):
    store = dict(store or {})
    for v, sid in zip(values, ids):
        try:
            store[str(sid["index"])] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            store[str(sid["index"])] = 0.0
    return store


@callback(
    Output("restart-seg-success-store", "data", allow_duplicate=True),
    Input({"type": "restart-seg-success", "index": ALL}, "value"),
    State({"type": "restart-seg-success", "index": ALL}, "id"),
    State("restart-seg-success-store", "data"),
    prevent_initial_call=True,
)
def update_restart_seg_success(values, ids, store):
    """区間ごとのダメージ独立成功率 % (0..100) を Store へ。空欄/不正は 100% 扱い。"""
    store = dict(store or {})
    for v, sid in zip(values, ids):
        try:
            store[str(sid["index"])] = min(100.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            store[str(sid["index"])] = 100.0
    return store


# ---------------------------------------------------------------------------
# 多段リスタ: 図・表の共通ヘルパー (最適表示 / インタラクティブ表示で共用)
# ---------------------------------------------------------------------------
def _restart_disp(res, cum_to_label, last_label, D):
    """解析結果 res から (ラベル, 残りダメージ, 区間通過率, 累積通過率, 完走?) の行を作る。"""
    disp = []
    prev_cum = 1.0
    for r in res["rows"]:
        cum_pass = r["pass_rate"]
        sect = (cum_pass / prev_cum) if prev_cum > 0 else 0.0
        label = cum_to_label.get(str(r["checkpoint"]), f"{r['checkpoint']}ヒット目")
        disp.append([label, D - r["gate"], sect, cum_pass, False])
        prev_cum = cum_pass
    final_cum = res["success"]
    final_sect = (final_cum / prev_cum) if prev_cum > 0 else 0.0
    disp.append([f"{last_label}(完走/目標達成)", 0.0, final_sect, final_cum, True])
    return disp


def _cutoff_figure(disp, title, *, color="#d63031", ref_disp=None):
    """足切りライン(残りダメージ)の折れ線図。ref_disp があれば最適ラインを点線で重ねる。"""
    fig = go.Figure()
    if ref_disp is not None:
        fig.add_trace(go.Scatter(
            x=[d[0] for d in ref_disp], y=[d[1] for d in ref_disp],
            mode="lines+markers", line=dict(color="#999", dash="dot"),
            marker=dict(size=8, color="#999"), name="最適ライン",
        ))
    fig.add_trace(go.Scatter(
        x=[d[0] for d in disp], y=[d[1] for d in disp],
        mode="lines+markers+text",
        text=[f"残り{rem:,.0f}<br>区間{sect:.1%} / 累積{cumr:.1%}"
              for (_lbl, rem, sect, cumr, _f) in disp],
        textposition="top center", marker=dict(size=11, color=color),
        line=dict(color=color),
        name="設定ライン" if ref_disp is not None else "最適足切り(残りダメージ)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="black",
                  annotation_text="目標達成(残り0)")
    fig.update_layout(
        title=title, xaxis_title="チェックポイント(カード)",
        yaxis_title="足切りライン(目標までの残りダメージ)",
        height=460, margin=dict(t=60),
    )
    return fig


@callback(
    Output("restart-graph", "figure"),
    Output("restart-summary", "children"),
    Output("restart-config", "data"),
    Output("restart-gate-sliders", "children"),
    Input("restart-run-btn", "n_clicks"),
    State("restart-D", "value"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    State("restart-cp-store", "data"),
    State("restart-seg-time-store", "data"),
    State("restart-seg-success-store", "data"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("damage-mode", "value"),
    State("hp-mode", "value"),
    State("hp-H", "value"),
    State("hp-H1", "value"),
    State("hp-R0", "value"),
    State("hp-R1", "value"),
    prevent_initial_call=True,
)
def run_restart(n_clicks, D, order, card_indices, param_values, param_ids,
                memo_values, memo_ids, cp_store, seg_times, seg_success_store,
                global_crit, global_evade,
                damage_mode, hp_mode, hp_H, hp_H1, hp_R0, hp_R1):
    if not n_clicks:
        raise PreventUpdate

    empty = go.Figure()

    def err(msg):
        return (empty, html.Div(f"⚠ {msg}", style={"color": "#d63031"}),
                None, [])

    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    if not ordered:
        return err("攻撃カードがありません。「カード読込」を押してください。")

    cards = [c for _i, c in ordered]
    hits = build_hit_mixtures(cards, float(global_crit or 0),
                              float(global_evade or 0), damage_mode or "post_decay")
    n = len(hits)
    if n < 2:
        return err("総ヒット数が2以上必要です。")

    # チェックポイント = プルダウンで追加した累積ヒット数
    cps = sorted({int(x) for x in (cp_store or [])})
    cps = [c for c in cps if 0 < c < n]
    if not cps:
        return err("足切り(チェックポイント)を1つ以上追加してください。")

    # 時間割合: 区間(足切り間)ごとの相対重みを各区間内のヒットへ等分。
    # 区間は開始境界の累積ヒット数 (0, cps...) でキー付けされる。
    seg_times = seg_times or {}
    bounds = [0, *cps, n]
    hit_times = [0.0] * n
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        try:
            w = float(seg_times.get(str(s), 1.0))
        except (TypeError, ValueError):
            w = 1.0
        length = e - s
        per = (w / length) if length > 0 else 0.0
        for j in range(s, e):
            hit_times[j] = per
    if sum(hit_times) <= 0:
        hit_times = [1.0] * n          # 全部0なら一様にフォールバック

    # 区間ごとのダメージ独立成功確率: ストアは区間開始境界 → % (0..100)。
    # 区間順 (0, cps...) に並べ、フラクション [0,1] へ変換 (未設定は 1.0)。
    seg_success_store = seg_success_store or {}
    seg_success = []
    for s in bounds[:-1]:
        try:
            pct = float(seg_success_store.get(str(s), 100.0))
        except (TypeError, ValueError):
            pct = 100.0
        seg_success.append(min(1.0, max(0.0, pct / 100.0)))

    D = float(D or 0)
    if D <= 0:
        return err("目標ダメージ D を正の値で入力してください。")

    if hp_mode == "on":
        try:
            hp = HPParams(H=float(hp_H), H1=float(hp_H1),
                          R0=float(hp_R0), R1=float(hp_R1))
            if hp.H == 0 or hp.beta == 0:
                raise ValueError
        except (TypeError, ValueError):
            return err("HP依存パラメータ (H, H1, R0, R1) を正しく入力してください。")
        if D >= hp.Htil:
            return err(f"目標 D は H̃₁={hp.Htil:,.0f} 未満にしてください(到達不能)。")
        ymix = [y_mixture(m, hp.beta) for m in hits]
        res = restart_cos.analyze_product(ymix, hp, cps, hit_times, D,
                                          seg_success=seg_success)
        model_note = "積モデル(HP依存)"
    else:
        res = restart_cos.analyze(hits, cps, hit_times, D, seg_success=seg_success)
        model_note = "和モデル(HP非依存)"

    # チェックポイントのヒット数 → カード名 の対応と、最終(完走)カード名を作る。
    memo_by = {mid["index"]: (v or "") for v, mid in zip(memo_values, memo_ids)}
    cum = 0
    cum_to_label = {}        # {str(累積ヒット): カード名}
    last_label = ""
    for pos, (idx, card) in enumerate(ordered):
        cum += int(card.get("hits") or 1)
        # 位置番号を前置してカード名を一意化 (同名カードでも図/表で衝突しない)
        memo = memo_by.get(idx)
        label = f"{pos + 1}: {memo}" if memo else f"カード{pos + 1}"
        cum_to_label[str(cum)] = label
        last_label = label

    # 表示用の行 (label, remaining, section_rate, cumulative_rate, is_final)。
    #   区間通過率 = P(関門通過 | 到達) = 累積_j / 累積_{j-1}
    #   累積通過率 = P(関門1..j を全通過) = forward の pass_rate(joint)
    # 「最終(完走=目標達成)」行を足切り0(残り0)で自動追加する。
    disp = _restart_disp(res, cum_to_label, last_label, D)

    # --- 図: カード名別の最適足切りライン (残りダメージ + 区間/累積通過率) ---
    fig = _cutoff_figure(
        disp, f"最適足切りライン(時短率 {res['speedup']:.2f}x)")

    # --- リスタライン手動調整用の設定 (Store) と スライダー ---
    config = {
        "model": "product" if hp_mode == "on" else "sum",
        "cards": cards, "crit": float(global_crit or 0),
        "evade": float(global_evade or 0), "damage_mode": damage_mode or "post_decay",
        "cps": cps, "hit_times": hit_times, "seg_success": seg_success, "D": D,
        "hp": ({"H": float(hp_H), "H1": float(hp_H1),
                "R0": float(hp_R0), "R1": float(hp_R1)} if hp_mode == "on" else None),
        "cum_to_label": cum_to_label, "last_label": last_label,
        # 最適ライン (重ね描き用) と基準
        "opt_disp": disp,
        "opt_gates": [r["gate"] for r in res["rows"]],
        "opt_success": res["success"], "opt_throughput": res["throughput"],
        "opt_exp_time": res["exp_time"], "opt_speedup": res["speedup"],
        "base_success": res["baseline"]["success"],
        "base_throughput": res["baseline"]["g"],
        "base_exp_time": res["baseline"]["exp_time"],
    }
    sliders = _gate_sliders(cps, res["rows"], cum_to_label, D)

    # 残りダメージが途中で増加する(=関門が累積ダメージで非単調)場合の注記。
    # これは不具合ではなく、後続区間の所要時間が大きいほど DP がその手前の関門を
    # 厳しくする(高コスト区間に入る前に「見切り」をつける)ため。関門通過後は
    # 削るだけなので、通過率がほぼ変わらない関門は実質的に拘束していない。
    # (最終=残り0 の行は常に最小なので除外して判定する。)
    real_remains = [d[1] for d in disp if not d[4]]
    non_monotonic = any(real_remains[i] > real_remains[i - 1] + 0.5
                        for i in range(1, len(real_remains)))

    # --- サマリ ---
    base = res["baseline"]
    rows = [html.Tr([html.Th("チェックポイント(カード)"),
                     html.Th("最適足切り(残りダメージ)"),
                     html.Th("区間通過率"), html.Th("累積通過率")])]
    for label, rem, sect, cumr, is_final in disp:
        style = {"background": "#fff3e0"} if is_final else {}
        rows.append(html.Tr([
            html.Td(label),
            html.Td(f"{rem:,.0f}"),
            html.Td(f"{sect:.1%}"),
            html.Td(f"{cumr:.1%}"),
        ], style=style))
    table = html.Table(rows, style={"borderCollapse": "collapse", "marginTop": "6px"},
                       className="restart-table")
    children = [
        html.Div(model_note, style={"fontSize": "0.85rem", "color": "#888"}),
        html.Div([
            html.Strong("結果: "),
            f"成功率 {res['success']:.3%} / 平均所要時間 {res['exp_time']:.2f} / "
            f"スループット {res['throughput']:.3e}(成功/時間)",
        ]),
        html.Div(
            f"足切り無し: 成功率 {base['success']:.3%} / 時間 {base['exp_time']:.2f} / "
            f"スループット {base['g']:.3e}  →  時短率 {res['speedup']:.2f}x",
            style={"color": "#555", "fontSize": "0.9rem"},
        ),
        table,
        html.Div(
            "区間通過率 = そのチェックポイントに到達した試行のうち足切りを通過する割合"
            "(条件付き)。累積通過率 = 開始からそこまで全関門を通過する割合。"
            "最終行(完走/目標達成)は足切り0(残り0)で自動追加し、累積=目標達成率です。",
            style={"color": "#777", "fontSize": "0.8rem", "marginTop": "6px"},
        ),
    ]
    if non_monotonic:
        children.append(html.Div(
            "※ 残りダメージが途中で増加している箇所があります。これは不具合ではなく、"
            "後続区間の所要時間(時間割合)が大きいほど、その手前の関門を厳しく"
            "(残りダメージを小さく)するのが最適なためです。関門を通過した後は累積"
            "ダメージが増えるだけなので、通過率がほぼ変わらない関門は実質的に拘束して"
            "おらず、設定上は無くても結果は変わりません。",
            style={"color": "#b35900", "fontSize": "0.85rem", "marginTop": "8px"},
        ))
    summary = html.Div(children)
    return fig, summary, config, sliders


# ---------------------------------------------------------------------------
# 多段リスタ: リスタライン手動調整 (スライダー → 成功率/スループット 再計算)
# ---------------------------------------------------------------------------
def _gate_sliders(cps, rows, cum_to_label, D):
    """各足切りの「残りダメージ」スライダーを生成。初期値 = 最適ライン。"""
    sliders = []
    Dmax = int(round(D))
    step = 1
    for k, m in enumerate(cps):
        label = cum_to_label.get(str(m), f"{m}ヒット目")
        opt_remain = min(Dmax, max(0, int(round(D - rows[k]["gate"]))))
        sliders.append(html.Div(
            [
                html.Div(f"足切り{k + 1}:{label}(残りダメージ)",
                         style={"fontSize": "0.83rem", "fontWeight": "bold"}),
                dcc.Slider(
                    id={"type": "restart-gate-slider", "index": m},
                    min=0, max=Dmax, step=step, value=opt_remain,
                    marks={0: "0", Dmax: f"{Dmax:,}"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"marginBottom": "10px"},
        ))
    if not sliders:
        return []
    sliders.append(html.Button(
        "最適ラインに戻す", id="restart-gate-reset-btn", n_clicks=0,
        style={"cursor": "pointer", "padding": "5px 12px", "marginTop": "2px"}))
    return sliders


def _rebuild_for_config(cfg):
    """config からヒット混合・hit_times・(積モデルなら)ymix/hp を再構築する。"""
    hits = build_hit_mixtures(cfg["cards"], cfg["crit"], cfg["evade"],
                              cfg["damage_mode"])
    return hits


@callback(
    Output({"type": "restart-gate-slider", "index": ALL}, "value"),
    Input("restart-gate-reset-btn", "n_clicks"),
    State({"type": "restart-gate-slider", "index": ALL}, "id"),
    State("restart-config", "data"),
    prevent_initial_call=True,
)
def reset_restart_gates(n_clicks, slider_ids, cfg):
    """「最適ラインに戻す」: 各スライダーを最適ラインの残りダメージへ。"""
    if not n_clicks or not cfg or not slider_ids:
        raise PreventUpdate
    D = float(cfg["D"])
    Dmax = int(round(D))
    remains_by = {m: min(Dmax, max(0, int(round(D - g))))
                  for m, g in zip(cfg["cps"], cfg["opt_gates"])}
    return [remains_by.get(sid["index"], 0) for sid in slider_ids]


@callback(
    Output("restart-interactive-graph", "figure"),
    Output("restart-interactive-summary", "children"),
    Input({"type": "restart-gate-slider", "index": ALL}, "value"),
    State({"type": "restart-gate-slider", "index": ALL}, "id"),
    State("restart-config", "data"),
    prevent_initial_call=True,
)
def update_restart_interactive(slider_values, slider_ids, cfg):
    if not cfg or not slider_ids:
        raise PreventUpdate

    D = float(cfg["D"])
    cps = cfg["cps"]
    remains_by = {sid["index"]: (v if v is not None else 0.0)
                  for sid, v in zip(slider_ids, slider_values)}
    # cps の順に残りダメージ → 累積ダメージしきい値 (gate) へ変換
    manual_gates = [D - float(remains_by.get(m, D)) for m in cps]

    hits = _rebuild_for_config(cfg)
    seg_success = cfg.get("seg_success")
    if cfg["model"] == "product":
        hp = HPParams(**cfg["hp"])
        ymix = [y_mixture(mm, hp.beta) for mm in hits]
        res = restart_cos.analyze_product(ymix, hp, cps, cfg["hit_times"], D,
                                          manual_gates=manual_gates,
                                          seg_success=seg_success)
    else:
        res = restart_cos.analyze(hits, cps, cfg["hit_times"], D,
                                  manual_gates=manual_gates,
                                  seg_success=seg_success)

    disp = _restart_disp(res, cfg["cum_to_label"], cfg["last_label"], D)
    fig = _cutoff_figure(disp, "あなたの設定したリスタライン vs 最適",
                         color="#0984e3", ref_disp=cfg["opt_disp"])

    # 最適・基準との比較サマリ
    def pct(x):
        return f"{x:.3%}"
    opt_s, opt_g, opt_t = cfg["opt_success"], cfg["opt_throughput"], cfg["opt_exp_time"]
    base_g = cfg["base_throughput"]
    speedup = (res["throughput"] / base_g) if base_g > 0 else float("nan")
    d_succ = res["success"] - opt_s
    g_ratio = (res["throughput"] / opt_g) if opt_g > 0 else float("nan")
    summary = html.Div([
        html.Div([
            html.Strong("あなたの設定: "),
            f"成功率 {pct(res['success'])} / 平均時間 {res['exp_time']:.2f} / "
            f"スループット {res['throughput']:.3e}(時短率 {speedup:.2f}x)",
        ]),
        html.Div(
            f"最適比: 成功率 {d_succ:+.3%}pt / スループット {g_ratio:.1%}"
            f"(最適 = 成功率 {pct(opt_s)}・スループット {opt_g:.3e}・時短率 "
            f"{cfg['opt_speedup']:.2f}x)",
            style={"color": "#555", "fontSize": "0.88rem"},
        ),
        html.Div(
            "スライダーを動かすと、その足切りライン(残りダメージ)での成功率・"
            "スループットが再計算されます。青=あなたの設定 / 灰点線=最適。",
            style={"color": "#777", "fontSize": "0.8rem", "marginTop": "4px"},
        ),
    ])
    return fig, summary
