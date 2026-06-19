"""Edgeworth 展開 実験 用の独立した Dash ミニアプリ。

ブラウザでカード追加・パラメータ編集・グローバル設定・実験設定をして、
「実験実行」ボタンで `experiments.edgeworth_animation.run_experiment` を呼び、
生成された GIF を画面に埋め込んで表示する。

実行例:
    uv run python -m experiments.edgeworth_app
    # → http://127.0.0.1:8060/
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.edgeworth_animation import (  # noqa: E402
    _frame_indices_for_anim_mode,
    prepare_simulation,
    save_animation,
    save_sup_error_plot,
)


# ---------------------------------------------------------------------------
# 既定値
# ---------------------------------------------------------------------------
DEFAULT_GLOBAL_CRIT = 50
DEFAULT_GLOBAL_EVADE = 10
DEFAULT_N_MC = 1_000_000
DEFAULT_BINS = 120
DEFAULT_SIGMA = 1.2
DEFAULT_FPS = 2

DEFAULT_CARD_PARAMS: dict = {
    "crit_min": 2_000_000,
    "crit_max": 3_000_000,
    "normal_min": 1_000_000,
    "normal_max": 1_500_000,
    "hits": 4,
    "crit_rate": None,
    "evade_rate": None,
}

LABEL_STYLE = {"fontSize": "0.85rem", "whiteSpace": "nowrap"}

CARD_BG = "#fafafa"
PRIMARY = "#d63031"


def _icon_button(label: str, btn_type: str, idx: int, title: str) -> html.Button:
    return html.Button(
        label,
        id={"type": btn_type, "index": idx},
        n_clicks=0,
        title=title,
        style={
            "marginLeft": "6px",
            "background": "none",
            "border": "none",
            "cursor": "pointer",
            "fontSize": "1.1rem",
        },
    )


def make_card(idx: int, params: dict | None = None, memo: str = "") -> html.Div:
    """1枚分の入力カード (本番アプリの make_damage_card に揃えたレイアウト)。"""
    p = dict(DEFAULT_CARD_PARAMS)
    if params:
        p.update(params)

    def field(label: str, param: str, value) -> html.Div:
        return html.Div(
            [
                html.Label(label, style=LABEL_STYLE),
                dcc.Input(
                    id={"type": "exp-param", "param": param, "index": idx},
                    type="number",
                    value=value,
                    style={"width": "100%"},
                ),
            ],
            style={"flex": "1", "minWidth": "100px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Strong(f"カード {idx + 1}"),
                    dcc.Input(
                        id={"type": "exp-memo", "index": idx},
                        type="text",
                        placeholder="備考",
                        value=memo or "",
                        style={"marginLeft": "8px", "flex": "1", "fontSize": "0.85rem"},
                    ),
                    _icon_button("↑", "exp-move-up", idx, "上へ移動"),
                    _icon_button("↓", "exp-move-down", idx, "下へ移動"),
                    _icon_button("📋", "exp-duplicate", idx, "カードを複製"),
                    _icon_button("✕", "exp-remove", idx, "カードを削除"),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    field("会心ダメージ下限", "crit_min", p["crit_min"]),
                    html.Span("~", style={"alignSelf": "end", "paddingBottom": "4px"}),
                    field("会心ダメージ上限", "crit_max", p["crit_max"]),
                    field("非会心ダメージ下限", "normal_min", p["normal_min"]),
                    html.Span("~", style={"alignSelf": "end", "paddingBottom": "4px"}),
                    field("非会心ダメージ上限", "normal_max", p["normal_max"]),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    field("Hit数", "hits", p["hits"]),
                    field("会心率 (%)", "crit_rate", p["crit_rate"]),
                    field("回避率 (%)", "evade_rate", p["evade_rate"]),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "6px"},
            ),
        ],
        id={"type": "exp-card", "index": idx},
        style={
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "padding": "12px",
            "marginBottom": "10px",
            "background": CARD_BG,
        },
    )


SECTION_STYLE = {
    "marginBottom": "16px",
    "padding": "10px",
    "border": "1px solid #ddd",
    "borderRadius": "8px",
}


def _labelled(label: str, comp) -> html.Div:
    return html.Div(
        [html.Label(label, style=LABEL_STYLE), comp],
        style={"marginTop": "6px"},
    )


def _sidebar() -> html.Div:
    return html.Div(
        [
            # ダメージ生成モード
            html.Div(
                [
                    html.Strong("ダメージ生成モード"),
                    dcc.RadioItems(
                        id="exp-damage-mode",
                        options=[
                            {"label": "減衰考慮済み（推奨）", "value": "post_decay"},
                            {"label": "減衰考慮前", "value": "pre_decay"},
                        ],
                        value="post_decay",
                        style={"display": "flex", "flexDirection": "column",
                               "gap": "4px", "marginTop": "6px"},
                    ),
                ],
                style={**SECTION_STYLE, "background": "#f0f8f0"},
            ),
            # 一括設定 (会心率・回避率)
            html.Div(
                [
                    html.Strong("一括設定"),
                    _labelled(
                        "会心率(%)",
                        dcc.Input(id="exp-global-crit", type="number",
                                  value=DEFAULT_GLOBAL_CRIT,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "回避率(%)",
                        dcc.Input(id="exp-global-evade", type="number",
                                  value=DEFAULT_GLOBAL_EVADE,
                                  style={"width": "100%"}),
                    ),
                ],
                style={**SECTION_STYLE, "background": "#f5f5ff"},
            ),
            # 実験設定
            html.Div(
                [
                    html.Strong("実験設定"),
                    _labelled(
                        "アニメーション単位",
                        dcc.RadioItems(
                            id="exp-anim-mode",
                            options=[
                                {"label": "Hit ごと", "value": "hit"},
                                {"label": "カード ごと", "value": "card"},
                            ],
                            value="hit",
                            style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                        ),
                    ),
                    _labelled(
                        "MC サンプル数",
                        dcc.Input(id="exp-n-mc", type="number",
                                  value=DEFAULT_N_MC, min=100_000, step=100_000,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "ヒストグラム bin 数",
                        dcc.Input(id="exp-hist-bins", type="number",
                                  value=DEFAULT_BINS, min=10, max=500, step=10,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "平滑化 σ (下段の誤差・sup誤差用, bin単位)",
                        dcc.Input(id="exp-smooth-sigma", type="number",
                                  value=DEFAULT_SIGMA, min=0, step=0.1,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "FPS",
                        dcc.Input(id="exp-fps", type="number",
                                  value=DEFAULT_FPS, min=1, max=30, step=1,
                                  style={"width": "100%"}),
                    ),
                ],
                style={**SECTION_STYLE, "background": "#fff5f5"},
            ),
        ],
        style={
            "width": "220px",
            "flexShrink": "0",
            "position": "sticky",
            "top": "20px",
            "alignSelf": "flex-start",
        },
    )


def create_layout() -> html.Div:
    return html.Div(
        [
            html.H2("Edgeworth 展開 vs Monte Carlo  実験UI"),
            html.Div(
                [
                    _sidebar(),
                    html.Div(
                        [
                            html.Div(
                                id="exp-cards-container",
                                children=[make_card(0)],
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "+ カード追加",
                                        id="exp-add-btn",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "実験実行",
                                        id="exp-run-btn",
                                        n_clicks=0,
                                        style={
                                            "marginLeft": "12px",
                                            "background": PRIMARY,
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "4px",
                                            "padding": "6px 16px",
                                            "cursor": "pointer",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    html.Span(
                                        " 注: サンプル数 × 総Hit数 が大きいとブラウザがしばらく待たされます。",
                                        style={"marginLeft": "12px", "fontSize": "0.8rem", "color": "#666"},
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "アニメGIFをダウンロード",
                                        id="exp-dl-anim-btn",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "sup誤差PNGをダウンロード",
                                        id="exp-dl-sup-btn",
                                        n_clicks=0,
                                        disabled=True,
                                        style={"marginLeft": "12px"},
                                    ),
                                ],
                                style={"marginTop": "12px"},
                            ),
                            dcc.Loading(
                                html.Div(id="exp-output", style={"marginTop": "12px"}),
                                type="circle",
                                color=PRIMARY,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
            ),
            dcc.Store(id="exp-card-indices", data=[0]),
            dcc.Store(id="exp-next-index", data=1),
            dcc.Store(id="exp-anim-b64"),
            dcc.Store(id="exp-sup-b64"),
            dcc.Download(id="exp-dl-anim"),
            dcc.Download(id="exp-dl-sup"),
        ],
        style={
            "maxWidth": "1200px",
            "margin": "0 auto",
            "padding": "20px",
            "fontFamily": "sans-serif",
        },
    )


app = Dash(__name__)
app.title = "Edgeworth 展開 実験"
app.layout = create_layout


# ---------------------------------------------------------------------------
# カード追加・削除
# ---------------------------------------------------------------------------

@app.callback(
    Output("exp-cards-container", "children"),
    Output("exp-card-indices", "data"),
    Output("exp-next-index", "data"),
    Input("exp-add-btn", "n_clicks"),
    Input({"type": "exp-remove", "index": ALL}, "n_clicks"),
    Input({"type": "exp-duplicate", "index": ALL}, "n_clicks"),
    Input({"type": "exp-move-up", "index": ALL}, "n_clicks"),
    Input({"type": "exp-move-down", "index": ALL}, "n_clicks"),
    State("exp-cards-container", "children"),
    State("exp-card-indices", "data"),
    State("exp-next-index", "data"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def manage_cards(_add_clicks, _remove_clicks, _dup_clicks, _up_clicks, _down_clicks,
                 cards, indices, next_idx,
                 param_values, param_ids, memo_values, memo_ids):
    trig = ctx.triggered_id
    if trig is None:
        return no_update, no_update, no_update

    cards = list(cards or [])
    indices = list(indices or [])

    if trig == "exp-add-btn":
        return cards + [make_card(next_idx)], indices + [next_idx], next_idx + 1

    if not (isinstance(trig, dict) and ctx.triggered[0]["value"]):
        # パターンマッチ要素の追加/削除に伴う空発火 (None や 0) は無視
        return no_update, no_update, no_update

    target = trig["index"]
    ttype = trig.get("type")

    if ttype == "exp-remove":
        cards = [c for c in cards
                 if c.get("props", {}).get("id", {}).get("index") != target]
        indices = [i for i in indices if i != target]
        return cards, indices, next_idx

    if target not in indices:
        return no_update, no_update, no_update
    pos = indices.index(target)

    if ttype == "exp-duplicate":
        src_params = {pid["param"]: v
                      for v, pid in zip(param_values, param_ids)
                      if pid["index"] == target}
        src_memo = next((v or "" for v, mid in zip(memo_values, memo_ids)
                         if mid["index"] == target), "")
        cards.insert(pos + 1, make_card(next_idx, params=src_params, memo=src_memo))
        indices.insert(pos + 1, next_idx)
        return cards, indices, next_idx + 1

    if ttype == "exp-move-up" and pos > 0:
        cards[pos - 1], cards[pos] = cards[pos], cards[pos - 1]
        indices[pos - 1], indices[pos] = indices[pos], indices[pos - 1]
        return cards, indices, next_idx

    if ttype == "exp-move-down" and pos < len(indices) - 1:
        cards[pos + 1], cards[pos] = cards[pos], cards[pos + 1]
        indices[pos + 1], indices[pos] = indices[pos], indices[pos + 1]
        return cards, indices, next_idx

    return no_update, no_update, no_update


# ---------------------------------------------------------------------------
# 実験実行
# ---------------------------------------------------------------------------

def _assemble_cards(values, ids, memo_values, memo_ids, ordered_indices) -> list[dict]:
    """flat な values/ids/memos を、カードごとの dict のリストに集約する。"""
    per_card: dict[int, dict] = {}
    for v, sid in zip(values, ids):
        per_card.setdefault(sid["index"], {})[sid["param"]] = v
    for v, sid in zip(memo_values, memo_ids):
        per_card.setdefault(sid["index"], {})["memo"] = v or ""
    return [per_card[i] for i in ordered_indices if i in per_card]


def _b64_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _card_breakpoints(sim, cards: list[dict]) -> list[tuple[int, str]]:
    """sup 誤差プロットの縦線用に、各カード末尾の (累積Hit数, ラベル) を返す。
    ラベルは備考があれば備考、無ければ空 (縦線のみ)。"""
    breakpoints: list[tuple[int, str]] = []
    for i, cum in enumerate(sim.card_boundaries):
        memo = str((cards[i] or {}).get("memo") or "").strip() if i < len(cards) else ""
        breakpoints.append((cum, memo))
    return breakpoints


@app.callback(
    Output("exp-output", "children"),
    Output("exp-anim-b64", "data"),
    Output("exp-sup-b64", "data"),
    Output("exp-dl-anim-btn", "disabled"),
    Output("exp-dl-sup-btn", "disabled"),
    Input("exp-run-btn", "n_clicks"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    State("exp-card-indices", "data"),
    State("exp-global-crit", "value"),
    State("exp-global-evade", "value"),
    State("exp-damage-mode", "value"),
    State("exp-anim-mode", "value"),
    State("exp-n-mc", "value"),
    State("exp-hist-bins", "value"),
    State("exp-smooth-sigma", "value"),
    State("exp-fps", "value"),
    prevent_initial_call=True,
)
def run(n_clicks, values, ids, memo_values, memo_ids, indices,
        global_crit, global_evade, damage_mode, anim_mode,
        n_mc, hist_bins, smooth_sigma, fps):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update

    cards = _assemble_cards(values, ids, memo_values, memo_ids, indices or [])
    if not cards:
        return (html.Div("カードが0枚です。", style={"color": "red"}),
                no_update, no_update, True, True)

    sigma = float(smooth_sigma if smooth_sigma is not None else DEFAULT_SIGMA)
    gif_path = sup_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            sup_path = tmp.name

        # MC は 1 回だけ実行し、アニメと sup 誤差の両方で再利用する。
        sim = prepare_simulation(
            cards,
            float(global_crit or 0),
            float(global_evade or 0),
            damage_mode,
            int(n_mc or DEFAULT_N_MC),
            int(hist_bins or DEFAULT_BINS),
            42,
        )
        frame_indices, frame_labels, suffix = _frame_indices_for_anim_mode(
            sim, anim_mode, cards
        )
        save_animation(
            sim, frame_indices, frame_labels, suffix,
            gif_path, int(fps or DEFAULT_FPS), sigma,
        )
        save_sup_error_plot(
            sim, sigma, sup_path, breakpoints=_card_breakpoints(sim, cards),
        )

        gif_b64 = _b64_of_file(gif_path)
        sup_b64 = _b64_of_file(sup_path)
    except Exception as e:
        return (html.Div(f"エラー: {e!s}", style={"color": "red", "whiteSpace": "pre-wrap"}),
                no_update, no_update, True, True)
    finally:
        for p in (gif_path, sup_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass

    output = html.Div(
        [
            html.Div(
                f"カード数: {len(cards)} / "
                f"GIFサイズ: {len(gif_b64) * 3 // 4 // 1024} KB / "
                f"sup誤差PNGサイズ: {len(sup_b64) * 3 // 4 // 1024} KB",
                style={"fontSize": "0.85rem", "color": "#555", "marginBottom": "6px"},
            ),
            html.Img(
                src=f"data:image/gif;base64,{gif_b64}",
                style={"maxWidth": "100%", "border": "1px solid #ddd"},
            ),
            html.H4("各手法の最大誤差 (sup 誤差)", style={"marginTop": "16px"}),
            html.Img(
                src=f"data:image/png;base64,{sup_b64}",
                style={"maxWidth": "100%", "border": "1px solid #ddd"},
            ),
        ]
    )
    return output, gif_b64, sup_b64, False, False


@app.callback(
    Output("exp-dl-anim", "data"),
    Input("exp-dl-anim-btn", "n_clicks"),
    State("exp-anim-b64", "data"),
    prevent_initial_call=True,
)
def download_anim(n_clicks, gif_b64):
    if not n_clicks or not gif_b64:
        return no_update
    return dict(content=gif_b64, filename="edgeworth_animation.gif", base64=True)


@app.callback(
    Output("exp-dl-sup", "data"),
    Input("exp-dl-sup-btn", "n_clicks"),
    State("exp-sup-b64", "data"),
    prevent_initial_call=True,
)
def download_sup(n_clicks, sup_b64):
    if not n_clicks or not sup_b64:
        return no_update
    return dict(content=sup_b64, filename="edgeworth_sup_error.png", base64=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8060))
    debug = "PORT" not in os.environ
    app.run(host="127.0.0.1", port=port, debug=debug)
