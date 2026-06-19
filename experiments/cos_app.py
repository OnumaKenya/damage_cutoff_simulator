"""COS法 vs Edgeworth 裾確率比較 用の独立した Dash ミニアプリ。

edgeworth_app.py と同じ操作感で、ブラウザ上でカード追加・パラメータ編集・
グローバル設定・比較設定をして、「比較実行」ボタンで
`experiments.cos_compare.make_comparison_plot` を呼び、生成された
比較プロット (PNG) を画面に埋め込んで表示する。MC・Edgeworth 2次・COS法
(DPあり/なし) を比較する。

カード編集まわりの UI とコールバックは edgeworth_app から再利用する
(make_card / manage_cards / _assemble_cards は "exp-*" の id を使う)。

実行例:
    uv run python -m experiments.cos_app
    # → http://127.0.0.1:8061/
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile

from dash import ALL, Dash, Input, Output, State, dcc, html, no_update

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.edgeworth_animation import (  # noqa: E402
    CARDS as _REAL_DATA_CARDS,
    DAMAGE_MODE as _REAL_DATA_DAMAGE_MODE,
    GLOBAL_CRIT_RATE as _REAL_DATA_CRIT,
    GLOBAL_EVADE_RATE as _REAL_DATA_EVADE,
    build_all_hits,
)
from experiments.edgeworth_app import (  # noqa: E402
    CARD_BG,
    DEFAULT_GLOBAL_CRIT,
    DEFAULT_GLOBAL_EVADE,
    PRIMARY,
    SECTION_STYLE,
    _assemble_cards,
    _labelled,
    make_card,
    manage_cards,
)
from experiments.cos_compare import Scenario, make_comparison_plot  # noqa: E402


# ---------------------------------------------------------------------------
# 既定値
# ---------------------------------------------------------------------------
DEFAULT_N_MC = 20_000_000     # 基準線用の多数サンプル MC
DEFAULT_N_GRID = 300
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# プリセット (シナリオ): カード列 + 一括設定をまとめて読み込めるようにする
# ---------------------------------------------------------------------------
_BLUEARCHIVE_7HIT_CARDS = [
    {"crit_min": 4_475_134, "crit_max": 5_508_470, "normal_min": 1_039_889,
     "normal_max": 1_332_274, "hits": 2, "crit_rate": None, "evade_rate": None,
     "memo": "1・2射目"},
    {"crit_min": 5_237_898, "crit_max": 6_356_831, "normal_min": 1_255_715,
     "normal_max": 1_608_784, "hits": 1, "crit_rate": None, "evade_rate": None,
     "memo": "3射目"},
    {"crit_min": 5_962_738, "crit_max": 7_150_830, "normal_min": 1_471_541,
     "normal_max": 1_885_293, "hits": 2, "crit_rate": None, "evade_rate": None,
     "memo": "4・5射目"},
    {"crit_min": 4_592_342, "crit_max": 5_658_632, "normal_min": 1_073_053,
     "normal_max": 1_374_763, "hits": 2, "crit_rate": 60.17, "evade_rate": None,
     "memo": "6・7射目"},
]

# UI から登録 (会心58.22%・回避0%・計39Hit)。各カードは下限=上限の点入力。
_PRESET2_CARDS = [
    {"crit_min": 630_786, "crit_max": 630_786, "normal_min": 169_868,
     "normal_max": 169_868, "hits": 6, "crit_rate": None, "evade_rate": None,
     "memo": "1・2射目"},
    {"crit_min": 564_998, "crit_max": 564_998, "normal_min": 152_151,
     "normal_max": 152_151, "hits": 7, "crit_rate": None, "evade_rate": None,
     "memo": "3射目"},
    {"crit_min": 776_921, "crit_max": 776_921, "normal_min": 209_221,
     "normal_max": 209_221, "hits": 12, "crit_rate": None, "evade_rate": None,
     "memo": "1,2射目"},
    {"crit_min": 695_891, "crit_max": 695_891, "normal_min": 187_400,
     "normal_max": 187_400, "hits": 14, "crit_rate": None, "evade_rate": None,
     "memo": "3射目"},
]

PRESETS: dict = {
    "ブルアカ 7発 (会心65.27%・回避0%)": {
        "cards": _BLUEARCHIVE_7HIT_CARDS,
        "global_crit": 65.27,
        "global_evade": 0.0,
        "damage_mode": "post_decay",
    },
    "プリセット2 (会心58.22%・回避0%・39Hit)": {
        "cards": _PRESET2_CARDS,
        "global_crit": 58.22,
        "global_evade": 0.0,
        "damage_mode": "post_decay",
    },
    "実データ (会心61.35%・回避0%・132Hit)": {
        "cards": _REAL_DATA_CARDS,
        "global_crit": float(_REAL_DATA_CRIT),
        "global_evade": float(_REAL_DATA_EVADE),
        "damage_mode": _REAL_DATA_DAMAGE_MODE,
    },
}
DEFAULT_PRESET = "ブルアカ 7発 (会心65.27%・回避0%)"


def _sidebar() -> html.Div:
    return html.Div(
        [
            # プリセット読込
            html.Div(
                [
                    html.Strong("プリセット"),
                    _labelled(
                        "シナリオ",
                        dcc.Dropdown(
                            id="cos-preset",
                            options=[{"label": k, "value": k} for k in PRESETS],
                            value=DEFAULT_PRESET,
                            clearable=False,
                        ),
                    ),
                    html.Button(
                        "プリセット読込", id="cos-preset-load-btn", n_clicks=0,
                        style={"marginTop": "8px", "width": "100%", "cursor": "pointer"},
                    ),
                ],
                style={**SECTION_STYLE, "background": "#fffbe6"},
            ),
            # ダメージ生成モード
            html.Div(
                [
                    html.Strong("ダメージ生成モード"),
                    dcc.RadioItems(
                        id="cos-damage-mode",
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
                        dcc.Input(id="cos-global-crit", type="number",
                                  value=DEFAULT_GLOBAL_CRIT,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "回避率(%)",
                        dcc.Input(id="cos-global-evade", type="number",
                                  value=DEFAULT_GLOBAL_EVADE,
                                  style={"width": "100%"}),
                    ),
                ],
                style={**SECTION_STYLE, "background": "#f5f5ff"},
            ),
            # 比較設定
            html.Div(
                [
                    html.Strong("比較設定"),
                    _labelled(
                        "MC サンプル数",
                        dcc.Input(id="cos-n-mc", type="number",
                                  value=DEFAULT_N_MC, min=100_000, step=100_000,
                                  style={"width": "100%"}),
                    ),
                    dcc.Checklist(
                        id="cos-show-edge",
                        options=[{"label": " Edgeworth 2次 を表示", "value": "show"}],
                        value=["show"],
                        style={"marginTop": "8px"},
                    ),
                    dcc.Checklist(
                        id="cos-show-nodp",
                        options=[{"label": " COS法 (DPなし) も表示", "value": "show"}],
                        value=[],
                        style={"marginTop": "4px"},
                    ),
                    _labelled(
                        "x グリッド点数",
                        dcc.Input(id="cos-n-grid", type="number",
                                  value=DEFAULT_N_GRID, min=50, max=1000, step=50,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "乱数シード",
                        dcc.Input(id="cos-seed", type="number",
                                  value=DEFAULT_SEED, step=1,
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
            html.H2("COS法 vs Edgeworth  裾確率比較UI"),
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
                                        "比較実行",
                                        id="cos-run-btn",
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
                                        " 注: MC サンプル数 × 総Hit数 が大きいとしばらく待たされます。",
                                        style={"marginLeft": "12px", "fontSize": "0.8rem", "color": "#666"},
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "裾確率PNGをダウンロード",
                                        id="cos-dl-btn",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "密度PNGをダウンロード",
                                        id="cos-dl-density-btn",
                                        n_clicks=0,
                                        disabled=True,
                                        style={"marginLeft": "12px"},
                                    ),
                                ],
                                style={"marginTop": "12px"},
                            ),
                            dcc.Loading(
                                html.Div(id="cos-output", style={"marginTop": "12px"}),
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
            dcc.Store(id="cos-png-b64"),
            dcc.Store(id="cos-density-b64"),
            dcc.Download(id="cos-dl"),
            dcc.Download(id="cos-dl-density"),
        ],
        style={
            "maxWidth": "1200px",
            "margin": "0 auto",
            "padding": "20px",
            "fontFamily": "sans-serif",
        },
    )


app = Dash(__name__)
app.title = "COS法 裾確率比較"
app.layout = create_layout


# ---------------------------------------------------------------------------
# カード追加・削除 (edgeworth_app の manage_cards をそのまま再利用)
# ---------------------------------------------------------------------------
app.callback(
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
)(manage_cards)


# ---------------------------------------------------------------------------
# プリセット読込: カード列と一括設定をまとめて差し替える
# ---------------------------------------------------------------------------
@app.callback(
    Output("exp-cards-container", "children", allow_duplicate=True),
    Output("exp-card-indices", "data", allow_duplicate=True),
    Output("exp-next-index", "data", allow_duplicate=True),
    Output("cos-global-crit", "value"),
    Output("cos-global-evade", "value"),
    Output("cos-damage-mode", "value"),
    Input("cos-preset-load-btn", "n_clicks"),
    State("cos-preset", "value"),
    prevent_initial_call=True,
)
def load_preset(n_clicks, name):
    if not n_clicks or name not in PRESETS:
        return (no_update,) * 6
    preset = PRESETS[name]
    cards = preset["cards"]
    children = [make_card(i, params=c, memo=c.get("memo", ""))
                for i, c in enumerate(cards)]
    indices = list(range(len(cards)))
    return (children, indices, len(cards),
            preset["global_crit"], preset["global_evade"], preset["damage_mode"])


# ---------------------------------------------------------------------------
# 比較実行
# ---------------------------------------------------------------------------

def _b64_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@app.callback(
    Output("cos-output", "children"),
    Output("cos-png-b64", "data"),
    Output("cos-dl-btn", "disabled"),
    Output("cos-density-b64", "data"),
    Output("cos-dl-density-btn", "disabled"),
    Input("cos-run-btn", "n_clicks"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    State("exp-card-indices", "data"),
    State("cos-global-crit", "value"),
    State("cos-global-evade", "value"),
    State("cos-damage-mode", "value"),
    State("cos-n-mc", "value"),
    State("cos-show-edge", "value"),
    State("cos-show-nodp", "value"),
    State("cos-n-grid", "value"),
    State("cos-seed", "value"),
    prevent_initial_call=True,
)
def run(n_clicks, values, ids, memo_values, memo_ids, indices,
        global_crit, global_evade, damage_mode,
        n_mc, show_edge, show_nodp, n_grid, seed):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update

    cards = _assemble_cards(values, ids, memo_values, memo_ids, indices or [])
    if not cards:
        return (html.Div("カードが0枚です。", style={"color": "red"}),
                no_update, True, no_update, True)

    png_path = density_path = ""
    try:
        hit_mixtures, _ = build_all_hits(
            cards, float(global_crit or 0), float(global_evade or 0), damage_mode
        )
        if not hit_mixtures:
            return (html.Div("Hit がありません。", style={"color": "red"}),
                    no_update, True, no_update, True)
        sc = Scenario("", hit_mixtures)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            density_path = tmp.name
        make_comparison_plot(
            sc,
            int(n_mc or DEFAULT_N_MC),
            int(seed if seed is not None else DEFAULT_SEED),
            int(n_grid or DEFAULT_N_GRID),
            png_path,
            density_output_path=density_path,
            show_cos_nodp=bool(show_nodp),
            show_edgeworth=bool(show_edge),
        )
        png_b64 = _b64_of_file(png_path)
        density_b64 = _b64_of_file(density_path)
    except Exception as e:
        return (html.Div(f"エラー: {e!s}",
                         style={"color": "red", "whiteSpace": "pre-wrap"}),
                no_update, True, no_update, True)
    finally:
        for p in (png_path, density_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass

    output = html.Div(
        [
            html.Div(
                f"Hit数: {sc.n_hits} / 平均: {sc.mean:,.0f} / 標準偏差: {sc.std:,.0f} / "
                f"λ3: {sc.lam3:.3f} / λ4: {sc.lam4:.3f}",
                style={"fontSize": "0.85rem", "color": "#555", "marginBottom": "6px"},
            ),
            html.H4("裾確率の比較", style={"marginTop": "8px"}),
            html.Img(
                src=f"data:image/png;base64,{png_b64}",
                style={"maxWidth": "100%", "border": "1px solid #ddd"},
            ),
            html.H4("密度関数の比較", style={"marginTop": "16px"}),
            html.Img(
                src=f"data:image/png;base64,{density_b64}",
                style={"maxWidth": "100%", "border": "1px solid #ddd"},
            ),
        ]
    )
    return output, png_b64, False, density_b64, False


@app.callback(
    Output("cos-dl", "data"),
    Input("cos-dl-btn", "n_clicks"),
    State("cos-png-b64", "data"),
    prevent_initial_call=True,
)
def download_png(n_clicks, png_b64):
    if not n_clicks or not png_b64:
        return no_update
    return dict(content=png_b64, filename="cos_tail.png", base64=True)


@app.callback(
    Output("cos-dl-density", "data"),
    Input("cos-dl-density-btn", "n_clicks"),
    State("cos-density-b64", "data"),
    prevent_initial_call=True,
)
def download_density(n_clicks, density_b64):
    if not n_clicks or not density_b64:
        return no_update
    return dict(content=density_b64, filename="cos_density.png", base64=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8061))
    debug = "PORT" not in os.environ
    app.run(host="127.0.0.1", port=port, debug=debug)
