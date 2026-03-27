from dash import html, dcc

DEFAULT_CRIT_RATE = 60
DEFAULT_EVADE_RATE = 0

LABEL_STYLE = {"fontSize": "0.85rem", "whiteSpace": "nowrap"}


def make_damage_card(index: int, crit_rate=None, evade_rate=None) -> html.Div:
    """ダメージデータ入力カードを1つ生成する。"""

    def field(label, param, value):
        return html.Div(
            [
                html.Label(label, style=LABEL_STYLE),
                dcc.Input(
                    id={"type": "param", "param": param, "index": index},
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
                    html.Span("⠿", className="drag-handle"),
                    html.Strong(f"ダメージ {index + 1}"),
                    dcc.Input(
                        id={"type": "memo", "index": index},
                        type="text",
                        placeholder="備考",
                        style={"marginLeft": "8px", "flex": "1", "fontSize": "0.85rem"},
                    ),
                    html.Button(
                        "✕",
                        id={"type": "remove-btn", "index": index},
                        n_clicks=0,
                        style={
                            "marginLeft": "auto",
                            "background": "none",
                            "border": "none",
                            "cursor": "pointer",
                            "fontSize": "1.1rem",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    field("会心ダメージ下限", "crit_min", 100000),
                    html.Span("~", style={"alignSelf": "end", "paddingBottom": "4px"}),
                    field("会心ダメージ上限", "crit_max", 120000),
                    field("非会心ダメージ下限", "normal_min", 50000),
                    html.Span("~", style={"alignSelf": "end", "paddingBottom": "4px"}),
                    field("非会心ダメージ上限", "normal_max", 60000),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    field("Hit数", "hits", 10),
                    field("会心率 (%)", "crit_rate", crit_rate),
                    field("回避率 (%)", "evade_rate", evade_rate),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "6px"},
            ),
        ],
        id={"type": "card", "index": index},
        style={
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "padding": "12px",
            "marginBottom": "10px",
            "background": "#fafafa",
        },
    )


def _cutoff_element_row(label: str, elem: str, index: int, is_percent: bool = False) -> html.Div:
    """足切りカードの1要素行（スライダー＋数値表示）を生成する。"""
    slider_max = 100 if is_percent else 10_000_000
    slider_step = 0.01 if is_percent else 1000
    return html.Div(
        [
            html.Label(label, style={**LABEL_STYLE, "fontWeight": "bold", "minWidth": "200px"}),
            html.Div(
                dcc.Slider(
                    id={"type": "cutoff-slider", "elem": elem, "index": index},
                    min=0,
                    max=slider_max,
                    step=slider_step,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                style={"flex": "3"},
            ),
            html.Span("%" if is_percent else "", style={"marginLeft": "4px", "minWidth": "16px"}),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"},
    )


def make_cutoff_card(index: int) -> html.Div:
    """足切りカードを生成する。index はパターンマッチング用。"""
    return html.Div(
        [
            html.Div(
                [
                    html.Span("⠿", className="drag-handle"),
                    html.Strong(f"✂ 足切りライン {index + 1}", style={"color": "#d63031"}),
                    dcc.Input(
                        id={"type": "cutoff-memo", "index": index},
                        type="text",
                        placeholder="備考",
                        style={"marginLeft": "8px", "flex": "1", "fontSize": "0.85rem"},
                    ),
                    html.Button(
                        "足切り計算",
                        id={"type": "cutoff-compute", "index": index},
                        n_clicks=0,
                        style={
                            "marginLeft": "auto",
                            "background": "#d63031",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "4px",
                            "padding": "4px 12px",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "✕",
                        id={"type": "cutoff-remove", "index": index},
                        n_clicks=0,
                        style={
                            "marginLeft": "8px",
                            "background": "none",
                            "border": "none",
                            "cursor": "pointer",
                            "fontSize": "1.1rem",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "12px"},
            ),
            _cutoff_element_row("上側ダメージ合計（足切り値）", "e1", index),
            _cutoff_element_row("上側超過確率", "e2", index, is_percent=True),
            _cutoff_element_row("残り必要ダメージ（目標−足切り値）", "e3", index),
            _cutoff_element_row("下側超過確率", "e4", index, is_percent=True),
            html.Div(
                id={"type": "cutoff-status", "index": index},
                style={"fontSize": "0.85rem", "color": "#666", "marginTop": "4px"},
            ),
        ],
        id={"type": "cutoff", "index": index},
        style={
            "border": "2px solid #d63031",
            "borderRadius": "8px",
            "padding": "12px",
            "marginBottom": "10px",
            "background": "#fff0f0",
        },
    )


def _sidebar() -> html.Div:
    """左サイドバー: 設定パネル。"""
    section_style = {
        "marginBottom": "16px",
        "padding": "10px",
        "border": "1px solid #ddd",
        "borderRadius": "8px",
    }
    return html.Div(
        [
            # ダメージ生成モード
            html.Div(
                [
                    html.Strong("ダメージ生成モード"),
                    dcc.RadioItems(
                        id="damage-mode",
                        options=[
                            {"label": "減衰考慮済み（推奨）", "value": "post_decay"},
                            {"label": "減衰考慮前", "value": "pre_decay"},
                        ],
                        value="post_decay",
                        style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginTop": "6px"},
                    ),
                ],
                style={**section_style, "background": "#f0f8f0"},
            ),
            # 目標ダメージ
            html.Div(
                [
                    html.Strong("目標ダメージ"),
                    dcc.Input(id="target-damage", type="number", value=0, style={"width": "100%", "marginTop": "6px"}),
                ],
                style={**section_style, "background": "#fff5f5"},
            ),
            # 一括設定
            html.Div(
                [
                    html.Strong("一括設定"),
                    html.Div(
                        [
                            html.Label("会心率(%)", style=LABEL_STYLE),
                            dcc.Input(id="global-crit-rate", type="number", value=DEFAULT_CRIT_RATE, style={"width": "100%"}),
                        ],
                        style={"marginTop": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("回避率(%)", style=LABEL_STYLE),
                            dcc.Input(id="global-evade-rate", type="number", value=DEFAULT_EVADE_RATE, style={"width": "100%"}),
                        ],
                        style={"marginTop": "6px"},
                    ),
                    html.Button("一括適用", id="apply-global-btn", n_clicks=0, style={"marginTop": "8px", "width": "100%"}),
                ],
                style={**section_style, "background": "#f5f5ff"},
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
            html.H1("ブルアカダメージ足切りシミュレータ", style={"marginBottom": "16px"}),
            html.Div(
                [
                    # サイドバー
                    _sidebar(),
                    # メインコンテンツ
                    html.Div(
                        [
                            html.Div(id="cards-container", children=[make_damage_card(0, DEFAULT_CRIT_RATE, DEFAULT_EVADE_RATE)]),
                            html.Div(
                                [
                                    html.Button("+ ダメージ追加", id="add-btn", n_clicks=0),
                                    html.Button("✂ 足切り追加", id="add-cutoff-btn", n_clicks=0, style={"marginLeft": "12px"}),
                                    html.Button(
                                        "シミュレーション実行",
                                        id="run-btn",
                                        n_clicks=0,
                                        style={"marginLeft": "12px"},
                                    ),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                            html.Div(id="pass-rate-text", style={"fontSize": "1.2rem", "fontWeight": "bold", "marginBottom": "8px"}),
                            dcc.Graph(id="result-graph"),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
            ),
            # 非表示 Store 群
            dcc.Store(id="drag-order", data=""),
            dcc.Store(id="card-indices", data=[0]),
            dcc.Store(id="sorted-indices", data=[]),
            dcc.Store(id="next-index", data=1),
            dcc.Store(id="cutoff-indices", data=[]),
            dcc.Store(id="cutoff-next-index", data=0),
            dcc.Store(id="cutoff-dist-store", data={}),
            dcc.Store(id="cutoff-values-store", data={}),
            dcc.Store(id="cutoff-trigger-store", data=None),
            dcc.Store(id="cutoff-generation", data=0),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px", "fontFamily": "sans-serif"},
    )
