import math
from pathlib import Path

from dash import html, dcc

_MANUAL_MD = (Path(__file__).resolve().parent.parent.parent / "docs" / "manual.md").read_text(encoding="utf-8")

DEFAULT_CRIT_RATE = 60
DEFAULT_EVADE_RATE = 0

LABEL_STYLE = {"fontSize": "0.85rem", "whiteSpace": "nowrap"}

# ---------------------------------------------------------------------------
# 対数スライダー変換 (内部値 0–400 ↔ 0.01%–100%)
# ---------------------------------------------------------------------------
LOG_SLIDER_MIN = 0
LOG_SLIDER_MAX = 400


def log_slider_to_pct(val: float) -> float:
    """内部スライダー値 (0–400) → パーセント (0.01–100)"""
    return 10 ** ((val / 100) - 2)


def pct_to_log_slider(pct: float) -> float:
    """パーセント (0.01–100) → 内部スライダー値 (0–400)"""
    if pct <= 0:
        return LOG_SLIDER_MIN
    val = (math.log10(pct) + 2) * 100
    return max(LOG_SLIDER_MIN, min(LOG_SLIDER_MAX, val))


_DEFAULT_PARAMS = {
    "crit_min": 100000,
    "crit_max": 120000,
    "normal_min": 50000,
    "normal_max": 60000,
    "hits": 10,
    "crit_rate": None,
    "evade_rate": None,
}


def make_damage_card(
    index: int,
    crit_rate=None,
    evade_rate=None,
    *,
    params: dict | None = None,
    memo: str = "",
) -> html.Div:
    """ダメージデータ入力カードを1つ生成する。

    params が渡された場合はその値を使い、なければデフォルト値を使う。
    crit_rate / evade_rate は後方互換のために残す（params 未指定時のみ有効）。
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)
    else:
        if crit_rate is not None:
            p["crit_rate"] = crit_rate
        if evade_rate is not None:
            p["evade_rate"] = evade_rate

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
                        value=memo or "",
                        style={"marginLeft": "8px", "flex": "1", "fontSize": "0.85rem"},
                    ),
                    html.Button(
                        "📋",
                        id={"type": "duplicate-btn", "index": index},
                        n_clicks=0,
                        title="カードを複製",
                        style={
                            "marginLeft": "auto",
                            "background": "none",
                            "border": "none",
                            "cursor": "pointer",
                            "fontSize": "1.1rem",
                        },
                    ),
                    html.Button(
                        "✕",
                        id={"type": "remove-btn", "index": index},
                        n_clicks=0,
                        style={
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
    if is_percent:
        slider = dcc.Slider(
            id={"type": "cutoff-slider", "elem": elem, "index": index},
            min=LOG_SLIDER_MIN,
            max=LOG_SLIDER_MAX,
            step=1,
            value=0,
            marks={0: "0.01%", 100: "0.1%", 200: "1%", 300: "10%", 400: "100%"},
        )
        suffix = html.Div(
            [
                dcc.Input(
                    id={"type": "cutoff-pct-input", "elem": elem, "index": index},
                    type="number",
                    value=0.01,
                    min=0.01,
                    max=100,
                    step=0.01,
                    debounce=True,
                    style={"width": "80px", "fontSize": "0.85rem", "textAlign": "right"},
                ),
                html.Span("%", style={"marginLeft": "2px"}),
            ],
            style={"display": "flex", "alignItems": "center", "minWidth": "100px"},
        )
    else:
        slider = dcc.Slider(
            id={"type": "cutoff-slider", "elem": elem, "index": index},
            min=0,
            max=10_000_000,
            step=1000,
            value=0,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
        )
        suffix = html.Span("", style={"marginLeft": "4px", "minWidth": "16px"})

    return html.Div(
        [
            html.Label(label, style={**LABEL_STYLE, "fontWeight": "bold", "minWidth": "200px"}),
            html.Div(slider, style={"flex": "3"}, className="log-slider-wrap" if is_percent else ""),
            suffix,
        ],
        style={"display": "flex", "alignItems": "center", "gap": "8px",
               "marginBottom": "40px" if not is_percent else "8px"},
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
            _cutoff_element_row("足切り値", "e1", index),
            _cutoff_element_row("足切り通過確率", "e2", index, is_percent=True),
            _cutoff_element_row("残り必要ダメージ", "e3", index),
            _cutoff_element_row("残り通過確率", "e4", index, is_percent=True),
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
                    dcc.Input(id="target-damage", type="number", value=1000000, style={"width": "100%", "marginTop": "6px"}),
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
            html.Div(
                [
                    html.H1("ブルアカダメージ足切りシミュレータ(α版)", style={"marginBottom": "0"}),
                    html.Div(
                        [
                            html.Button(
                                "📖 マニュアル",
                                id="open-manual-btn",
                                n_clicks=0,
                                style={
                                    "background": "#4a90d9",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "padding": "6px 16px",
                                    "cursor": "pointer",
                                    "fontSize": "0.9rem",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                            
                            html.Span(
                                [
                                    "不具合報告、要望などは",
                                    html.A(
                                        "こちら",
                                        href="https://x.com/yankeiori",
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        style={"color": "#4a90d9"},
                                    ),
                                    "まで",
                                ],
                                style={
                                    "fontSize": "0.85rem",
                                    "color": "#555",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                        style={"marginLeft": "auto", "display": "flex", "alignItems": "center", "gap": "12px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
            ),
            # マニュアルモーダル
            html.Div(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong("マニュアル", style={"fontSize": "1.2rem"}),
                                html.Button(
                                    "✕",
                                    id="close-manual-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "auto",
                                        "background": "none",
                                        "border": "none",
                                        "cursor": "pointer",
                                        "fontSize": "1.3rem",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "borderBottom": "1px solid #ddd",
                                "paddingBottom": "8px",
                                "marginBottom": "12px",
                            },
                        ),
                        dcc.Markdown(_MANUAL_MD, style={"overflowY": "auto", "flex": "1"}),
                    ],
                    className="manual-modal-content",
                ),
                id="manual-modal",
                className="manual-modal-overlay",
                style={"display": "none"},
            ),
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
