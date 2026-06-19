"""直接 FFT 反転 vs COS 法 比較用の独立した Dash ミニアプリ。

cos_app.py と同じ操作感で、ブラウザ上でカード追加・パラメータ編集・グローバル
設定・比較設定をして、「比較実行」ボタンで
`experiments.fft_compare.make_comparison_plot` を呼び、生成された比較プロット
(裾確率 PNG・密度 PNG) を画面に埋め込んで表示する。特性関数の「直接 FFT 反転」
(Gil-Pelaez/Carr-Madan 型) と COS 法 (準厳密基準) を MC 真値に対して比較する。

カード編集まわりの UI とコールバックは edgeworth_app から再利用する
(make_card / manage_cards / _assemble_cards は "exp-*" の id を使う)。
プリセットは cos_app と共通のものを流用する。

実行例:
    uv run python -m experiments.fft_app
    # → http://127.0.0.1:8062/
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile

from dash import ALL, Dash, Input, Output, State, dcc, html, no_update

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.edgeworth_animation import build_all_hits  # noqa: E402
from experiments.edgeworth_app import (  # noqa: E402
    DEFAULT_GLOBAL_CRIT,
    DEFAULT_GLOBAL_EVADE,
    PRIMARY,
    SECTION_STYLE,
    _assemble_cards,
    _labelled,
    make_card,
    manage_cards,
)
from experiments.cos_app import PRESETS, DEFAULT_PRESET  # noqa: E402
from experiments.cos_compare import Scenario  # noqa: E402
from experiments.fft_compare import make_comparison_plot  # noqa: E402


# ---------------------------------------------------------------------------
# 既定値
# ---------------------------------------------------------------------------
DEFAULT_N_MC = 20_000_000     # 真値 (点) 用の多数サンプル MC
DEFAULT_N_MC_SMALL = 10_000   # 線で重ねる少数サンプル MC
DEFAULT_N_GRID = 300
DEFAULT_SEED = 42
DEFAULT_N_FFT = 1 << 14       # 直接 FFT 反転の FFT サイズ (2 の冪)


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
                            id="fft-preset",
                            options=[{"label": k, "value": k} for k in PRESETS],
                            value=DEFAULT_PRESET,
                            clearable=False,
                        ),
                    ),
                    html.Button(
                        "プリセット読込", id="fft-preset-load-btn", n_clicks=0,
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
                        id="fft-damage-mode",
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
                        dcc.Input(id="fft-global-crit", type="number",
                                  value=DEFAULT_GLOBAL_CRIT,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "回避率(%)",
                        dcc.Input(id="fft-global-evade", type="number",
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
                        "MC サンプル数 (真値・点)",
                        dcc.Input(id="fft-n-mc", type="number",
                                  value=DEFAULT_N_MC, min=100_000, step=100_000,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "MC サンプル数 (少数・線)",
                        dcc.Input(id="fft-n-mc-small", type="number",
                                  value=DEFAULT_N_MC_SMALL, min=100, step=100,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "x グリッド点数",
                        dcc.Input(id="fft-n-grid", type="number",
                                  value=DEFAULT_N_GRID, min=50, max=1000, step=50,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "FFT サイズ N (2の冪)",
                        dcc.Input(id="fft-n-fft", type="number",
                                  value=DEFAULT_N_FFT, min=256, step=256,
                                  style={"width": "100%"}),
                    ),
                    _labelled(
                        "乱数シード",
                        dcc.Input(id="fft-seed", type="number",
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
            html.H2("直接 FFT 反転 vs COS法  比較UI"),
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
                                        id="fft-run-btn",
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
                                        " 注: 真値サンプル数 × 総Hit数 が大きいとしばらく待たされます。",
                                        style={"marginLeft": "12px", "fontSize": "0.8rem", "color": "#666"},
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "裾確率PNGをダウンロード",
                                        id="fft-dl-btn",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "密度PNGをダウンロード",
                                        id="fft-dl-density-btn",
                                        n_clicks=0,
                                        disabled=True,
                                        style={"marginLeft": "12px"},
                                    ),
                                    html.Button(
                                        "収束PNGをダウンロード",
                                        id="fft-dl-conv-btn",
                                        n_clicks=0,
                                        disabled=True,
                                        style={"marginLeft": "12px"},
                                    ),
                                ],
                                style={"marginTop": "12px"},
                            ),
                            dcc.Loading(
                                html.Div(id="fft-output", style={"marginTop": "12px"}),
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
            dcc.Store(id="fft-png-b64"),
            dcc.Store(id="fft-density-b64"),
            dcc.Store(id="fft-conv-b64"),
            dcc.Download(id="fft-dl"),
            dcc.Download(id="fft-dl-density"),
            dcc.Download(id="fft-dl-conv"),
        ],
        style={
            "maxWidth": "1200px",
            "margin": "0 auto",
            "padding": "20px",
            "fontFamily": "sans-serif",
        },
    )


app = Dash(__name__)
app.title = "直接FFT反転 vs COS法 比較"
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
    Output("fft-global-crit", "value"),
    Output("fft-global-evade", "value"),
    Output("fft-damage-mode", "value"),
    Input("fft-preset-load-btn", "n_clicks"),
    State("fft-preset", "value"),
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
    Output("fft-output", "children"),
    Output("fft-png-b64", "data"),
    Output("fft-dl-btn", "disabled"),
    Output("fft-density-b64", "data"),
    Output("fft-dl-density-btn", "disabled"),
    Output("fft-conv-b64", "data"),
    Output("fft-dl-conv-btn", "disabled"),
    Input("fft-run-btn", "n_clicks"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    State("exp-card-indices", "data"),
    State("fft-global-crit", "value"),
    State("fft-global-evade", "value"),
    State("fft-damage-mode", "value"),
    State("fft-n-mc", "value"),
    State("fft-n-mc-small", "value"),
    State("fft-n-grid", "value"),
    State("fft-n-fft", "value"),
    State("fft-seed", "value"),
    prevent_initial_call=True,
)
def run(n_clicks, values, ids, memo_values, memo_ids, indices,
        global_crit, global_evade, damage_mode,
        n_mc, n_mc_small, n_grid, n_fft, seed):
    if not n_clicks:
        return (no_update,) * 7

    cards = _assemble_cards(values, ids, memo_values, memo_ids, indices or [])
    if not cards:
        return (html.Div("カードが0枚です。", style={"color": "red"}),
                no_update, True, no_update, True, no_update, True)

    png_path = density_path = conv_path = ""
    try:
        hit_mixtures, _ = build_all_hits(
            cards, float(global_crit or 0), float(global_evade or 0), damage_mode
        )
        if not hit_mixtures:
            return (html.Div("Hit がありません。", style={"color": "red"}),
                    no_update, True, no_update, True, no_update, True)
        sc = Scenario("", hit_mixtures)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            density_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            conv_path = tmp.name
        make_comparison_plot(
            sc,
            int(n_mc or DEFAULT_N_MC),
            int(n_mc_small or DEFAULT_N_MC_SMALL),
            int(seed if seed is not None else DEFAULT_SEED),
            int(n_grid or DEFAULT_N_GRID),
            png_path,
            density_output_path=density_path,
            n_fft=int(n_fft or DEFAULT_N_FFT),
            convergence_output_path=conv_path,
        )
        png_b64 = _b64_of_file(png_path)
        density_b64 = _b64_of_file(density_path)
        conv_b64 = _b64_of_file(conv_path)
    except Exception as e:
        return (html.Div(f"エラー: {e!s}",
                         style={"color": "red", "whiteSpace": "pre-wrap"}),
                no_update, True, no_update, True, no_update, True)
    finally:
        for p in (png_path, density_path, conv_path):
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
            html.H4("収束の比較 (CF 評価点数 vs 最大相対誤差)",
                    style={"marginTop": "16px"}),
            html.Img(
                src=f"data:image/png;base64,{conv_b64}",
                style={"maxWidth": "100%", "border": "1px solid #ddd"},
            ),
        ]
    )
    return output, png_b64, False, density_b64, False, conv_b64, False


@app.callback(
    Output("fft-dl", "data"),
    Input("fft-dl-btn", "n_clicks"),
    State("fft-png-b64", "data"),
    prevent_initial_call=True,
)
def download_png(n_clicks, png_b64):
    if not n_clicks or not png_b64:
        return no_update
    return dict(content=png_b64, filename="fft_tail.png", base64=True)


@app.callback(
    Output("fft-dl-density", "data"),
    Input("fft-dl-density-btn", "n_clicks"),
    State("fft-density-b64", "data"),
    prevent_initial_call=True,
)
def download_density(n_clicks, density_b64):
    if not n_clicks or not density_b64:
        return no_update
    return dict(content=density_b64, filename="fft_density.png", base64=True)


@app.callback(
    Output("fft-dl-conv", "data"),
    Input("fft-dl-conv-btn", "n_clicks"),
    State("fft-conv-b64", "data"),
    prevent_initial_call=True,
)
def download_conv(n_clicks, conv_b64):
    if not n_clicks or not conv_b64:
        return no_update
    return dict(content=conv_b64, filename="fft_convergence.png", base64=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8062))
    debug = "PORT" not in os.environ
    app.run(host="127.0.0.1", port=port, debug=debug)
