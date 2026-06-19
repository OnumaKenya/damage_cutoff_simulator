"""積で表される分布 (HP依存ダメージ) の COS 法インタラクティブ可視化 (Dash アプリ)。

cos_app.py と同様に、基礎ダメージをカード形式で入力する。各カードの会心/非会心
ダメージ範囲・会心率・回避率・Hit 数が 1Hit の基礎ダメージ x_n の混合を与え、
HP 依存倍率 (β H_n + R_0) のもとで累積ダメージ
    D = H̃_1 (1 - Π_n (1 - β x_n))
の分布を COS 法 (experiments.product_cos) で準厳密に求め、MC と比較表示する。

カード編集まわりの UI とコールバックは edgeworth_app から再利用する
(make_card / manage_cards / _assemble_cards は "exp-*" の id を使う)。

実行:
    uv run python -m experiments.product_cos_app
    # → http://127.0.0.1:8062/
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.simulation import inverse_decay  # noqa: E402
from experiments.edgeworth_animation import Uniform, build_all_hits  # noqa: E402
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
from experiments.product_cos import (  # noqa: E402
    damage_dist,
    damage_moments,
    mc_damage_hits,
    support_bounds_hits,
    y_mixture,
)


# ---------------------------------------------------------------------------
# 既定値 (ミカ R1=2, R0=1 を基礎ダメージカード 1 枚で表現)
# ---------------------------------------------------------------------------
DEFAULT_H = 1_000_000.0
DEFAULT_H1 = 1_000_000.0
DEFAULT_R0 = 1.0
DEFAULT_R1 = 2.0
DEFAULT_N_MC = 20_000_000
DEFAULT_SEED = 42
DEFAULT_N_GRID = 1200

# 基礎ダメージカードの既定 (非会心 8000–12000 / 会心 16000–24000, 20 Hit)
_DEFAULT_CARD = {
    "crit_min": 16_000, "crit_max": 24_000,
    "normal_min": 8_000, "normal_max": 12_000,
    "hits": 20, "crit_rate": None, "evade_rate": None,
}


# ---------------------------------------------------------------------------
# プリセット (シナリオ): HP依存パラメータ + 一括設定 + カード列をまとめて読み込む
#
# hp0_input=True のプリセットは、カード値を「減衰後・敵HP=0 で観測される実ダメージ」で
# 持つ。HP=0 では HP依存倍率が R0 なので 観測値 = decay(R0 · x)。読込時はカードを生値の
# まま表示し (入力種類=hp0, 減衰=post_decay を選択)、計算時に
#   x = round( inverse_decay(観測値) / R0 )            (減衰を戻す → R0 で割る → 整数丸め)
# で基礎ダメージ x へ戻す (_hp0_to_base_cards)。
# 「減衰」(DAMAGE_FUNC; 大ダメージ圧縮) と「HP倍率 R0」は別軸であることに注意。
# R0=1 (ミカ等) なら割り算は恒等で従来と同じ、R0≠1 (ゲブラ等) で効く。
# ---------------------------------------------------------------------------
_BINAH_MIKA_CARDS = [
    {"crit_min": 48_340, "crit_max": 62_043,
     "normal_min": 48_340, "normal_max": 62_043, "hits": 4},
    {"crit_min": 87_350, "crit_max": 112_113,
     "normal_min": 87_350, "normal_max": 112_113, "hits": 6},
    {"crit_min": 873_496, "crit_max": 1_121_125,
     "normal_min": 873_496, "normal_max": 1_121_125, "hits": 1},
    {"crit_min": 136_801, "crit_max": 175_583,
     "normal_min": 136_801, "normal_max": 175_583, "hits": 10},
    {"crit_min": 1_368_011, "crit_max": 1_755_831,
     "normal_min": 1_368_011, "normal_max": 1_755_831, "hits": 1},
]

_GEBURA_FINA_CARDS = [
    {"crit_min": 885_129, "crit_max": 1_090_936,
     "normal_min": 177_132, "normal_max": 218_318, "hits": 1},
    {"crit_min": 774_488, "crit_max": 954_569,
     "normal_min": 154_991, "normal_max": 191_029, "hits": 6},
    {"crit_min": 885_129, "crit_max": 1_090_936,
     "normal_min": 177_132, "normal_max": 218_318, "hits": 1},
    {"crit_min": 774_488, "crit_max": 954_569,
     "normal_min": 154_991, "normal_max": 191_029, "hits": 6},
]

_GEBURA_KARIN_CARDS = [
    {"crit_min": 1_289_693, "crit_max": 1_446_886,
     "normal_min": 215_419, "normal_max": 241_675, "hits": 2},
    {"crit_min": 8_170_374, "crit_max": 8_716_035,
     "normal_min": 1_723_354, "normal_max": 1_933_403, "hits": 1},
    {"crit_min": 974_419, "crit_max": 1_093_185,
     "normal_min": 162_759, "normal_max": 182_596, "hits": 1},
    {"crit_min": 1_289_693, "crit_max": 1_446_886,
     "normal_min": 215_419, "normal_max": 241_675, "hits": 2},
    {"crit_min": 8_170_374, "crit_max": 8_716_035,
     "normal_min": 1_723_354, "normal_max": 1_933_403, "hits": 1},
]

PRESETS: dict = {
    "ビナーミカ (R1=2, R0=1)": {
        "H": 50_000_000.0,
        "H1": 49_700_000.0,
        "R0": 1.0,
        "R1": 2.0,
        "hp0_input": True,        # カード値は減衰考慮後・HP=0 のダメージ
        "global_crit": 100.0,
        "global_evade": 0.0,
        "cards": _BINAH_MIKA_CARDS,
    },
    "ゲブラフィーナ (R1=1, R0=1.5)": {
        "H": 175_000_000.0,
        "H1": 16_000_000.0,
        "R0": 1.5,
        "R1": 1.0,
        "hp0_input": True,        # カード値は減衰考慮後・HP=0 のダメージ
        "global_crit": 64.17,
        "global_evade": 0.0,
        "cards": _GEBURA_FINA_CARDS,
    },
    "ゲブラカリン (R1=1, R0=3)": {
        "H": 175_000_000.0,
        "H1": 35_000_000.0,
        "R0": 3.0,
        "R1": 1.0,
        "hp0_input": True,        # カード値は減衰考慮後・HP=0 のダメージ (R0=3 で割る)
        "global_crit": 68.22,
        "global_evade": 0.0,
        "cards": _GEBURA_KARIN_CARDS,
    },
}
DEFAULT_PRESET = "ビナーミカ (R1=2, R0=1)"


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
                            id="pc-preset",
                            options=[{"label": k, "value": k} for k in PRESETS],
                            value=DEFAULT_PRESET,
                            clearable=False,
                        ),
                    ),
                    html.Button(
                        "プリセット読込", id="pc-preset-load-btn", n_clicks=0,
                        style={"marginTop": "8px", "width": "100%", "cursor": "pointer"},
                    ),
                ],
                style={**SECTION_STYLE, "background": "#fffbe6"},
            ),
            # HP依存パラメータ
            html.Div(
                [
                    html.Strong("HP依存パラメータ"),
                    _labelled("敵最大HP (H)",
                              dcc.Input(id="pc-H", type="number", value=DEFAULT_H,
                                        style={"width": "100%"})),
                    _labelled("開始時HP (H1)",
                              dcc.Input(id="pc-H1", type="number", value=DEFAULT_H1,
                                        style={"width": "100%"})),
                    _labelled("HP=0 倍率 (R0)",
                              dcc.Input(id="pc-R0", type="number", value=DEFAULT_R0,
                                        step=0.1, style={"width": "100%"})),
                    _labelled("HP満タン倍率 (R1)",
                              dcc.Input(id="pc-R1", type="number", value=DEFAULT_R1,
                                        step=0.1, style={"width": "100%"})),
                ],
                style={**SECTION_STYLE, "background": "#fff0f5"},
            ),
            # 入力ダメージの種類 (HP依存倍率 R0 軸: R0 とは別物の「減衰」と混同しないよう分離)
            html.Div(
                [
                    html.Strong("入力ダメージの種類 (HP倍率)"),
                    html.Div(
                        "ゲーム表示のダメージは HP=0 で倍率 R0 が乗った値。"
                        "それを入力するなら ÷R0 で基礎 x に戻す。",
                        style={"fontSize": "0.72rem", "color": "#888", "margin": "4px 0"},
                    ),
                    dcc.RadioItems(
                        id="pc-input-type",
                        options=[
                            {"label": "HP=0 の実ダメージ（R0込み→÷R0）", "value": "hp0"},
                            {"label": "基礎ダメージ x（倍率なし）", "value": "base"},
                        ],
                        value="hp0",
                        style={"display": "flex", "flexDirection": "column",
                               "gap": "4px", "marginTop": "6px"},
                    ),
                ],
                style={**SECTION_STYLE, "background": "#fff0f0"},
            ),
            # ダメージ減衰 (DAMAGE_FUNC: 大ダメージを圧縮する区分線形カーブ。R0 とは無関係)
            html.Div(
                [
                    html.Strong("ダメージ減衰 (DAMAGE_FUNC)"),
                    html.Div(
                        "大ダメージを圧縮する減衰カーブ。HP倍率 R0 とは無関係。"
                        "入力がゲーム表示値なら「減衰後」を選ぶ。",
                        style={"fontSize": "0.72rem", "color": "#888", "margin": "4px 0"},
                    ),
                    dcc.RadioItems(
                        id="pc-damage-mode",
                        options=[
                            {"label": "減衰後（表示値・推奨）", "value": "post_decay"},
                            {"label": "減衰前（生値）", "value": "pre_decay"},
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
                    _labelled("会心率(%)",
                              dcc.Input(id="pc-global-crit", type="number",
                                        value=DEFAULT_GLOBAL_CRIT, style={"width": "100%"})),
                    _labelled("回避率(%)",
                              dcc.Input(id="pc-global-evade", type="number",
                                        value=DEFAULT_GLOBAL_EVADE, style={"width": "100%"})),
                ],
                style={**SECTION_STYLE, "background": "#f5f5ff"},
            ),
            # 比較設定
            html.Div(
                [
                    html.Strong("比較設定"),
                    _labelled("MC サンプル数",
                              dcc.Input(id="pc-n-mc", type="number", value=DEFAULT_N_MC,
                                        min=100_000, step=100_000, style={"width": "100%"})),
                    _labelled("乱数シード",
                              dcc.Input(id="pc-seed", type="number", value=DEFAULT_SEED,
                                        step=1, style={"width": "100%"})),
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
            html.H2("積で表される分布の COS 法 (HP依存ダメージ) — MC比較UI"),
            html.Div(
                "各カードの会心/非会心ダメージ範囲が 1Hit の「基礎ダメージ x」を与えます。"
                "HP依存倍率 (β·H + R0) は HP に応じて自動で掛かります。",
                style={"fontSize": "0.85rem", "color": "#666", "marginBottom": "12px"},
            ),
            html.Div(
                [
                    _sidebar(),
                    html.Div(
                        [
                            html.Div(
                                id="exp-cards-container",
                                children=[make_card(0, params=_DEFAULT_CARD)],
                            ),
                            html.Div(
                                [
                                    html.Button("+ カード追加", id="exp-add-btn", n_clicks=0),
                                    html.Button(
                                        "計算実行 (COS vs MC)",
                                        id="pc-run-btn",
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
                                ],
                                style={"marginBottom": "12px"},
                            ),
                            dcc.Loading(
                                [
                                    html.Div(id="pc-info",
                                             style={"fontSize": "0.9rem", "color": "#555",
                                                    "marginBottom": "8px"}),
                                    dcc.Graph(
                                        id="pc-graph",
                                        config={
                                            "displaylogo": False,
                                            "toImageButtonOptions": {
                                                "format": "png",
                                                "filename": "product_cos_mc",
                                                "scale": 2,   # 高解像度で書き出す
                                            },
                                        },
                                    ),
                                ],
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
        ],
        style={
            "maxWidth": "1200px",
            "margin": "0 auto",
            "padding": "20px",
            "fontFamily": "sans-serif",
        },
    )


app = Dash(__name__)
app.title = "積 COS ダメージ分布 (HP依存)"
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
# プリセット読込: HP依存パラメータ + 一括設定 + カード列をまとめて差し替える
# ---------------------------------------------------------------------------
_HP0_DAMAGE_FIELDS = ("crit_min", "crit_max", "normal_min", "normal_max")


def _hp0_to_base_cards(cards: list[dict], R0: float, undo_decay: bool) -> list[dict]:
    """「HP=0 の実ダメージ」で書かれたカードを、モデルの基礎ダメージ x のカードへ
    変換する。HP=0 では HP依存倍率が R0 なので 実ダメージ = (減衰) ∘ (R0·x)。
    各値を [入力が減衰後なら inverse_decay で減衰を戻し] → R0 で割り → 整数丸め する。
    減衰 (DAMAGE_FUNC) と HP倍率 (R0) は別軸: 前者は undo_decay、後者は /R0 が担う。
    変換後の値は生スケールの基礎 x なので、呼び出し側は pre_decay で build する。"""
    out: list[dict] = []
    for c in cards:
        nc = dict(c)
        for f in _HP0_DAMAGE_FIELDS:
            v = c.get(f)
            if v is not None:
                raw = inverse_decay(float(v)) if undo_decay else float(v)  # 減衰を戻す
                nc[f] = int(round(raw / R0))                               # /R0 → 丸め
        out.append(nc)
    return out


@app.callback(
    Output("exp-cards-container", "children", allow_duplicate=True),
    Output("exp-card-indices", "data", allow_duplicate=True),
    Output("exp-next-index", "data", allow_duplicate=True),
    Output("pc-H", "value"),
    Output("pc-H1", "value"),
    Output("pc-R0", "value"),
    Output("pc-R1", "value"),
    Output("pc-input-type", "value"),
    Output("pc-damage-mode", "value"),
    Output("pc-global-crit", "value"),
    Output("pc-global-evade", "value"),
    Input("pc-preset-load-btn", "n_clicks"),
    State("pc-preset", "value"),
    prevent_initial_call=True,
)
def load_preset(n_clicks, name):
    if not n_clicks or name not in PRESETS:
        return (no_update,) * 11
    preset = PRESETS[name]
    # カードはプリセットの生入力 (HP=0 実ダメージ等) をそのまま表示し、変換は計算時に
    # 入力モードに従って行う。hp0_input プリセットは「HP=0 実ダメージ・減衰後」表示なので
    # 入力種類=hp0, 減衰=post_decay を選ぶ。それ以外は基礎 x 扱い。
    if preset.get("hp0_input"):
        input_type, damage_mode = "hp0", "post_decay"
    else:
        input_type = "base"
        damage_mode = preset.get("damage_mode", "post_decay")
    cards = preset["cards"]
    children = [make_card(i, params=c, memo=c.get("memo", ""))
                for i, c in enumerate(cards)]
    indices = list(range(len(cards)))
    return (children, indices, len(cards),
            preset["H"], preset["H1"], preset["R0"], preset["R1"],
            input_type, damage_mode, preset["global_crit"], preset["global_evade"])


# ---------------------------------------------------------------------------
# 計算実行 (COS vs MC)
# ---------------------------------------------------------------------------
def _sanitize_components(comps: list[Uniform]) -> list[Uniform]:
    """lo<=hi を保証し、重み 0 を除く。"""
    out: list[Uniform] = []
    for c in comps:
        if c.weight <= 0:
            continue
        lo, hi = (c.lo, c.hi) if c.hi >= c.lo else (c.hi, c.lo)
        out.append(Uniform(c.weight, lo, hi))
    return out


@app.callback(
    Output("pc-graph", "figure"),
    Output("pc-info", "children"),
    Input("pc-run-btn", "n_clicks"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    State("exp-card-indices", "data"),
    State("pc-H", "value"),
    State("pc-H1", "value"),
    State("pc-R0", "value"),
    State("pc-R1", "value"),
    State("pc-input-type", "value"),
    State("pc-damage-mode", "value"),
    State("pc-global-crit", "value"),
    State("pc-global-evade", "value"),
    State("pc-n-mc", "value"),
    State("pc-seed", "value"),
    prevent_initial_call=True,
)
def run(n_clicks, values, ids, memo_values, memo_ids, indices,
        H, H1, R0, R1, input_type, damage_mode, global_crit, global_evade, n_mc, seed):
    if not n_clicks:
        return no_update, no_update

    # --- パラメータ検証 ---
    # R1 > R0 (β>0, 高HPほど高倍率) も R1 < R0 (β<0, 低HPほど高倍率) も扱える。
    # R1 == R0 は β=0 で HP 非依存 (積にならない) ため除外する。
    if None in (H, H1, R0, R1) or float(R1) == float(R0) or float(H) <= 0:
        return go.Figure(), "R1 ≠ R0 かつ H > 0 にしてください (R1=R0 は HP 非依存)。"
    H, H1, R0, R1 = float(H), float(H1), float(R0), float(R1)
    beta = (R1 - R0) / H
    Htil = H1 + R0 / beta

    # --- カード → Hit ごとの基礎ダメージ x 混合 ---
    # 入力種類=hp0 (HP=0 の実ダメージ) なら、減衰を戻して /R0 で基礎 x へ変換し、
    # 以降は pre_decay (生スケール) として扱う。減衰 (DAMAGE_FUNC) と HP倍率 (R0) を
    # 別軸として分離する: 前者は damage_mode==post_decay の inverse_decay、後者は /R0。
    cards = _assemble_cards(values, ids, memo_values, memo_ids, indices or [])
    if input_type == "hp0":
        cards = _hp0_to_base_cards(cards, R0, undo_decay=(damage_mode == "post_decay"))
        build_mode = "pre_decay"
    else:
        build_mode = damage_mode
    hit_mixtures, _card_boundaries = build_all_hits(
        cards, float(global_crit or 0), float(global_evade or 0), build_mode
    )
    if not hit_mixtures:
        return go.Figure(), "Hit がありません。カードを設定してください。"
    n_hits = len(hit_mixtures)

    base_per_hit = [_sanitize_components(hm) for hm in hit_mixtures]

    # --- x 混合 → Y = 1 - β x 混合 ---
    try:
        ymix_per_hit = [y_mixture(base, beta) for base in base_per_hit]
    except ValueError as e:
        return go.Figure(), f"設定エラー: {e} (基礎ダメージが大きすぎて HP が負になります)"

    # --- D の台を S の台から算出 ---
    # D = H̃_1(1 - e^S)。S∈[a,b] の両端を写し、β の符号で順序が変わるので min/max。
    a, b = support_bounds_hits(ymix_per_hit)
    d_end1 = Htil * (1.0 - math.exp(a))
    d_end2 = Htil * (1.0 - math.exp(b))
    d_lo, d_hi = min(d_end1, d_end2), max(d_end1, d_end2)
    pad = 0.04 * (d_hi - d_lo)
    d_grid = np.linspace(max(0.0, d_lo - pad), d_hi + pad, DEFAULT_N_GRID)

    f_D, F_D = damage_dist(ymix_per_hit, Htil, d_grid, verbose=True)

    # --- 標準化基準: 厳密な期待値・標準偏差 (MC 非依存)。z = (D − E[D]) / σ ---
    mean_exact, var_exact = damage_moments(ymix_per_hit, Htil)
    std_exact = math.sqrt(var_exact) if var_exact > 0 else 1.0
    z_grid = (d_grid - mean_exact) / std_exact

    # --- MC ---
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    n_mc = int(n_mc or DEFAULT_N_MC)
    samples = mc_damage_hits(base_per_hit, H, H1, R0, R1, n_mc, rng)
    z_samples = (samples - mean_exact) / std_exact
    sorted_s = np.sort(samples)

    # --- 共通の x グリッドで MC と COS を評価 (裾確率・誤差で共有) ---
    # 両側裾確率 min(P(D≤x), P(D>x)): 中央値を境に左は下側裾、右は上側裾。
    xs = np.linspace(d_grid[0], d_grid[-1], 600)
    zs = (xs - mean_exact) / std_exact
    cnt_le = np.searchsorted(sorted_s, xs, side="right")
    mc_cdf = cnt_le / n_mc
    mc_sf = np.minimum(mc_cdf, 1.0 - mc_cdf)           # MC 両側裾確率
    _, F_xs = damage_dist(ymix_per_hit, Htil, xs)
    cos_sf = np.minimum(F_xs, 1.0 - F_xs)
    # 信頼領域: 近い側の標本数が十分ある所だけ誤差を出す (cos_compare と同じ流儀)
    near_count = np.minimum(cnt_le, n_mc - cnt_le)
    reliable = near_count >= 50

    # --- 縦3段: 密度 / 両側裾確率 / MC との相対誤差 ---
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("標準化ダメージ密度 f(z)", "両側裾確率 min(P(D≤x), P(D>x))",
                        "COS の裾確率の相対誤差 (COS − MC) / MC"),
    )

    # 1段目: 密度 (標準化: x=z、密度は f_Z(z)=f_D(D)·σ)
    fig.add_trace(
        go.Histogram(x=z_samples, histnorm="probability density", nbinsx=160,
                     name="MC (真値)", marker_color="lightgray"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=z_grid, y=f_D * std_exact, mode="lines", name="COS 法 (DPあり)",
                   line=dict(color="magenta", width=2)),
        row=1, col=1,
    )

    # 2段目: 両側裾確率 (片対数、x=z)
    fig.add_trace(
        go.Scatter(x=zs, y=np.where(mc_sf > 0, mc_sf, np.nan), mode="lines",
                   name="MC 裾", line=dict(color="gray", width=2.5),
                   showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=zs, y=np.where(cos_sf > 0, cos_sf, np.nan), mode="lines",
                   name="COS 裾", line=dict(color="magenta", width=1.6, dash="dash"),
                   showlegend=False),
        row=2, col=1,
    )

    # 3段目: MC との相対誤差 (信頼領域のみ、x=z)
    rel_err = np.where(reliable & (mc_sf > 0), (cos_sf - mc_sf) / mc_sf, np.nan)
    fig.add_trace(
        go.Scatter(x=zs, y=rel_err, mode="lines", name="相対誤差",
                   line=dict(color="magenta", width=1.4), showlegend=False),
        row=3, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="gray", width=0.8), row=3, col=1)

    fig.update_yaxes(title_text="確率密度", row=1, col=1)
    fig.update_yaxes(type="log", range=[-6, 0], title_text="min(P(D≤x), P(D>x))", row=2, col=1)
    fig.update_yaxes(range=[-0.5, 0.5], title_text="相対誤差", row=3, col=1)
    fig.update_xaxes(title_text="標準化ダメージ z = (D − E[D]) / σ", row=3, col=1)
    fig.update_layout(height=900, bargap=0.02,
                      legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))

    # --- 健全性: 平均ダメージ ---
    mean_mc = float(samples.mean())
    mean_cos = float(np.trapezoid(d_grid * f_D, d_grid))
    rel = abs(mean_cos - mean_mc) / mean_mc * 100 if mean_mc else float("nan")
    info = (
        f"Hit数 {n_hits},  H̃_1 = {Htil:,.0f},  β = {beta:.3e},  "
        f"D の台 ≈ [{max(0.0, d_lo):,.0f}, {d_hi:,.0f}]  |  "
        f"平均ダメージ: MC={mean_mc:,.0f}  COS={mean_cos:,.0f}  (相対誤差 {rel:.3f}%)"
    )
    return fig, info


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8062))
    debug = "PORT" not in os.environ
    app.run(host="127.0.0.1", port=port, debug=debug)
