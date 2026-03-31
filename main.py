import os

from dash import ALL, Input, Output, State
from app import app as application
from app.frontend.layout import create_layout
import app.frontend.callbacks  # noqa: F401 - コールバック登録

application.layout = create_layout()

# ===========================================================================
# クライアントサイドコールバック (assets/simulation.js の関数を参照)
# ===========================================================================

# --- ドラッグ順同期 ---
application.clientside_callback(
    "dash_clientside.sim.syncDragOrder",
    Output("sorted-indices", "data"),
    Input("drag-order", "data"),
    State("card-indices", "data"),
    prevent_initial_call=True,
)

# --- 世代カウンタ ---
application.clientside_callback(
    "dash_clientside.sim.incrementGeneration",
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

# --- 一括適用 ---
application.clientside_callback(
    "dash_clientside.sim.applyGlobal",
    Output({"type": "param", "param": "crit_rate", "index": ALL}, "value"),
    Output({"type": "param", "param": "evade_rate", "index": ALL}, "value"),
    Input("apply-global-btn", "n_clicks"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State({"type": "param", "param": "crit_rate", "index": ALL}, "value"),
    prevent_initial_call=True,
)

# --- シミュレーション実行 ---
application.clientside_callback(
    "dash_clientside.sim.runSimulation",
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

# --- 足切り計算 ---
application.clientside_callback(
    "dash_clientside.sim.computeCutoff",
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

# --- 足切りスライダー操作 ---
application.clientside_callback(
    "dash_clientside.sim.onSliderChange",
    Output("cutoff-values-store", "data", allow_duplicate=True),
    Input({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "value"),
    State("cutoff-values-store", "data"),
    State("cutoff-dist-store", "data"),
    State("target-damage", "value"),
    State({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)

# --- 足切りスライダー表示更新 ---
application.clientside_callback(
    "dash_clientside.sim.updateCutoffDisplay",
    Output({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "value"),
    Output({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "min"),
    Output({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "max"),
    Input("cutoff-values-store", "data"),
    State("cutoff-dist-store", "data"),
    State({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)

# --- % Input 直接入力 ---
application.clientside_callback(
    "dash_clientside.sim.onPctInputChange",
    Output("cutoff-values-store", "data", allow_duplicate=True),
    Input({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "value"),
    State("cutoff-values-store", "data"),
    State("cutoff-dist-store", "data"),
    State("target-damage", "value"),
    State({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)

# --- Store → % Input 表示同期 ---
application.clientside_callback(
    "dash_clientside.sim.updatePctInputDisplay",
    Output({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "value", allow_duplicate=True),
    Input("cutoff-values-store", "data"),
    State({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)

# --- スライダー → % Input リアルタイム同期 (サーバー往復なし) ---
application.clientside_callback(
    "dash_clientside.sim.sliderToPctSync",
    Output({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "value"),
    Input({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "value"),
    State({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "id"),
    State({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "id"),
)

# --- マニュアルモーダル開閉 ---
application.clientside_callback(
    "dash_clientside.sim.toggleManualModal",
    Output("manual-modal", "style"),
    Input("open-manual-btn", "n_clicks"),
    Input("close-manual-btn", "n_clicks"),
    prevent_initial_call=True,
)

# gunicorn から参照される WSGI サーバー (gunicorn main:server)
server = application.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = "PORT" not in os.environ
    application.run(host="0.0.0.0", port=port, debug=debug)
