import os

from dash import ALL, Input, Output, State
from app import app as application
from app.frontend.layout import create_layout
import app.frontend.callbacks  # noqa: F401 - コールバック登録

application.layout = create_layout()

# スライダー ドラッグ → % Input をリアルタイム更新（サーバー往復なし）
application.clientside_callback(
    """
    function(sliderValues, sliderIds, inputIds) {
        var out = [];
        // inputIds (e2,e4 のみ) の各要素に対応するスライダー値を探す
        for (var j = 0; j < inputIds.length; j++) {
            var found = false;
            for (var i = 0; i < sliderIds.length; i++) {
                if (sliderIds[i].elem === inputIds[j].elem &&
                    sliderIds[i].index === inputIds[j].index) {
                    var v = sliderValues[i];
                    if (v === null || v === undefined) { out.push(0.01); }
                    else {
                        var pct = Math.pow(10, (v / 100) - 2);
                        out.push(Math.round(pct * 100) / 100);
                    }
                    found = true;
                    break;
                }
            }
            if (!found) out.push(window.dash_clientside.no_update);
        }
        return out;
    }
    """,
    Output({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "value"),
    Input({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "value"),
    State({"type": "cutoff-slider", "elem": ALL, "index": ALL}, "id"),
    State({"type": "cutoff-pct-input", "elem": ALL, "index": ALL}, "id"),
)

# gunicorn から参照される WSGI サーバー (gunicorn main:server)
server = application.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = "PORT" not in os.environ
    application.run(host="0.0.0.0", port=port, debug=debug)
