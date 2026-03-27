from dash import Dash

app = Dash(
    __name__,
    assets_folder="../assets",
    suppress_callback_exceptions=True,
    title="ブルアカダメージ足切りシミュレータ",
    update_title=None,
)
