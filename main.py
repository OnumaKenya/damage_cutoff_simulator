from app import app as application
from app.frontend.layout import create_layout
import app.frontend.callbacks  # noqa: F401 - コールバック登録

application.layout = create_layout()

if __name__ == "__main__":
    application.run(debug=True)
