import os
import sys
import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEV_PHASE_DIR = BASE_DIR / "5.Project Development Phase" / "Credit Card Approval Prediction"

sys.path.insert(0, str(DEV_PHASE_DIR))
os.chdir(str(DEV_PHASE_DIR))

APP_PATH = DEV_PHASE_DIR / "app.py"

module_globals = runpy.run_path(str(APP_PATH), run_name="credit_card_app")
app = module_globals["app"]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
