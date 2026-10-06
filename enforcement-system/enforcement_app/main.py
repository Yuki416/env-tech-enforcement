from __future__ import annotations

import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enforcement_app.applog import configure_logging
from enforcement_app.config import load_config
from enforcement_app.gui import launch_gui


def main() -> int:
    config = load_config()
    # File-based logging so frozen exe clients (no console) still produce a
    # readable trace alongside enforce_watchdog.log and vlm_decisions.csv.
    configure_logging(log_path=config.log_path, level=config.log_level)
    logging.getLogger("enforcement_app").info(
        "enforcement_app starting; log_path=%s level=%s",
        config.log_path, config.log_level,
    )
    return launch_gui(config)


if __name__ == "__main__":
    raise SystemExit(main())
