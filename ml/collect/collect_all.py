"""Orchestrator: run all collectors in sequence."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLLECTORS = [
    ("international_results", "ml.collect.international_results", "download"),
    ("elo_ratings",           "ml.collect.elo_ratings",           "download"),
    ("transfermarkt",         "ml.collect.transfermarkt",         "download"),
    ("betting_odds",          "ml.collect.betting_odds",          "download"),
    ("squad_quality",         "ml.collect.squad_quality",         "download"),
    ("fifa_rankings",         "ml.collect.fifa_rankings",         "fetch"),
]


def main():
    failed = []
    for name, module_path, fn_name in COLLECTORS:
        log.info("=== Running collector: %s ===", name)
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, fn_name)()
        except Exception as exc:
            log.error("Collector %s failed: %s", name, exc)
            failed.append(name)

    if failed:
        log.error("Failed collectors: %s", failed)
        sys.exit(1)
    log.info("All collectors completed successfully.")


if __name__ == "__main__":
    main()
