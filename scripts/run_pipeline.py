"""Run the complete pipeline: data -> db -> features -> models -> reports -> smoke."""
import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STEPS = [
    (["generate_data.py"], "generate synthetic data"),
    (["initialize_database.py"], "initialize DuckDB"),
    (["run_sql_analytics.py"], "execute and export SQL analytics"),
    (["build_features.py"], "build features"),
    (["train_all_models.py"], "train models"),
    (["evaluate_all_models.py"], "evaluate + reports"),
    (["run_monitoring.py"], "generate monitoring reports"),
    (["check_model_quality.py"], "enforce deployment quality gates"),
    (["smoke_test.py"], "smoke test"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    import pathlib
    here = pathlib.Path(__file__).parent
    for script, desc in STEPS:
        cmd = [sys.executable, str(here / script[0])]
        if args.small and script[0] == "generate_data.py":
            cmd.append("--small")
        logging.info(">>> %s", desc)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            logging.error("pipeline failed at: %s", desc)
            sys.exit(1)
    logging.info("PIPELINE COMPLETE")
