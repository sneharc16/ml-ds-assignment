"""Re-export evaluation reports (metrics were computed during training)."""
import logging

from driveintent.analytics.analytics import ablation_study, export_reports
from driveintent.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = load_config()
    export_reports(cfg)
    ab = ablation_study(cfg)
    logging.info("ablation:\n%s", ab.to_string(index=False))
    # model comparison summary
    import pandas as pd
    rows = []
    for name in ("price", "booking", "sellthrough", "ranking"):
        p = cfg.artifacts / "metrics" / f"{name}_metrics.json"
        if p.exists():
            rows.append(dict(model=name, metrics_file=str(p.relative_to(cfg.root))))
    pd.DataFrame(rows).to_csv(cfg.artifacts / "reports" / "model_comparison.csv", index=False)
    logging.info("reports written to %s", cfg.artifacts / "reports")
