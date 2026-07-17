"""Train all (or selected) models."""
import argparse
import logging

import pandas as pd

from driveintent.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["price", "booking", "sellthrough",
                                        "recommender", "ranker", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config()
    todo = ([args.model] if args.model != "all"
            else ["price", "booking", "sellthrough", "recommender", "ranker"])
    if "price" in todo:
        from driveintent.models import price
        m = price.train(cfg)
        logging.info("price MAE=%.0f R2=%.3f", m["catboost"]["mae"], m["catboost"]["r2"])
    if "booking" in todo:
        from driveintent.models import classifiers
        m = classifiers.train(cfg, "booking")
        logging.info("booking PR-AUC=%.3f", m["catboost_calibrated"]["pr_auc"])
    if "sellthrough" in todo:
        from driveintent.models import classifiers
        m = classifiers.train(cfg, "sellthrough")
        logging.info("sellthrough PR-AUC=%.3f", m["catboost_calibrated"]["pr_auc"])
    if "recommender" in todo:
        from driveintent.models.recommender import RecommenderBundle
        cars = pd.read_parquet(cfg.raw_data / "cars.parquet")
        events = pd.read_parquet(cfg.raw_data / "events.parquet")
        RecommenderBundle(cfg).fit(cars, events).save()
        logging.info("recommender bundle saved")
    if "ranker" in todo:
        from driveintent.models import ranking
        m = ranking.train_ranker(cfg)
        logging.info("ranker NDCG@10=%.3f", m["ranker"]["ndcg_at_10"])

if __name__ == "__main__":
    main()
