"""Session-scoped tiny fixture: the small pipeline runs once for all tests."""
import pytest

from driveintent.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tables(cfg):
    """Small-profile generated tables. Regenerates deterministically."""
    from driveintent.data.generate import build_all
    return build_all(cfg, small=True)


@pytest.fixture(scope="session")
def pipeline(cfg, tables):
    """Runs db init + features + all model training once on small data."""
    import pandas as pd

    from driveintent.data import load_database as db
    from driveintent.features.build import build_all as build_features
    from driveintent.models import classifiers, price, ranking
    from driveintent.models.recommender import RecommenderBundle

    db.initialize(cfg)
    build_features(cfg)
    price.train(cfg)
    classifiers.train(cfg, "booking")
    classifiers.train(cfg, "sellthrough")
    cars = pd.read_parquet(cfg.raw_data / "cars.parquet")
    events = pd.read_parquet(cfg.raw_data / "events.parquet")
    train_end = pd.Timestamp(cfg.get("splits", "train_end")) + pd.Timedelta(days=1)
    events_train = events[pd.to_datetime(events["event_timestamp"]) < train_end]
    RecommenderBundle(cfg).fit(cars, events_train).save()
    ranking.train_ranker(cfg)
    from driveintent.monitoring.monitor import run_monitoring
    run_monitoring(cfg)
    return cfg
