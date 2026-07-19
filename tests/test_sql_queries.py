from driveintent.data import load_database as db


def test_all_sql_files_run_nonempty(pipeline):
    cfg = pipeline
    for f in db.all_analytics_files(cfg):
        res = db.run_sql_file(cfg, f.name)
        assert len(res) > 0, f"{f.name} returned no rows"
        assert len(res.columns) > 1


def test_sql_data_quality_checks_have_zero_violations(pipeline):
    cfg = pipeline
    for f in db.all_quality_files(cfg):
        checks = db.run_quality_file(cfg, f.name)
        assert checks["violations"].sum() == 0, checks.to_dict(orient="records")


def test_semantic_marts_are_queryable(pipeline):
    cfg = pipeline
    for mart in ("mart_session_funnel", "mart_car_performance", "mart_campaign_daily"):
        result = db.query(cfg, f"SELECT * FROM {mart} LIMIT 5")
        assert len(result) > 0
