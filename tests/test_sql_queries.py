from driveintent.data import load_database as db


def test_all_sql_files_run_nonempty(pipeline):
    cfg = pipeline
    for f in db.all_analytics_files(cfg):
        res = db.run_sql_file(cfg, f.name)
        assert len(res) > 0, f"{f.name} returned no rows"
        assert len(res.columns) > 1
