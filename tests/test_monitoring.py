import json

from driveintent.monitoring.monitor import build_drift_report, evaluate_quality_gates, run_monitoring


def test_monitoring_reports_feature_level_drift(pipeline):
    report = build_drift_report(pipeline)
    assert report["status"] in {"ok", "warning", "critical"}
    assert report["summary"]["feature_checks"] > 20
    assert {"model", "feature", "metric", "severity"} <= set(report["features"][0])


def test_quality_gates_pass_for_verified_small_profile(pipeline):
    report = evaluate_quality_gates(pipeline)
    assert report["total"] == 11
    assert report["status"] == "pass", [c for c in report["checks"] if not c["passed"]]


def test_combined_monitoring_status_is_persisted(pipeline):
    status = run_monitoring(pipeline)
    saved = json.loads((pipeline.artifacts / "monitoring" / "status.json").read_text())
    assert status == saved and saved["deployment_ready"] is True
