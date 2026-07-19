from scripts.serve import missing_requirements


def test_serving_readiness_detects_complete_verified_artifacts(pipeline):
    assert missing_requirements() == []
