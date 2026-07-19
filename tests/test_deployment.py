from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_render_blueprint_defines_both_health_checked_services():
    config = yaml.safe_load((ROOT / "render.yaml").read_text())
    services = {service["name"]: service for service in config["services"]}
    assert services["driveintent-api"]["healthCheckPath"] == "/health"
    assert services["driveintent-dashboard"]["healthCheckPath"] == "/_stcore/health"
    assert all(service["autoDeployTrigger"] == "checksPass" for service in services.values())


def test_containers_use_verified_serving_entrypoint():
    api_dockerfile = (ROOT / "Dockerfile.api").read_text()
    dashboard_dockerfile = (ROOT / "Dockerfile.dashboard").read_text()
    assert 'scripts/serve.py", "api"' in api_dockerfile
    assert 'scripts/serve.py", "dashboard"' in dashboard_dockerfile
    assert "RUN python scripts/run_pipeline.py --small" in api_dockerfile
    assert "RUN python scripts/run_pipeline.py --small" in dashboard_dockerfile
