# Deployment runbook

DriveIntent ships separate FastAPI and Streamlit containers. Both verify the
DuckDB database, four supervised model bundles, the recommender bundle, and
artifact checksums before accepting traffic.

## Local Docker Compose

```bash
docker compose up --build
```

Compose explicitly enables deterministic demo bootstrapping and mounts shared
`data/` and `artifacts/` directories. The first start trains the small synthetic
profile. Later starts reuse the verified artifacts. The dashboard waits until
the API health check succeeds.

API: <http://localhost:8000/docs>  
Dashboard: <http://localhost:8501>

## Render blueprint

1. Push the repository to GitHub.
2. In Render, create a Blueprint and select `render.yaml`.
3. Review the two services and deploy.
4. Wait for the initial synthetic bootstrap and verify `/health` and
   `/_stcore/health`.

`DRIVEINTENT_BOOTSTRAP_DEMO=1` is appropriate for this self-contained portfolio
deployment. A real marketplace should disable it and obtain versioned data/model
artifacts from persistent object storage or a model registry.

## Release gate

```bash
make security-check
make lint
make test
make quality-gate
```

The quality gate checks 11 thresholds covering price MAE/R², booking and
sell-through discrimination/lift, and recommendation NDCG. Feature drift is
reported separately: it is an alert for investigation, not silently treated as
proof that model quality failed.

## Rollback and integrity

The model manifest records SHA-256 hashes and feature-contract hashes. Serving
refuses corrupted artifacts. Roll back by deploying a previously successful Git
commit together with its matching versioned artifact set. Do not copy model
files without their metadata and manifest.
