# Verified evaluation results

Run date: 2026-07-19  
Profile: deterministic small synthetic dataset, seed 42  
Split policy: chronological train / validation / test windows

## Dataset

| Table | Rows |
| --- | ---: |
| Cars | 300 |
| Users | 200 |
| Campaigns | 5 |
| Sessions | 500 |
| Events | 9,054 |
| Impressions | 7,131 |

## Held-out results

| Model | Metrics |
| --- | --- |
| Price champion | MAE ₹88,358.41; RMSE ₹120,383.25; R² 0.858508; P10–P90 conformal coverage 94.12% |
| Booking champion | ROC-AUC 0.740096; PR-AUC 0.098070; Brier 0.030925; top-5% lift 4.504× |
| Sell-through champion | ROC-AUC 0.554429; PR-AUC 0.347065; Brier 0.192124; top-5% lift 1.918× |
| Ranking champion | NDCG@10 0.628236; MAP@10 0.669898; Recall@10 0.878159; coverage@10 94.38% |

Price blend weights were selected on validation data: Extra Trees 0.70,
CatBoost RMSE 0.20, and CatBoost MAE 0.10. Ranking validation selected the
session-intent score and rejected a weaker raw learning-to-rank output. The raw
ranker achieved NDCG@10 0.465102 on the test window.

## Verification

- 14/14 SQL analytics queries executed with non-empty results.
- SQL warehouse quality checks reported zero violations.
- 11/11 configured model-quality gates passed.
- 71/71 automated tests passed.
- Python bytecode compilation, Ruff lint, and credential scanning passed.

Feature drift reported `critical` on this intentionally time-evolving synthetic
test window (113 checks: 33 critical, 23 warning, 57 okay). This is surfaced
explicitly in monitoring and does not invalidate the held-out metrics. In a real
system it would trigger segment review, retraining analysis, and a controlled
champion/challenger evaluation.

All figures are synthetic and should not be represented as CARS24 production
performance.
