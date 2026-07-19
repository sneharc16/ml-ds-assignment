# SQL analytics portfolio

DriveIntent uses DuckDB as a local analytical warehouse. SQL is part of the
tested application pipeline rather than presentation-only snippets.

## Layers

- `sql/ddl`: typed source-table contracts.
- `sql/marts`: reusable semantic views at session, vehicle, and campaign-day grain.
- `sql/analytics`: decision-focused analytical queries.
- `sql/quality`: executable warehouse contract checks.

Run every query and export reproducible CSV results with:

```bash
python scripts/run_sql_analytics.py
```

The command records result dimensions and execution times in
`artifacts/reports/sql/query_catalog.csv`. It exits non-zero when a SQL quality
check reports a violation.

## Advanced techniques demonstrated

| Query | Techniques | Business question |
| --- | --- | --- |
| `cohort_retention.sql` | date spine, cohort grain, conditional aggregation | Do acquisition cohorts return and convert? |
| `campaign_attribution.sql` | 30-day range join, `ROW_NUMBER`, window counts | How do first-, last- and linear-touch credit differ? |
| `funnel_transition_times.sql` | filtered aggregates, timestamp deltas, quantiles | Where does the journey slow down? |
| `inventory_survival.sql` | generated intervals, risk sets, cumulative window product | How quickly does inventory leave the platform? |
| `customer_360.sql` | filtered distincts, `MODE`, `NTILE`, recency scoring | Which users are customers, booked leads or hot leads? |
| `recommendation_ips.sql` | inverse propensity weighting | How much does position bias distort engagement? |
| `data_quality_checks.sql` | anti-joins and invariant checks | Are warehouse relationships and outcomes trustworthy? |

## Grain discipline

Each mart and query states its analytical grain. Session funnel metrics are
created before joining campaign dimensions, preventing event fan-out from
inflating conversions. Purchase attribution partitions by purchase ID before
campaign aggregation. Customer-360 uses distinct sessions for session-level
outcomes. These choices are as important as the SQL syntax.
