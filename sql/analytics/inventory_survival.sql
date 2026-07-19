-- Kaplan-Meier-style inventory survival curve by body type.
WITH intervals AS (
    SELECT UNNEST(GENERATE_SERIES(0, 180, 15)) AS interval_start
),
risk AS (
    SELECT
        c.body_type,
        i.interval_start,
        COUNT(*) FILTER (WHERE c.days_in_inventory >= i.interval_start) AS at_risk,
        COUNT(*) FILTER (
            WHERE c.sold_flag
              AND c.days_in_inventory >= i.interval_start
              AND c.days_in_inventory < i.interval_start + 15
        ) AS sold_in_interval
    FROM cars c
    CROSS JOIN intervals i
    GROUP BY 1, 2
),
hazards AS (
    SELECT *, sold_in_interval::DOUBLE / NULLIF(at_risk, 0) AS interval_hazard
    FROM risk
    WHERE at_risk > 0
)
SELECT
    body_type,
    interval_start,
    at_risk,
    sold_in_interval,
    ROUND(interval_hazard, 4) AS interval_hazard,
    ROUND(EXP(SUM(LN(GREATEST(1.0 - interval_hazard, 0.000001))) OVER (
        PARTITION BY body_type ORDER BY interval_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )), 4) AS survival_probability
FROM hazards
ORDER BY body_type, interval_start;
