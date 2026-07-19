-- Position-bias-corrected engagement by candidate segment.
WITH weighted AS (
    SELECT
        i.*,
        c.body_type,
        c.make,
        CASE i.list_position
            WHEN 1 THEN 0.95 WHEN 2 THEN 0.90 WHEN 3 THEN 0.84 WHEN 4 THEN 0.77
            WHEN 5 THEN 0.70 WHEN 6 THEN 0.64 WHEN 7 THEN 0.58 WHEN 8 THEN 0.53
            WHEN 9 THEN 0.48 WHEN 10 THEN 0.44 ELSE 0.25
        END AS propensity
    FROM impressions i
    JOIN cars c USING (car_id)
)
SELECT
    body_type,
    COUNT(*) AS impressions,
    ROUND(AVG(clicked::INTEGER), 4) AS naive_ctr,
    ROUND(SUM(clicked::INTEGER / propensity) / NULLIF(SUM(1.0 / propensity), 0), 4) AS ips_ctr,
    ROUND(SUM(booked::INTEGER / propensity) / NULLIF(SUM(1.0 / propensity), 0), 4) AS ips_booking_rate,
    ROUND(AVG(propensity), 4) AS average_examination_propensity
FROM weighted
GROUP BY 1
ORDER BY ips_booking_rate DESC, ips_ctr DESC;
