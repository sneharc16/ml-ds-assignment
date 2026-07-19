-- Thirty-day first-touch, last-touch and linear multi-touch purchase attribution.
WITH purchases AS (
    SELECT
        e.event_id AS purchase_id,
        e.user_id,
        e.event_timestamp AS purchase_timestamp,
        c.transaction_price - c.acquisition_price AS gross_margin
    FROM events e
    JOIN cars c USING (car_id)
    WHERE e.event_name = 'purchase'
),
eligible_touches AS (
    SELECT
        p.*,
        s.session_id,
        s.campaign_id,
        s.session_start,
        ROW_NUMBER() OVER (PARTITION BY p.purchase_id ORDER BY s.session_start) AS first_touch_rank,
        ROW_NUMBER() OVER (PARTITION BY p.purchase_id ORDER BY s.session_start DESC) AS last_touch_rank,
        COUNT(*) OVER (PARTITION BY p.purchase_id) AS touch_count
    FROM purchases p
    JOIN sessions s
      ON s.user_id = p.user_id
     AND s.session_start <= p.purchase_timestamp
     AND s.session_start >= p.purchase_timestamp - INTERVAL 30 DAY
),
attributed AS (
    SELECT
        campaign_id,
        purchase_id,
        gross_margin,
        (first_touch_rank = 1)::INTEGER AS first_touch_credit,
        (last_touch_rank = 1)::INTEGER AS last_touch_credit,
        1.0 / touch_count AS linear_credit
    FROM eligible_touches
)
SELECT
    a.campaign_id,
    c.campaign_name,
    c.channel,
    COUNT(DISTINCT purchase_id) AS touched_purchases,
    ROUND(SUM(first_touch_credit), 3) AS first_touch_purchases,
    ROUND(SUM(last_touch_credit), 3) AS last_touch_purchases,
    ROUND(SUM(linear_credit), 3) AS linear_attributed_purchases,
    ROUND(SUM(gross_margin * first_touch_credit), 0) AS first_touch_margin,
    ROUND(SUM(gross_margin * last_touch_credit), 0) AS last_touch_margin,
    ROUND(SUM(gross_margin * linear_credit), 0) AS linear_attributed_margin
FROM attributed a
JOIN campaigns c USING (campaign_id)
GROUP BY 1, 2, 3
ORDER BY linear_attributed_margin DESC;
