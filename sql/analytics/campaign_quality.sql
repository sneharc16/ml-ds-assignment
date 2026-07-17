-- Campaign lead quality: spend, funnel outcomes, unit economics
WITH sess AS (
    SELECT s.campaign_id, s.session_id,
        MAX(CASE WHEN e.event_name IN ('view_inspection_report','calculate_emi','add_to_wishlist') THEN 1 ELSE 0 END) AS qualified,
        MAX(CASE WHEN e.event_name IN ('request_callback','book_test_drive') THEN 1 ELSE 0 END) AS lead,
        MAX(CASE WHEN e.event_name = 'booking_complete' THEN 1 ELSE 0 END) AS booked,
        MAX(CASE WHEN e.event_name = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM sessions s LEFT JOIN events e USING (session_id)
    GROUP BY 1,2
),
rev AS (
    SELECT e.campaign_id, SUM(c.transaction_price - c.acquisition_price) AS gross_margin
    FROM events e JOIN cars c USING (car_id)
    WHERE e.event_name = 'purchase'
    GROUP BY 1
),
agg AS (
    SELECT campaign_id, COUNT(*) AS sessions, SUM(qualified) AS qualified_sessions,
           SUM(lead) AS leads, SUM(booked) AS bookings, SUM(purchased) AS purchases
    FROM sess GROUP BY 1
)
SELECT c.campaign_id, c.campaign_name, c.channel,
    ROUND(c.daily_budget * GREATEST(DATE_DIFF('day', c.start_date, c.end_date), 1) / 100.0, 0) AS spend,
    a.sessions,
    ROUND(a.sessions * c.cost_per_click, 0) AS click_cost,
    a.qualified_sessions, a.leads, a.bookings, a.purchases,
    ROUND(a.sessions * c.cost_per_click / NULLIF(a.sessions, 0), 2) AS cost_per_session,
    ROUND(a.sessions * c.cost_per_click / NULLIF(a.leads, 0), 2) AS cost_per_lead,
    ROUND(a.sessions * c.cost_per_click / NULLIF(a.bookings, 0), 2) AS cost_per_booking,
    ROUND(a.bookings::DOUBLE / NULLIF(a.sessions, 0), 4) AS conversion_rate,
    COALESCE(r.gross_margin, 0) AS actual_revenue,
    ROUND(COALESCE(r.gross_margin, 0) / NULLIF(a.sessions * c.cost_per_click, 0), 3) AS actual_roas
FROM campaigns c
JOIN agg a USING (campaign_id)
LEFT JOIN rev r USING (campaign_id)
ORDER BY conversion_rate DESC;
