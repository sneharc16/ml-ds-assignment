-- Customer-360/RFM view using windows, filtered aggregates and observed preferences.
WITH user_activity AS (
    SELECT
        u.user_id,
        u.home_city,
        u.signup_date,
        u.maximum_budget,
        COUNT(DISTINCT s.session_id) AS lifetime_sessions,
        MAX(s.session_start) AS last_session_at,
        COUNT(DISTINCT s.session_id) FILTER (
            WHERE e.event_name IN ('view_inspection_report', 'calculate_emi', 'add_to_wishlist')
        ) AS qualified_sessions,
        COUNT(DISTINCT s.session_id) FILTER (WHERE e.event_name = 'booking_complete') AS bookings,
        COUNT(DISTINCT s.session_id) FILTER (WHERE e.event_name = 'purchase') AS purchases,
        SUM(COALESCE(e.engagement_time_seconds, 0)) AS engagement_seconds,
        MODE(c.body_type) FILTER (WHERE e.event_name = 'view_item') AS observed_body_preference,
        MODE(c.make) FILTER (WHERE e.event_name = 'view_item') AS observed_make_preference
    FROM users u
    LEFT JOIN sessions s USING (user_id)
    LEFT JOIN events e USING (session_id)
    LEFT JOIN cars c USING (car_id)
    GROUP BY 1, 2, 3, 4
),
scored AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY lifetime_sessions) AS frequency_quartile,
        NTILE(4) OVER (ORDER BY engagement_seconds) AS engagement_quartile,
        DATE_DIFF('day', last_session_at, MAX(last_session_at) OVER ()) AS recency_days
    FROM user_activity
)
SELECT *,
    CASE
        WHEN purchases > 0 THEN 'customer'
        WHEN bookings > 0 THEN 'booked_lead'
        WHEN qualified_sessions > 0 AND recency_days <= 30 THEN 'hot_lead'
        WHEN lifetime_sessions > 1 THEN 'returning_browser'
        ELSE 'new_or_dormant'
    END AS lifecycle_segment
FROM scored
ORDER BY purchases DESC, bookings DESC, engagement_seconds DESC;
