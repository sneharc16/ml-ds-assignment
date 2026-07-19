-- Monthly signup cohorts with activity and cumulative conversion retention.
WITH user_cohorts AS (
    SELECT user_id, DATE_TRUNC('month', signup_date)::DATE AS cohort_month
    FROM users
),
monthly_activity AS (
    SELECT
        s.user_id,
        DATE_TRUNC('month', s.session_start)::DATE AS activity_month,
        COUNT(DISTINCT s.session_id) AS sessions,
        MAX((e.event_name = 'booking_complete')::INTEGER) AS booked,
        MAX((e.event_name = 'purchase')::INTEGER) AS purchased
    FROM sessions s
    LEFT JOIN events e USING (session_id)
    GROUP BY 1, 2
),
cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_users
    FROM user_cohorts
    GROUP BY 1
)
SELECT
    c.cohort_month,
    DATE_DIFF('month', c.cohort_month, a.activity_month) AS months_since_signup,
    z.cohort_users,
    COUNT(DISTINCT a.user_id) AS active_users,
    ROUND(COUNT(DISTINCT a.user_id)::DOUBLE / z.cohort_users, 4) AS retention_rate,
    SUM(a.sessions) AS sessions,
    SUM(a.booked) AS users_booking,
    SUM(a.purchased) AS users_purchasing
FROM user_cohorts c
JOIN monthly_activity a USING (user_id)
JOIN cohort_size z USING (cohort_month)
WHERE a.activity_month >= c.cohort_month
GROUP BY 1, 2, 3
ORDER BY 1, 2;
