-- Time-to-next-step diagnostics with conditional aggregation and quantiles.
WITH steps AS (
    SELECT
        s.session_id,
        s.device_category,
        s.is_returning_user,
        MIN(e.event_timestamp) FILTER (WHERE e.event_name = 'view_search_results') AS results_at,
        MIN(e.event_timestamp) FILTER (WHERE e.event_name = 'view_item') AS detail_at,
        MIN(e.event_timestamp) FILTER (WHERE e.event_name = 'view_inspection_report') AS inspection_at,
        MIN(e.event_timestamp) FILTER (WHERE e.event_name = 'booking_complete') AS booking_at
    FROM sessions s
    LEFT JOIN events e USING (session_id)
    GROUP BY 1, 2, 3
),
durations AS (
    SELECT *,
        DATE_DIFF('second', results_at, detail_at) AS results_to_detail_seconds,
        DATE_DIFF('second', detail_at, inspection_at) AS detail_to_inspection_seconds,
        DATE_DIFF('second', detail_at, booking_at) AS detail_to_booking_seconds
    FROM steps
)
SELECT
    device_category,
    is_returning_user,
    COUNT(*) AS sessions,
    COUNT(results_to_detail_seconds) AS detail_transitions,
    ROUND(QUANTILE_CONT(results_to_detail_seconds, 0.5), 1) AS median_results_to_detail_seconds,
    ROUND(QUANTILE_CONT(results_to_detail_seconds, 0.9), 1) AS p90_results_to_detail_seconds,
    ROUND(QUANTILE_CONT(detail_to_inspection_seconds, 0.5), 1) AS median_detail_to_inspection_seconds,
    ROUND(QUANTILE_CONT(detail_to_booking_seconds, 0.5), 1) AS median_detail_to_booking_seconds
FROM durations
GROUP BY 1, 2
ORDER BY sessions DESC;
