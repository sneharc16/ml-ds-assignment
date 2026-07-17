-- Brand / body switching: first viewed vs booked within session
WITH firsts AS (
    SELECT e.session_id,
        FIRST(c.make ORDER BY e.event_timestamp) AS first_viewed_make,
        FIRST(c.body_type ORDER BY e.event_timestamp) AS first_viewed_body
    FROM events e JOIN cars c USING (car_id)
    WHERE e.event_name = 'view_item'
    GROUP BY 1
),
booked AS (
    SELECT e.session_id, c.make AS booked_make, c.body_type AS booked_body
    FROM events e JOIN cars c USING (car_id)
    WHERE e.event_name = 'booking_complete'
)
SELECT f.first_viewed_body, b.booked_body, COUNT(*) AS n_sessions
FROM firsts f JOIN booked b USING (session_id)
GROUP BY 1,2 ORDER BY n_sessions DESC;
