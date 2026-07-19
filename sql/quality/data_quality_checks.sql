-- Zero violations means the warehouse contracts pass.
SELECT 'orphan_session_user' AS check_name,
       COUNT(*) AS violations
FROM sessions s LEFT JOIN users u USING (user_id)
WHERE u.user_id IS NULL
UNION ALL
SELECT 'orphan_event_session', COUNT(*)
FROM events e LEFT JOIN sessions s USING (session_id)
WHERE s.session_id IS NULL
UNION ALL
SELECT 'orphan_event_car', COUNT(*)
FROM events e LEFT JOIN cars c USING (car_id)
WHERE e.car_id IS NOT NULL AND c.car_id IS NULL
UNION ALL
SELECT 'event_before_session_start', COUNT(*)
FROM events e JOIN sessions s USING (session_id)
WHERE e.event_timestamp < s.session_start
UNION ALL
SELECT 'event_after_session_end', COUNT(*)
FROM events e JOIN sessions s USING (session_id)
WHERE e.event_timestamp > s.session_end + INTERVAL 1 SECOND
UNION ALL
SELECT 'duplicate_impression_key', COUNT(*)
FROM (
    SELECT session_id, car_id
    FROM impressions
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
) duplicates
UNION ALL
SELECT 'purchase_without_booking', COUNT(*)
FROM impressions
WHERE purchased AND NOT booked
UNION ALL
SELECT 'sold_car_missing_transaction_price', COUNT(*)
FROM cars
WHERE sold_flag AND transaction_price IS NULL
ORDER BY check_name;
