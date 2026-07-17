-- Demand-supply gap by city x body_type x transmission
WITH session_filters AS (
    SELECT session_id,
           ARG_MAX(filter_value, event_timestamp) FILTER (WHERE filter_name = 'body_type') AS body_type,
           ARG_MAX(filter_value, event_timestamp) FILTER (WHERE filter_name = 'transmission') AS transmission
    FROM events
    WHERE event_name = 'apply_filter'
    GROUP BY session_id
),
demand AS (
    SELECT e.city, f.body_type,
           COALESCE(f.transmission, 'Any') AS transmission,
           COUNT(*) AS qualified_sessions,
           COUNT(DISTINCT e.user_id) AS unique_users
    FROM events e
    JOIN session_filters f USING (session_id)
    WHERE e.event_name = 'view_search_results' AND f.body_type IS NOT NULL
    GROUP BY 1,2,3
),
supply AS (
    SELECT city, body_type, transmission,
           COUNT(*) AS available_cars, MEDIAN(listed_price) AS median_price
    FROM cars WHERE NOT sold_flag
    GROUP BY 1,2,3
)
SELECT d.city, d.body_type, d.transmission,
    d.qualified_sessions, d.unique_users,
    COALESCE(s.available_cars, 0) AS available_cars,
    ROUND(COALESCE(s.median_price, 0), 0) AS median_listed_price,
    ROUND(d.qualified_sessions::DOUBLE / (COALESCE(s.available_cars, 0) + 1), 2) AS demand_supply_gap
FROM demand d
LEFT JOIN supply s ON s.city = d.city AND s.body_type = d.body_type
    AND (d.transmission = 'Any' OR s.transmission = d.transmission)
ORDER BY demand_supply_gap DESC
LIMIT 100;
