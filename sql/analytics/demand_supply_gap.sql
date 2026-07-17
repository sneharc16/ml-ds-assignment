-- Demand-supply gap by city x body_type x transmission
WITH demand AS (
    SELECT e.city, f_body.filter_value AS body_type,
           COALESCE(f_trans.filter_value, 'Any') AS transmission,
           COUNT(DISTINCT e.session_id) AS qualified_sessions,
           COUNT(DISTINCT e.user_id) AS unique_users
    FROM events e
    JOIN events f_body ON f_body.session_id = e.session_id
        AND f_body.event_name = 'apply_filter' AND f_body.filter_name = 'body_type'
    LEFT JOIN events f_trans ON f_trans.session_id = e.session_id
        AND f_trans.event_name = 'apply_filter' AND f_trans.filter_name = 'transmission'
    WHERE e.event_name = 'view_search_results'
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
