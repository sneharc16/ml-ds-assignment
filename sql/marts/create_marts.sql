-- Reusable semantic marts. Views preserve reproducibility and avoid duplicated logic.
CREATE OR REPLACE VIEW mart_session_funnel AS
SELECT
    s.session_id,
    s.user_id,
    s.campaign_id,
    s.session_start,
    s.device_category,
    s.city,
    s.is_returning_user,
    COUNT(e.event_id) AS event_count,
    SUM(COALESCE(e.engagement_time_seconds, 0)) AS engagement_seconds,
    MAX((e.event_name = 'view_search_results')::INTEGER) AS saw_results,
    MAX((e.event_name = 'view_item')::INTEGER) AS viewed_car,
    MAX((e.event_name IN ('view_inspection_report', 'calculate_emi', 'add_to_wishlist'))::INTEGER) AS deep_engagement,
    MAX((e.event_name IN ('request_callback', 'book_test_drive'))::INTEGER) AS generated_lead,
    MAX((e.event_name = 'booking_complete')::INTEGER) AS booked,
    MAX((e.event_name = 'purchase')::INTEGER) AS purchased
FROM sessions s
LEFT JOIN events e USING (session_id)
GROUP BY ALL;

CREATE OR REPLACE VIEW mart_car_performance AS
SELECT
    c.car_id,
    c.make,
    c.model,
    c.body_type,
    c.city,
    c.listed_price,
    c.inspection_score,
    c.days_in_inventory,
    c.sold_flag,
    COUNT(e.event_id) FILTER (WHERE e.event_name = 'view_item') AS detail_views,
    COUNT(e.event_id) FILTER (WHERE e.event_name = 'view_inspection_report') AS inspection_views,
    COUNT(e.event_id) FILTER (WHERE e.event_name = 'add_to_wishlist') AS wishlists,
    COUNT(e.event_id) FILTER (WHERE e.event_name = 'booking_complete') AS bookings,
    COUNT(e.event_id) FILTER (WHERE e.event_name = 'purchase') AS purchases
FROM cars c
LEFT JOIN events e USING (car_id)
GROUP BY ALL;

CREATE OR REPLACE VIEW mart_campaign_daily AS
SELECT
    s.session_start::DATE AS activity_date,
    s.campaign_id,
    c.campaign_name,
    c.channel,
    COUNT(*) AS sessions,
    SUM(f.deep_engagement) AS qualified_sessions,
    SUM(f.generated_lead) AS leads,
    SUM(f.booked) AS bookings,
    SUM(f.purchased) AS purchases,
    COUNT(*) * MAX(c.cost_per_click) AS estimated_click_cost
FROM sessions s
JOIN mart_session_funnel f USING (session_id)
JOIN campaigns c ON c.campaign_id = s.campaign_id
GROUP BY ALL;
