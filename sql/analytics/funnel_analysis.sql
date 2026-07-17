-- Funnel: session -> search results -> detail view -> deep engagement -> booking -> purchase
WITH sess AS (
    SELECT s.session_id, s.campaign_id, s.device_category, s.city, s.is_returning_user,
        MAX(CASE WHEN e.event_name = 'view_search_results' THEN 1 ELSE 0 END) AS saw_results,
        MAX(CASE WHEN e.event_name = 'view_item' THEN 1 ELSE 0 END) AS viewed_detail,
        MAX(CASE WHEN e.event_name = 'view_inspection_report' THEN 1 ELSE 0 END) AS viewed_inspection,
        MAX(CASE WHEN e.event_name = 'calculate_emi' THEN 1 ELSE 0 END) AS calculated_emi,
        MAX(CASE WHEN e.event_name = 'add_to_wishlist' THEN 1 ELSE 0 END) AS wishlisted,
        MAX(CASE WHEN e.event_name = 'booking_complete' THEN 1 ELSE 0 END) AS booked,
        MAX(CASE WHEN e.event_name = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM sessions s LEFT JOIN events e USING (session_id)
    GROUP BY 1,2,3,4,5
)
SELECT campaign_id, device_category, is_returning_user,
    COUNT(*) AS sessions,
    ROUND(AVG(saw_results), 4) AS session_to_results_rate,
    ROUND(SUM(viewed_detail)::DOUBLE / NULLIF(SUM(saw_results),0), 4) AS results_to_detail_rate,
    ROUND(SUM(viewed_inspection)::DOUBLE / NULLIF(SUM(viewed_detail),0), 4) AS detail_to_inspection_rate,
    ROUND(SUM(calculated_emi)::DOUBLE / NULLIF(SUM(viewed_detail),0), 4) AS detail_to_emi_rate,
    ROUND(SUM(wishlisted)::DOUBLE / NULLIF(SUM(viewed_detail),0), 4) AS detail_to_wishlist_rate,
    ROUND(SUM(booked)::DOUBLE / NULLIF(SUM(viewed_detail),0), 4) AS detail_to_booking_rate,
    ROUND(SUM(purchased)::DOUBLE / NULLIF(SUM(booked),0), 4) AS booking_to_purchase_rate
FROM sess
GROUP BY 1,2,3
ORDER BY sessions DESC;
