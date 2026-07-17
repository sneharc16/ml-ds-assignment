-- Session-level offline recommendation evaluation dataset (with graded relevance)
SELECT i.session_id, i.user_id, i.car_id, i.list_position AS rank_position,
    CASE WHEN i.purchased THEN 10 WHEN i.booked THEN 9 WHEN i.callback THEN 7
         WHEN i.wishlisted THEN 6 WHEN i.emi_calculated THEN 5
         WHEN i.viewed_inspection THEN 4 WHEN i.compared THEN 4
         WHEN i.viewed_gallery THEN 3 WHEN i.clicked THEN 2 ELSE 0 END AS relevance_label,
    i.clicked, i.wishlisted, i.booked, i.purchased,
    ROUND(1.0 / GREATEST(CASE i.list_position
        WHEN 1 THEN 0.95 WHEN 2 THEN 0.90 WHEN 3 THEN 0.84 WHEN 4 THEN 0.77
        WHEN 5 THEN 0.70 WHEN 6 THEN 0.64 WHEN 7 THEN 0.58 WHEN 8 THEN 0.53
        WHEN 9 THEN 0.48 WHEN 10 THEN 0.44 ELSE 0.25 END, 0.1), 3) AS propensity_weight
FROM impressions i;
