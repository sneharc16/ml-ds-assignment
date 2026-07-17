-- Price by segment: realized vs listed, discount depth
SELECT make, model, body_type, city,
    COUNT(*) AS n_listings,
    ROUND(MEDIAN(listed_price), 0) AS median_listed_price,
    ROUND(MEDIAN(transaction_price), 0) AS median_transaction_price,
    ROUND(AVG((listed_price - transaction_price) / listed_price), 4) AS avg_discount,
    ROUND(AVG(CASE WHEN sold_flag THEN 1 ELSE 0 END), 4) AS sell_through
FROM cars
GROUP BY 1,2,3,4
HAVING COUNT(*) >= 3
ORDER BY n_listings DESC
LIMIT 200;
