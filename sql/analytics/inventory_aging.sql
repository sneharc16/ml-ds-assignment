-- Inventory aging buckets and sell-through
WITH aged AS (
    SELECT *,
        CASE WHEN days_in_inventory <= 15 THEN '0-15'
             WHEN days_in_inventory <= 30 THEN '16-30'
             WHEN days_in_inventory <= 45 THEN '31-45'
             WHEN days_in_inventory <= 60 THEN '46-60'
             WHEN days_in_inventory <= 90 THEN '61-90'
             ELSE '90+' END AS aging_bucket
    FROM cars
)
SELECT aging_bucket, city, body_type,
    COUNT(*) AS n_cars,
    ROUND(MEDIAN(days_in_inventory), 1) AS median_days_in_inventory,
    ROUND(AVG(CASE WHEN sold_flag THEN 1 ELSE 0 END), 4) AS sell_through_rate,
    ROUND(MEDIAN(listed_price), 0) AS median_listed_price
FROM aged
GROUP BY 1,2,3
ORDER BY aging_bucket, n_cars DESC;
