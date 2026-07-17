-- CTR by list position (evidence of position bias in click labels)
SELECT list_position,
    COUNT(*) AS impressions,
    SUM(CASE WHEN clicked THEN 1 ELSE 0 END) AS clicks,
    ROUND(AVG(CASE WHEN clicked THEN 1.0 ELSE 0 END), 4) AS ctr,
    ROUND(AVG(CASE WHEN examined THEN 1.0 ELSE 0 END), 4) AS empirical_examination_rate
FROM impressions
GROUP BY 1
ORDER BY 1;
