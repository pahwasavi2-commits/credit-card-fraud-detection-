-- ==========================================================
-- CREDIT FRAUD EXPLORATORY DATA ANALYSIS (EDA)
-- ==========================================================

-- 1. DISTRIBUTION: Checking the massive class imbalance
SELECT Class, COUNT(*) as txn_count, 
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM creditcard), 3) as percent
FROM creditcard
GROUP BY Class;

-- 2. VALUE ANALYSIS: Comparison of transaction amounts
SELECT Class, 
       AVG(Amount) as avg_amount, 
       MAX(Amount) as max_amount,
       SUM(Amount) as total_volume
FROM creditcard
GROUP BY Class;

-- 3. TIME DYNAMICS: When does fraud peak? (Time is in seconds)
SELECT CAST(Time / 3600 AS INT) % 24 as hour_of_day, 
       COUNT(*) as fraud_count
FROM creditcard
WHERE Class = 1
GROUP BY hour_of_day
ORDER BY fraud_count DESC;

-- 4. ANOMALY DETECTION: Transactions 3 standard deviations above mean
SELECT * FROM creditcard
WHERE Amount > (SELECT AVG(Amount) + (3 * STDDEV(Amount)) FROM creditcard)
AND Class = 1;