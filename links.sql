-- links.sql
-- All active ownership links (filtered to matched IDs in Python/Spark).
-- Placeholders: ${src}
SELECT
    l.Subsidiary_BvD_ID  AS Subsidiary,
    l.Shareholder_BvD_ID AS Shareholder
FROM ${src}.links_current l
WHERE l.Subsidiary_BvD_ID  IS NOT NULL
  AND l.Shareholder_BvD_ID IS NOT NULL
  AND l.Subsidiary_BvD_ID <> l.Shareholder_BvD_ID
