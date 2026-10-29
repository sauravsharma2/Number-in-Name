-- prev_names.sql
-- Previous company names (filtered to the needed Good IDs in main).
-- Placeholders: ${src}
SELECT
    l.BvD_ID_number         AS Id,
    l.Previous_company_name AS Previous_Name
FROM ${src}.legal_info l
WHERE l.Previous_company_name IS NOT NULL
  AND trim(l.Previous_company_name) <> ''
