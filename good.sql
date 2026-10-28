-- good.sql
-- Reference set, all countries: real IDs (no '*'), non-individuals.
-- Placeholders: ${src}
SELECT
    e.BvD_ID_of_the_subsidiary_or_shareholder AS Id,
    e.Name                                    AS Name
FROM ${src}.entities e
WHERE e.BvD_ID_of_the_subsidiary_or_shareholder IS NOT NULL
  AND e.BvD_ID_of_the_subsidiary_or_shareholder NOT LIKE '%*%'
  AND e.Entity_type != 'I'
