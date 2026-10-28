-- direct.sql
-- Entities whose name starts with a number (all countries), non-individuals,
-- with Information_provider from legal_info.
-- Placeholders: ${src}
SELECT
    e.BvD_ID_of_the_subsidiary_or_shareholder AS Id,
    e.Name                                    AS Name,
    l.Information_provider                    AS Information_provider
FROM ${src}.entities e
LEFT JOIN ${src}.legal_info l
       ON e.BvD_ID_of_the_subsidiary_or_shareholder = l.BvD_ID_number
WHERE (   e.Name RLIKE '^[0-9]{7}.* '
       OR e.Name RLIKE '^[0-9]{3}-[0-9]{3}'
       OR e.Name RLIKE '^[0-9]{3} [0-9]')
  AND e.Entity_type != 'I'
