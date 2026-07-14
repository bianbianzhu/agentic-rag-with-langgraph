CREATE FUNCTION source_document_is_authorized (
    input_principal_id text,
    input_knowledge_source text,
    input_document_id text
)
RETURNS boolean
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM principals AS authorized_principal
        WHERE authorized_principal.principal_id = input_principal_id
    )
    AND EXISTS (
        SELECT 1
        FROM access_grants AS matching_grant
        WHERE matching_grant.knowledge_source = input_knowledge_source
          AND matching_grant.document_id = input_document_id
          AND (
              matching_grant.grant_type = 'organization-public'
              OR (
                  matching_grant.grant_type = 'principal'
                  AND matching_grant.principal_id = input_principal_id
              )
              OR (
                  matching_grant.grant_type = 'group'
                  AND EXISTS (
                      SELECT 1
                      FROM principal_group_memberships AS membership
                      WHERE membership.principal_id = input_principal_id
                        AND membership.group_id = matching_grant.group_id
                  )
              )
          )
    )
$$;
