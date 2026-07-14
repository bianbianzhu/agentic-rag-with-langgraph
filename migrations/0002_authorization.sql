CREATE TABLE principals (
    principal_id text PRIMARY KEY CHECK (btrim(principal_id) <> '')
);

CREATE TABLE groups (
    group_id text PRIMARY KEY CHECK (btrim(group_id) <> '')
);

CREATE TABLE principal_group_memberships (
    principal_id text NOT NULL REFERENCES principals (principal_id)
        ON DELETE CASCADE,
    group_id text NOT NULL REFERENCES groups (group_id)
        ON DELETE CASCADE,
    PRIMARY KEY (principal_id, group_id)
);

CREATE TABLE access_grants (
    grant_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    knowledge_source text NOT NULL,
    document_id text NOT NULL,
    grant_type text NOT NULL CHECK (
        grant_type IN ('organization-public', 'principal', 'group')
    ),
    principal_id text REFERENCES principals (principal_id) ON DELETE CASCADE,
    group_id text REFERENCES groups (group_id) ON DELETE CASCADE,
    FOREIGN KEY (knowledge_source, document_id)
        REFERENCES source_documents (knowledge_source, document_id)
        ON DELETE CASCADE,
    CHECK (
        (grant_type = 'organization-public'
            AND principal_id IS NULL AND group_id IS NULL)
        OR (grant_type = 'principal'
            AND principal_id IS NOT NULL AND group_id IS NULL)
        OR (grant_type = 'group'
            AND principal_id IS NULL AND group_id IS NOT NULL)
    )
);

CREATE UNIQUE INDEX access_grants_organization_public_idx
    ON access_grants (knowledge_source, document_id)
    WHERE grant_type = 'organization-public';

CREATE UNIQUE INDEX access_grants_principal_idx
    ON access_grants (knowledge_source, document_id, principal_id)
    WHERE grant_type = 'principal';

CREATE UNIQUE INDEX access_grants_group_idx
    ON access_grants (knowledge_source, document_id, group_id)
    WHERE grant_type = 'group';

CREATE INDEX access_grants_source_document_idx
    ON access_grants (knowledge_source, document_id);
