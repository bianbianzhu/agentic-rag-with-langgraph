DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'agentic_rag_retrieval'
    ) THEN
        CREATE ROLE agentic_rag_retrieval LOGIN;
    END IF;
END
$$;

ALTER ROLE agentic_rag_retrieval SET default_transaction_read_only = on;

GRANT USAGE ON SCHEMA public TO agentic_rag_retrieval;
GRANT SELECT ON
    knowledge_sources,
    source_documents,
    indexed_chunks,
    principals,
    groups,
    principal_group_memberships,
    access_grants
TO agentic_rag_retrieval;

REVOKE ALL ON FUNCTION source_document_is_authorized(text, text, text)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION source_document_is_authorized(text, text, text)
    TO agentic_rag_retrieval;
