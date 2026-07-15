# Reference System graph prototype

This static prototype lets students replay deterministic, synthetic Reference
System scenarios in a browser. It does not call OpenAI, LangSmith, PostgreSQL,
or any other external service, and it does not require API keys.

From the repository root, run:

```bash
python3 -m http.server 8000 --directory docs/prototype
```

Then open <http://127.0.0.1:8000/>.

The prototype uses only the files in this directory. The scenario identities,
documents, incidents, and traces are fictional teaching fixtures. This is a
design-level deterministic replay, not the live Reference System.
