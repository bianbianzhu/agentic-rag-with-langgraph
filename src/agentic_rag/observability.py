"""Client-side LangSmith secret exclusion."""

from collections.abc import Callable
import os
import re
from typing import Any

import langsmith as ls
from langsmith.anonymizer import (
    SECRET_PLACEHOLDER,
    StringNodeRule,
    create_secret_anonymizer,
)


_DATABASE_URL_RULE: StringNodeRule = {
    "pattern": re.compile(r"\bpostgres(?:ql)?://[^\s\"']+"),
    "replace": SECRET_PLACEHOLDER,
}


def trace_anonymizer() -> Callable[[Any], Any]:
    """Return the credential scrubber applied before trace upload."""

    return create_secret_anonymizer(extra_rules=[_DATABASE_URL_RULE])


def configure_tracing() -> None:
    """Configure automatic tracing with client-side secret scrubbing."""

    enabled = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
    ls.configure(
        client=ls.Client(anonymizer=trace_anonymizer()),
        enabled=enabled,
    )
