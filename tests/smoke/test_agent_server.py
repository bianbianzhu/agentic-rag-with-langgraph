import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import httpx
import pytest
from langgraph_sdk import get_sync_client


REPOSITORY_ROOT = Path(__file__).parents[2]


def _unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_healthy(process: subprocess.Popen[str], base_url: str) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(f"Agent Server stopped during startup:\n{output}")
        try:
            with urlopen(f"{base_url}/ok", timeout=0.2) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.1)
    raise AssertionError("Agent Server did not become healthy within 15 seconds")


def test_agent_server_reuses_thread_for_two_turns() -> None:
    port = _unused_local_port()
    base_url = f"http://127.0.0.1:{port}"
    langgraph = Path(sys.executable).with_name("langgraph")
    environment = os.environ.copy()
    environment["LANGSMITH_TRACING"] = "false"
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("LANGSMITH_API_KEY", None)

    process = subprocess.Popen(
        [
            str(langgraph),
            "dev",
            "--no-browser",
            "--no-reload",
            "--port",
            str(port),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = ""
    try:
        _wait_until_healthy(process, base_url)
        client = get_sync_client(url=base_url)
        thread = client.threads.create()
        thread_id = thread["thread_id"]

        client.runs.wait(
            thread_id,
            "engineering_assistant",
            input={
                "thread": {},
                "current_turn": {
                    "turn_id": "turn-1",
                    "user_message": "First Turn",
                },
            },
            context={"principal_id": "alice"},
        )
        second_result = client.runs.wait(
            thread_id,
            "engineering_assistant",
            input={
                "current_turn": {
                    "turn_id": "turn-2",
                    "user_message": "Second Turn",
                }
            },
            context={"principal_id": "alice"},
        )
        final_state = client.threads.get_state(thread_id)

        assert isinstance(second_result, dict)
        assert isinstance(final_state, dict)
        assert isinstance(final_state["values"], dict)
        assert len(second_result["thread"]["turn_records"]) == 2
        assert len(final_state["values"]["thread"]["turn_records"]) == 2
        assert final_state["values"]["current_turn"] is None

        client.threads.delete(thread_id)
        with pytest.raises(httpx.HTTPStatusError) as deleted:
            client.threads.get(thread_id)
        assert deleted.value.response.status_code == 404
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=5)

    assert "Blocked deserialization" not in output
    assert "will be blocked in a future version" not in output
