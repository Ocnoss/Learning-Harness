"""Stub tape smoke test (offline) -- record -> save_tape -> replay -> miss.

Covers the recording/replay path that layer3 only exercises in memory:
    RecordingLLMClient  wraps StubLLMClient, records one request/response;
    save_tape()         persists the tape as JSONL;
    RecordedLLMClient   replays by fingerprint (hit) and raises TapeMissError
                        for an unknown fingerprint (no silent network fallback);
    LLMClientFactory    can instantiate both "stub" and "recorded" providers.

Fingerprint rule (core.stub_llm._compute_fingerprint):
    sha256(last user message content).hexdigest()[:16]

Purely offline: no API key, no network, no wall-clock. Temp files are removed
in a finally block.

Run:
    D:\\miniconda\\python.exe tests\\context_compiler\\test_stub_tape.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core.stub_llm  # noqa: E402,F401 -- triggers factory registration
from core.llm_client import LLMClientFactory  # noqa: E402
from core.stub_llm import (  # noqa: E402
    RecordedLLMClient,
    RecordingLLMClient,
    StubLLMClient,
    TapeMissError,
)

PROMPT = "give one next-step study advice for xingce"
RESPONSE = "[stub] focus on data-analysis speed drills"
OTHER_PROMPT = "a totally different prompt that was never recorded"


def _fingerprint(messages: list[dict]) -> str:
    """Local re-implementation of the tape fingerprint rule (last user content)."""
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break
    return hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- record/replay

async def test_tape_record_save_replay():
    """Record 1 response with RecordingLLMClient(StubLLMClient), save, replay."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="lh_tape_"))
    tape_path = tmp_dir / "tape.jsonl"
    try:
        inner = StubLLMClient({"responses": [RESPONSE], "model": "stub-model"})
        recorder = RecordingLLMClient(
            {"model": "stub-model", "tape_path": str(tape_path)},
            wrapped_client=inner,
        )

        messages = [{"role": "user", "content": PROMPT}]
        resp = await recorder.complete(messages)

        # (a) recorder is a transparent pass-through
        assert resp.content == RESPONSE, "recorder should pass through stub content"
        assert len(recorder.tape) == 1, "exactly one tape entry expected"
        entry = recorder.tape[0]
        assert entry["fingerprint"] == _fingerprint(messages), \
            "tape fingerprint must follow sha256(last user content)[:16]"
        assert entry["response"]["content"] == RESPONSE, "tape should store content"
        assert set(entry["response"]["usage"]) >= {
            "prompt_tokens", "completion_tokens", "total_tokens"}, "usage must be recorded"

        # (b) save_tape() -> JSONL on disk (default path from config)
        recorder.save_tape()
        assert tape_path.exists(), "save_tape() should create the JSONL file"
        lines = [ln for ln in tape_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1, "one tape entry -> one JSONL line"
        saved = json.loads(lines[0])
        assert saved["fingerprint"] == entry["fingerprint"], "persisted fingerprint"

        # (c) save_tape(explicit path) writes a second copy
        alt_path = tmp_dir / "alt" / "tape_copy.jsonl"
        recorder.save_tape(alt_path)
        assert alt_path.exists(), "save_tape(path) should create parent dirs and file"

        # (d) replay from tape_path: same fingerprint hits, content/usage identical
        player = RecordedLLMClient({"tape_path": str(tape_path)})
        resp2 = await player.complete(messages)
        assert resp2.content == RESPONSE, "replay should return recorded content"
        assert resp2.usage == resp.usage, "replay should return recorded usage"
        assert len(player.call_history) == 1, "replay should log the call"

        # (e) same fingerprint again -> cursor wraps, still hits (no exception)
        resp3 = await player.complete(messages)
        assert resp3.content == RESPONSE, "repeat replay of same fingerprint"

        # (f) unknown fingerprint -> TapeMissError (never falls back to network)
        raised = False
        try:
            await player.complete([{"role": "user", "content": OTHER_PROMPT}])
        except TapeMissError:
            raised = True
        assert raised, "unknown fingerprint must raise TapeMissError"

        # (g) stream() replays too (offline), joined chunks equal the content
        player2 = RecordedLLMClient({"tape_path": str(alt_path)})
        chunks = [c async for c in player2.stream(messages)]
        assert "".join(chunks) == RESPONSE, "stream replay should reassemble content"

        print("[PASS] tape record -> save_tape -> replay hit -> TapeMissError on miss")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------- factory wiring

async def test_factory_stub_and_recorded():
    """LLMClientFactory.create("stub"/"recorded", minimal config) instantiates."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="lh_tape_factory_"))
    tape_path = tmp_dir / "factory_tape.jsonl"
    try:
        stub = LLMClientFactory.create("stub", {"responses": ["ok"]})
        assert isinstance(stub, StubLLMClient), "provider 'stub' -> StubLLMClient"
        resp = await stub.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "ok", "stub replays configured response"

        # minimal in-memory tape config (no file needed)
        mem = LLMClientFactory.create("recorded", {"tape": []})
        assert isinstance(mem, RecordedLLMClient), "provider 'recorded' -> RecordedLLMClient"

        # file-backed tape via factory
        inner = StubLLMClient({"responses": [RESPONSE]})
        rec = RecordingLLMClient({"tape_path": str(tape_path)}, wrapped_client=inner)
        msgs = [{"role": "user", "content": PROMPT}]
        await rec.complete(msgs)
        rec.save_tape()
        player = LLMClientFactory.create("recorded", {"tape_path": str(tape_path)})
        assert isinstance(player, RecordedLLMClient), "factory-built recorded client"
        out = await player.complete(msgs)
        assert out.content == RESPONSE, "factory-built recorded client replays tape"

        print("[PASS] LLMClientFactory.create('stub'/'recorded') wired and usable")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------- runner

_TESTS = [
    ("stub tape record/save/replay", test_tape_record_save_replay),
    ("factory stub + recorded", test_factory_stub_and_recorded),
]


async def main() -> int:
    print("=" * 60)
    print("Stub tape smoke test (offline, no API key, no network)")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    for name, fn in _TESTS:
        try:
            await fn()
            results.append((name, True))
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}")
            results.append((name, False))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            results.append((name, False))

    print()
    print("=" * 60)
    print("Stub tape smoke summary")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_passed = all(p for _, p in results)
    print()
    print("all green" if all_passed else "there are failures")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
