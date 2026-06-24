# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Trace format shared by Agent and RAG lenses.

Schema follows the OpenTelemetry GenAI semantic conventions (2026-01 draft)
so Langfuse / Langsmith / Phoenix can ingest MLFP traces without translation.

A trace is a JSONL file — one event per line — with a stable schema:

    {"ts": "2026-04-15T12:01:03.142Z", "run_id": "r_abc123",
     "kind": "thought"|"action"|"observation"|"tool_start"|"tool_end"|"token"|"complete"|"error",
     ...kind-specific fields...,
     "cost_usd": 0.0, "latency_ms": 0.0}

This module owns:

    * ``TraceEvent`` dataclass  — one event, serialisable to JSON
    * ``AgentTrace`` class      — in-memory list of events + JSONL writer
    * ``kaizen_events_to_trace``— Kaizen ``StreamEvent`` → ``TraceEvent`` converter
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Event schema
# ════════════════════════════════════════════════════════════════════════


@dataclass
class TraceEvent:
    """One event in an agent / RAG trace.

    Fields map 1:1 to the JSONL wire format. Optional fields default to
    ``None`` and are dropped from the serialised form.

    Attributes:
        ts: ISO-8601 UTC timestamp.
        run_id: Correlation ID for the enclosing run.
        kind: Event category. One of: ``token``, ``thought``, ``action``,
            ``observation``, ``tool_start``, ``tool_end``, ``complete``,
            ``error``.
        content: Free-text content (thought text, response chunk, etc).
        tool: Name of tool when ``kind`` is ``tool_*``.
        args: Tool arguments as a JSON-serialisable dict.
        result: Tool result preview (truncated for large outputs).
        error: Error string when ``kind == "error"``.
        cost_usd: Cost attributable to this event.
        latency_ms: Latency of the event (tool call, LLM turn, etc).
        tokens_in, tokens_out: Token counts for LLM calls.
        meta: Arbitrary structured metadata.
    """

    ts: str
    run_id: str
    kind: str
    content: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialise to one line of JSON (no trailing newline)."""
        payload = {k: v for k, v in asdict(self).items() if v is not None and v != {}}
        return json.dumps(payload, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        return cls(
            ts=data.get("ts", _now_iso()),
            run_id=data.get("run_id", "unknown"),
            kind=data.get("kind", "unknown"),
            content=data.get("content"),
            tool=data.get("tool"),
            args=data.get("args"),
            result=data.get("result"),
            error=data.get("error"),
            cost_usd=float(data.get("cost_usd", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            tokens_in=data.get("tokens_in"),
            tokens_out=data.get("tokens_out"),
            meta=data.get("meta", {}) or {},
        )


# ════════════════════════════════════════════════════════════════════════
# AgentTrace — in-memory + JSONL persistence
# ════════════════════════════════════════════════════════════════════════


class AgentTrace:
    """An append-only in-memory trace with optional JSONL persistence.

    Context-manager compatible: any writes are flushed on ``__exit__``.

    Example::

        with AgentTrace(run_id="demo", path=Path("runs/demo.jsonl")) as trace:
            trace.append(TraceEvent(ts=_now_iso(), run_id="demo", kind="thought",
                                    content="Planning..."))
            for k, v in enumerate(range(3)):
                trace.append_simple(kind="action", tool="search", args={"q": str(k)})
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        path: Path | str | None = None,
    ) -> None:
        self.run_id = run_id or f"r_{uuid.uuid4().hex[:12]}"
        self._events: list[TraceEvent] = []
        self.path: Path | None = Path(path) if path is not None else None
        self._file: Any = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "trace.init",
            extra={
                "run_id": self.run_id,
                "path": str(self.path) if self.path else None,
            },
        )

    def __enter__(self) -> "AgentTrace":
        if self.path is not None:
            self._file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                # Cleanup path — zero-tolerance Rule 3 carve-out.
                pass
            self._file = None

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def append(self, event: TraceEvent) -> None:
        """Append a fully-constructed event."""
        self._events.append(event)
        if self._file is not None:
            self._file.write(event.to_jsonl())
            self._file.write("\n")

    def append_simple(self, *, kind: str, **kwargs: Any) -> TraceEvent:
        """Construct + append in one step. Returns the event for chaining."""
        event = TraceEvent(ts=_now_iso(), run_id=self.run_id, kind=kind, **kwargs)
        self.append(event)
        return event

    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self._events)

    def total_latency_ms(self) -> float:
        return sum(e.latency_ms for e in self._events)

    def filter_kind(self, kind: str) -> list[TraceEvent]:
        return [e for e in self._events if e.kind == kind]

    # ── Persistence ─────────────────────────────────────────────────────

    def write(self, path: Path | str) -> Path:
        """Write the full trace to a JSONL file. Returns the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for event in self._events:
                f.write(event.to_jsonl())
                f.write("\n")
        logger.info(
            "trace.written",
            extra={"run_id": self.run_id, "path": str(p), "events": len(self._events)},
        )
        return p

    @classmethod
    def read(cls, path: Path | str) -> "AgentTrace":
        """Load a JSONL trace from disk."""
        p = Path(path)
        trace = cls(run_id=p.stem)
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace._events.append(TraceEvent.from_dict(json.loads(line)))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "trace.read.skip_malformed_line",
                        extra={"path": str(p), "line": line_no, "error": str(exc)},
                    )
        logger.info(
            "trace.read",
            extra={"path": str(p), "events": len(trace._events)},
        )
        return trace


# ════════════════════════════════════════════════════════════════════════
# Kaizen StreamEvent → TraceEvent adapter
# ════════════════════════════════════════════════════════════════════════


def kaizen_events_to_trace(
    stream_events: Iterable[Any],
    *,
    run_id: str | None = None,
    path: Path | str | None = None,
) -> AgentTrace:
    """Convert a Kaizen ``StreamEvent`` iterable into an :class:`AgentTrace`.

    Recognised event types (from ``kaizen_agents.events``):
        * ``TextDelta``     → ``kind="token"``
        * ``ToolCallStart`` → ``kind="tool_start"``
        * ``ToolCallEnd``   → ``kind="tool_end"`` or ``kind="error"``
        * ``TurnComplete``  → ``kind="complete"``
        * ``ErrorEvent``    → ``kind="error"``

    The adapter is defensive — any unknown event type is recorded as
    ``kind="meta"`` with the raw class name in ``meta``.
    """
    trace = AgentTrace(run_id=run_id, path=path)
    pending_tool_starts: dict[str, float] = {}

    for event in stream_events:
        cls_name = type(event).__name__
        ts = _now_iso()

        if cls_name == "TextDelta":
            trace.append(
                TraceEvent(
                    ts=ts,
                    run_id=trace.run_id,
                    kind="token",
                    content=getattr(event, "text", ""),
                )
            )
        elif cls_name == "ToolCallStart":
            call_id = getattr(event, "call_id", "unknown")
            pending_tool_starts[call_id] = _monotonic_ms()
            trace.append(
                TraceEvent(
                    ts=ts,
                    run_id=trace.run_id,
                    kind="tool_start",
                    tool=getattr(event, "name", None),
                    meta={"call_id": call_id},
                )
            )
        elif cls_name == "ToolCallEnd":
            call_id = getattr(event, "call_id", "unknown")
            start_ms = pending_tool_starts.pop(call_id, None)
            latency = (_monotonic_ms() - start_ms) if start_ms is not None else 0.0
            err = getattr(event, "error", None)
            trace.append(
                TraceEvent(
                    ts=ts,
                    run_id=trace.run_id,
                    kind="error" if err else "tool_end",
                    tool=getattr(event, "name", None),
                    result=_truncate(getattr(event, "result", None)),
                    error=str(err) if err else None,
                    latency_ms=latency,
                    meta={"call_id": call_id},
                )
            )
        elif cls_name == "TurnComplete":
            usage = getattr(event, "usage", {}) or {}
            trace.append(
                TraceEvent(
                    ts=ts,
                    run_id=trace.run_id,
                    kind="complete",
                    content=_truncate(getattr(event, "text", None)),
                    tokens_in=usage.get("input_tokens") or usage.get("prompt_tokens"),
                    tokens_out=usage.get("output_tokens")
                    or usage.get("completion_tokens"),
                    cost_usd=float(usage.get("cost_usd", 0.0)),
                    meta={"iterations": getattr(event, "iterations", None)},
                )
            )
        elif cls_name == "ErrorEvent":
            trace.append(
                TraceEvent(
                    ts=ts,
                    run_id=trace.run_id,
                    kind="error",
                    error=str(getattr(event, "error", "unknown")),
                    meta={"details": getattr(event, "details", {})},
                )
            )
        else:
            trace.append(
                TraceEvent(
                    ts=ts,
                    run_id=trace.run_id,
                    kind="meta",
                    meta={"stream_event_type": cls_name},
                )
            )
    return trace


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _monotonic_ms() -> float:
    import time

    return time.monotonic() * 1000.0


def _truncate(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(s) <= limit:
        return s
    return s[:limit] + f"... <+{len(s) - limit} chars>"
