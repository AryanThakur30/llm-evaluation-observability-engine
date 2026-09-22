"""Lightweight span tracer (OpenTelemetry-compatible concepts, zero deps).

Spans nest via contextvars, carry trace_id/span_id/parent_id exactly like
OTel, and export to any callable sink (DuckDB in this project). Swapping to
a real OTel exporter later is mechanical: the data model already matches.
"""
from __future__ import annotations

import contextvars
import time
import uuid

_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar("span_stack", default=())


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Span:
    __slots__ = ("trace_id", "span_id", "parent_id", "name", "start_ns", "end_ns", "attributes")

    def __init__(self, trace_id, span_id, parent_id, name, start_ns, attributes):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.start_ns = start_ns
        self.end_ns = 0
        self.attributes = dict(attributes)

    def set(self, key: str, value):
        self.attributes[key] = value

    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def as_row(self):
        return (
            self.trace_id,
            self.span_id,
            self.parent_id,
            self.name,
            self.start_ns,
            self.end_ns,
            self.attributes,
        )


class _ActiveSpan:
    """Context manager returned by Tracer.span()."""

    def __init__(self, tracer: "Tracer", span: Span, token):
        self.tracer = tracer
        self.span = span
        self._token = token

    def __enter__(self) -> Span:
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.span.end_ns = time.time_ns()
        if exc is not None:
            self.span.set("error", repr(exc))
        self.tracer.finished.append(self.span)
        if self.tracer.exporter is not None:
            self.tracer.exporter(self.span)
        _stack.reset(self._token)
        return False

    def set(self, key: str, value):
        self.span.set(key, value)


class Tracer:
    """Creates nested spans and exports finished ones to `exporter`."""

    def __init__(self, exporter=None):
        self.exporter = exporter
        self.finished: list[Span] = []

    def span(self, name: str, **attributes) -> _ActiveSpan:
        stack = _stack.get()
        parent = stack[-1] if stack else None
        trace_id = parent.trace_id if parent else _new_id()
        sp = Span(trace_id, _new_id(), parent.span_id if parent else None, name, time.time_ns(), attributes)
        token = _stack.set(stack + (sp,))
        return _ActiveSpan(self, sp, token)
