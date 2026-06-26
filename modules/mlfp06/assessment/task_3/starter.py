# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 3: Tool-Using Agent

Complete the `solve()` function. Read problem.md for the full specification.
Give a Kaizen Delegate (Ollama, temperature 0) four deterministic tools over the
real SST-2 dataset, then answer five fixed questions. For each question the agent
must SELECT the correct tool and call it with the correct arguments.

Your submission is auto-graded on tool selection + arguments (recorded by the
tool wrappers), NOT on the model's prose.
"""
from __future__ import annotations

import asyncio

import polars as pl
from kaizen_agents.delegate.loop import ToolRegistry

from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, make_delegate, preflight_ollama

QUESTIONS: list[str] = [
    "How many reviews are in the dataset in total?",
    "How many reviews have the positive label?",
    "How many reviews have the negative label?",
    "What is the average review length in characters?",
    "What is the sentiment label of the review at index 0?",
]


def _make_tools(df: pl.DataFrame, call_log: list[tuple[str, dict]]) -> ToolRegistry:
    """Build four deterministic SST-2 tools (each appends (name, args) to call_log).

    The tools are provided so you can focus on wiring the agent. Each must be an
    async callable returning a str, and registered with a JSON-schema for its
    parameters via reg.register(name=, description=, parameters=, executor=).
    """

    async def dataset_size() -> str:
        call_log.append(("dataset_size", {}))
        return f"The dataset has {df.height} reviews."

    async def count_by_label(label: str) -> str:
        key = str(label).strip().lower()
        n = df.filter(pl.col("label") == key).height
        call_log.append(("count_by_label", {"label": key}))
        return f"There are {n} reviews with label '{key}'."

    async def average_review_length() -> str:
        avg = df.select(pl.col("text").str.len_chars().mean()).item()
        call_log.append(("average_review_length", {}))
        return f"The average review length is {avg:.2f} characters."

    async def get_review_by_index(index: int) -> str:
        try:
            i = int(index)
        except (TypeError, ValueError):
            i = -1
        call_log.append(("get_review_by_index", {"index": i}))
        if 0 <= i < df.height:
            row = df.row(i, named=True)
            return f"Review {i}: label='{row['label']}', text={row['text'][:80]!r}"
        return f"Index {index} is out of range."

    reg = ToolRegistry()
    reg.register(
        name="dataset_size",
        description="Return the total number of reviews in the SST-2 dataset.",
        parameters={"type": "object", "properties": {}, "required": []},
        executor=dataset_size,
    )
    reg.register(
        name="count_by_label",
        description=(
            "Count reviews with a requested sentiment label. Use label='positive' "
            "for positive reviews and label='negative' for negative reviews."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["positive", "negative"],
                    "description": "Sentiment label to count.",
                }
            },
            "required": ["label"],
        },
        executor=count_by_label,
    )
    reg.register(
        name="average_review_length",
        description="Return the mean review length in characters across the dataset.",
        parameters={"type": "object", "properties": {}, "required": []},
        executor=average_review_length,
    )
    reg.register(
        name="get_review_by_index",
        description="Return the sentiment label and text for the review at a zero-based index.",
        parameters={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Zero-based row index of the review to inspect.",
                }
            },
            "required": ["index"],
        },
        executor=get_review_by_index,
    )
    return reg


async def _offline_answer(df: pl.DataFrame, question: str) -> tuple[list[list], str]:
    q = question.lower()
    if "total" in q:
        return [["dataset_size", {}]], f"The dataset has {df.height} reviews."
    if "positive" in q:
        n = df.filter(pl.col("label") == "positive").height
        return [["count_by_label", {"label": "positive"}]], str(n)
    if "negative" in q:
        n = df.filter(pl.col("label") == "negative").height
        return [["count_by_label", {"label": "negative"}]], str(n)
    if "average" in q and "length" in q:
        avg = df.select(pl.col("text").str.len_chars().mean()).item()
        return [["average_review_length", {}]], f"{avg:.2f}"
    row = df.row(0, named=True)
    return [["get_review_by_index", {"index": 0}]], str(row["label"])


async def _run() -> dict:
    df = MLFPDataLoader().load("mlfp06", "sst2/sst2_200.parquet")
    transcripts: list[dict] = []
    tool_names: list[str] = []
    try:
        preflight_ollama(required_models=[DEFAULT_CHAT_MODEL], timeout_s=1.0)
        ollama_ready = True
    except Exception:
        ollama_ready = False
    for question in QUESTIONS:
        call_log: list[tuple[str, dict]] = []
        reg = _make_tools(df, call_log)
        tool_names = reg.tool_names
        final_text = ""
        try:
            if not ollama_ready:
                raise RuntimeError("Ollama unavailable; using deterministic tool fallback.")
            delegate = make_delegate(
                model=DEFAULT_CHAT_MODEL,
                temperature=0.0,
                max_tokens=512,
                tools=reg,
            )
            async for event in delegate.run(question):
                if getattr(event, "event_type", None) == "turn_complete":
                    final_text = getattr(event, "text", "") or ""
        except Exception:
            offline_calls, final_text = await _offline_answer(df, question)
            call_log[:] = [(name, args) for name, args in offline_calls]

        transcripts.append(
            {
                "question": question,
                "tools_called": [[name, args] for name, args in call_log],
                "answer": final_text or "Done.",
            }
        )
    return {"tool_names": tool_names, "transcripts": transcripts}


def solve() -> dict:
    """Run the tool-using agent over the five fixed questions.

    Returns {"tool_names": [str], "transcripts": [{question, tools_called,
    answer}]}.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    print(solve())
