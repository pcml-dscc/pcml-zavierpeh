from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.mlfp06.assessment.task_1 import starter as task1
from modules.mlfp06.assessment.task_2 import starter as task2
from modules.mlfp06.assessment.task_3 import starter as task3
from modules.mlfp06.assessment.task_4 import starter as task4


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _squad_frame() -> pl.DataFrame:
    names = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
        "amber",
        "cobalt",
        "silver",
        "violet",
    ]
    return pl.DataFrame(
        {
            "text": [
                f"The {name} warehouse stores the distinctive token {name}fact."
                for name in names
            ],
            "question": [f"Which token is stored in the {name} warehouse?" for name in names],
            "answer": [f"{name}fact" for name in names],
        }
    )


def _sst2_frame() -> pl.DataFrame:
    labels = ["positive" if i % 2 == 0 else "negative" for i in range(20)]
    return pl.DataFrame(
        {
            "text": [f"review {i} has a clear {labels[i]} tone" for i in range(20)],
            "label": labels,
        }
    )


class _FakeLoader:
    def load(self, module: str, filename: str) -> pl.DataFrame:
        _assert(module == "mlfp06", f"unexpected module: {module}")
        if filename == "squad/squad_v2_300.parquet":
            return _squad_frame()
        if filename == "sst2/sst2_200.parquet":
            return _sst2_frame()
        raise AssertionError(f"unexpected filename: {filename}")


def validate_task1() -> None:
    records = task1.solve()
    _assert(len(records) == 6, "task1 should return six records")
    _assert(records[0]["incident_id"] == "INC-3001", "task1 incident id mismatch")
    _assert(records[0]["claim_required"] is True, "task1 claim flag mismatch")
    _assert(records[-1]["severity"] == "medium", "task1 severity mismatch")


def validate_task2() -> None:
    task2.MLFPDataLoader = _FakeLoader
    task2.preflight_ollama = lambda **_: (_ for _ in ()).throw(RuntimeError("offline"))
    out = task2.solve()
    _assert(set(out) == {"retrieved", "answers"}, "task2 keys mismatch")
    _assert(len(out["retrieved"]) == 6, "task2 should answer six questions")
    _assert(all(len(row) == 3 for row in out["retrieved"]), "task2 top-k mismatch")
    _assert(all(isinstance(i, int) for row in out["retrieved"] for i in row), "task2 indices")
    _assert(len(out["answers"]) == 6, "task2 answer count mismatch")
    _assert(all(str(answer).strip() for answer in out["answers"]), "task2 empty answer")


def validate_task3() -> None:
    task3.MLFPDataLoader = _FakeLoader
    task3.preflight_ollama = lambda **_: (_ for _ in ()).throw(RuntimeError("offline"))
    out = task3.solve()
    expected_tools = {
        "dataset_size",
        "count_by_label",
        "average_review_length",
        "get_review_by_index",
    }
    _assert(set(out["tool_names"]) == expected_tools, "task3 registered tools mismatch")
    _assert(len(out["transcripts"]) == len(task3.QUESTIONS), "task3 transcript count")
    called = [entry["tools_called"][0][0] for entry in out["transcripts"]]
    _assert(
        called
        == [
            "dataset_size",
            "count_by_label",
            "count_by_label",
            "average_review_length",
            "get_review_by_index",
        ],
        "task3 tool routing mismatch",
    )
    _assert(out["transcripts"][1]["tools_called"][0][1] == {"label": "positive"}, "positive args")
    _assert(out["transcripts"][4]["tools_called"][0][1] == {"index": 0}, "index args")


def validate_task4() -> None:
    out = task4.solve()
    _assert(out["org_stats"] == {"n_agents": 6, "n_delegations": 6, "n_departments": 3}, "task4 org")
    _assert(out["escalation_caught"] is True, "task4 escalation")
    _assert(len(out["verdicts"]) == 10, "task4 verdict count")
    _assert(out["verdicts"].count(True) == 5, "task4 expected five allowed verdicts")


if __name__ == "__main__":
    validate_task1()
    print("task1 ok")
    validate_task2()
    print("task2 ok")
    validate_task3()
    print("task3 ok")
    validate_task4()
    print("task4 ok")
