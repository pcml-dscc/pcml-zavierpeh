# MLFP06 — Task 3: Tool-Using Agent over a Real Dataset

**Weight**: 25 marks · **Difficulty**: Hard · **Framework**: Kaizen `Delegate`

- `ToolRegistry` (Ollama, `llama3.2:3b`) · **Dataset**:
  `data/mlfp06/sst2/sst2_200.parquet` (real SST-2 sentiment, 200 rows)

## Scenario

A data desk wants one agent that can answer a whole class of questions about a
dataset by **choosing the right tool** — instead of a bespoke script per
question. You will give a Kaizen Delegate four deterministic tools over the real
SST-2 dataset and let the LLM decide, per question, which tool to call and with
what arguments. This is the core agentic skill: **tool selection + argument
extraction**, with the deterministic computation done by the tools, not the
model.

The four tools are provided (they query the real data and record every call).
Your job is to register them with JSON schemas and run the agent over five fixed
questions.

Implement `solve() -> dict`.

## The four tools (provided)

| Tool                    | Parameters   | Returns                                  |
| ----------------------- | ------------ | ---------------------------------------- |
| `dataset_size`          | _(none)_     | total number of reviews                  |
| `count_by_label`        | `label: str` | count of reviews with that label         |
| `average_review_length` | _(none)_     | mean review length in characters         |
| `get_review_by_index`   | `index: int` | label + text of the review at that index |

## The five questions (single-hop — exactly one correct tool each)

| #   | Question                                              | Correct tool            | Correct arg        |
| --- | ----------------------------------------------------- | ----------------------- | ------------------ |
| 0   | How many reviews are in the dataset in total?         | `dataset_size`          | —                  |
| 1   | How many reviews have the positive label?             | `count_by_label`        | `label="positive"` |
| 2   | How many reviews have the negative label?             | `count_by_label`        | `label="negative"` |
| 3   | What is the average review length in characters?      | `average_review_length` | —                  |
| 4   | What is the sentiment label of the review at index 0? | `get_review_by_index`   | `index=0`          |

## What to build

1. **Register** all four tools on a `ToolRegistry` with correct JSON-schema
   `parameters` (the no-arg tools use `properties={}`). The tool description is
   what the LLM reads to choose — make it precise.
2. **Run** a `make_delegate(..., tools=reg)` (temperature 0) on each question;
   capture the `turn_complete` event's text as the answer.
3. **Record** each transcript: the question, the list of `[tool_name, args]`
   actually called (from the provided `call_log`), and the final answer.

## Return contract

```python
def solve() -> dict:
    return {
        "tool_names": [str, ...],          # the 4 registered tool names
        "transcripts": [                   # one per question, in order
            {"question": str,
             "tools_called": [[name, args], ...],
             "answer": str},
            ...
        ],
    }
```

## Visible sanity check

A correct agent selects `dataset_size` for Q0, `count_by_label(label="positive")`
for Q1, …, `get_review_by_index(index=0)` for Q4 — one correct tool per question,
recorded in `tools_called`.

## Grading (11 automated checks, all must pass)

return type is dict · all four tool names registered · 5 transcripts · transcript
keys present · every question invoked ≥1 tool · no hallucinated tool names ·
**correct tool selected for all 5 questions** · `count_by_label` args correct
(positive + negative) · `get_review_by_index` index arg is 0 · all four tools
exercised across the run · every answer non-empty.

**How this stays deterministic.** The graded signal is the **tool call + its
arguments**, recorded by the tool wrappers themselves — independent of the
model's prose. Because each tool computes its result deterministically from the
real SST-2 data, "correct tool + correct args" guarantees the correct computed
value was produced as an observation. The model's final wording (which small
local models often fail to populate with the value) is NOT graded. At
temperature 0 the tool-selection outcome is byte-stable across runs.

## Rules

- **Kaizen `Delegate` + `ToolRegistry` only** — no raw LLM HTTP, no hand-rolled
  routing. The LLM must choose the tool; do not `if`/`else` on the question.
- **Local Ollama only**, temperature 0.
- Do not modify the provided tools or questions.
