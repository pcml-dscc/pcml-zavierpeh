# MLFP06 — Task 2: RAG Pipeline with Evaluation

**Weight**: 25 marks · **Difficulty**: Hard · **Framework**: Kaizen Ollama
embeddings + Delegate (`nomic-embed-text`, `llama3.2:3b`) · **Dataset**:
`data/mlfp06/squad/squad_v2_300.parquet` (real SQuAD v2 contexts + Q&A)

## Scenario

A knowledge-base assistant must answer staff questions using a fixed document
corpus — and it must be **grounded**: every answer comes from a retrieved
passage, not the model's memory. You will build the two-stage RAG pipeline
(dense retrieval, then grounded generation) and it will be evaluated the way
production RAG is evaluated: **retrieval recall@k** and **grounded-answer
fact containment**.

The corpus and the six evaluation questions are built for you, deterministically,
by `build_corpus_and_questions()` (first 30 unique SQuAD contexts; six questions
whose gold answer is a short distinctive fact).

Implement `solve() -> dict`.

## Pipeline to build

1. **Embed** the corpus and the six questions with the Ollama embedder
   (`make_embedder(model="nomic-embed-text")`, `await embedder.embed([...])`).
   Vectors are 768-dim.
2. **Retrieve** — for each question, rank the 30 corpus docs by cosine
   similarity (`_cosine` is provided) and take the **top-3 indices** (most
   similar first).
3. **Generate** — for each question, build a context string from its top-3 docs
   and generate a **grounded** answer with `make_delegate(temperature=0.0)` +
   `run_delegate_text`. Instruct the model to answer using ONLY the context.

## Return contract

```python
def solve() -> dict:
    return {
        "retrieved": [[int, int, int], ...],  # 6 lists, top-3 doc indices each
        "answers":   [str, ...],              # 6 grounded answer strings
    }
```

## Visible sanity check

For a correct pipeline, every question's gold passage is retrieved at rank 1
(recall@1 = 6/6 on this corpus), and each answer contains the gold fact — e.g.
the "Pacific Ocean" question returns an answer containing `pacific`/`ocean`.

## Grading (9 automated checks, all must pass)

return type is dict · `retrieved` shape (6 lists) · `answers` shape (6 non-empty
strings) · top-3 size · indices in range `[0,30)` · **recall@1 ≥ 5/6** ·
**recall@3 = 6/6** · **answers grounded ≥ 5/6** (each answer contains a content
token from the gold answer) · answers non-trivial length.

**How this stays deterministic.** Retrieval is pure embedding cosine over a
fixed corpus — the grader independently re-derives each query's gold passage
index from the SQuAD parquet and checks recall@k. Generation is graded by
**grounded fact containment** (does the answer contain a content token from the
gold answer), never by exact text, because LLM phrasing is not bit-stable. At
temperature 0 the grounded outcome is stable; the 5/6 floors absorb at most one
drift.

## Rules

- **Kaizen Ollama embeddings + Delegate** — no external vector DB, no cloud
  embeddings/models. Temperature 0 for generation.
- Retrieval must be real cosine similarity over your embeddings — no hardcoded
  indices.
- Do not modify the corpus/question builder.
