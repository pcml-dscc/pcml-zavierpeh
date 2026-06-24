# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Lens 1 — Output Diagnostics (the Stethoscope).

Question answered: *Is the generation coherent, faithful, and on-task?*

Wraps industry evaluation libraries (``deepeval``, ``ragas``, ``sacrebleu``,
``rouge-score``, ``bert-score``) plus a Kaizen-powered LLM-as-judge. Raw
``openai.*`` calls are BLOCKED per ``rules/framework-first.md``.

Typical use::

    from shared.mlfp06.diagnostics import LLMDiagnostics

    diag = LLMDiagnostics(max_judge_calls=20)
    verdict = diag.llm_as_judge(
        prompt="Capital of France?",
        response="Paris.",
        criteria="factual_accuracy",
    )
    faithful_df = diag.faithfulness(response, context=["Paris is the capital..."])
    diag.plot_output_dashboard().show()
    print(diag.report())
"""
from __future__ import annotations

import logging
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import plotly.graph_objects as go
import polars as pl

from . import _plots
from ._judges import JudgeCallable, JudgeVerdict

logger = logging.getLogger(__name__)

__all__ = ["LLMDiagnostics", "JudgeVerdict"]

# Refusal detection patterns — used ONLY as a heuristic for `refusal_rate`.
# Per rules/agent-reasoning.md: this is not agent decision-making, it's a
# metric heuristic over already-generated outputs. The LLM judge is used
# for the nuanced calls.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "as an ai",
    "i'm not able",
    "refuse",
    "not appropriate",
    "unable to help",
)


@dataclass
class _OutputEntry:
    prompt: str
    response: str
    criteria: str
    verdict: JudgeVerdict


class LLMDiagnostics:
    """Output-lens diagnostics — faithfulness, coherence, refusal calibration.

    Args:
        judge_model: Judge model name. Resolved via
            :func:`~shared.mlfp06.diagnostics._judges.resolve_judge_model`.
        max_judge_calls: Hard cap on live judge calls (default ``50``).
        delegate: Optional pre-built Kaizen ``Delegate`` (used as the judge).
        sensitive: When ``True``, prompt/response bodies are not logged.
    """

    def __init__(
        self,
        *,
        judge_model: str | None = None,
        max_judge_calls: int = 50,
        delegate: Any = None,
        sensitive: bool = False,
    ) -> None:
        self._judge = JudgeCallable(
            judge_model=judge_model,
            max_judge_calls=max_judge_calls,
            delegate=delegate,
            sensitive=sensitive,
        )
        self._log: list[_OutputEntry] = []
        self._refusal_log: list[dict[str, Any]] = []
        self._faithful_log: list[dict[str, Any]] = []
        self._consistency_log: list[dict[str, Any]] = []
        logger.info(
            "output_diagnostics.init",
            extra={"judge_model": self._judge.model, "max_calls": max_judge_calls},
        )

    def __enter__(self) -> "LLMDiagnostics":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._judge.close()

    # ── Judge (Kaizen-backed) ──────────────────────────────────────────

    def llm_as_judge(
        self,
        prompt: str,
        response: str,
        *,
        criteria: str = "coherence,helpfulness,harmlessness",
        run_id: str | None = None,
    ) -> JudgeVerdict:
        """Score a single (prompt, response) pair via the Kaizen-backed judge.

        Args:
            prompt: The user prompt the model was responding to.
            response: The model's response under evaluation.
            criteria: Comma-separated scoring criteria.
            run_id: Correlation ID per ``rules/observability.md``.

        Returns:
            A :class:`~shared.mlfp06.diagnostics._judges.JudgeVerdict`.
        """
        run_id = run_id or f"llm_judge-{uuid.uuid4().hex[:12]}"
        verdict = self._judge.score(
            response,
            criteria=criteria,
            context=f"User prompt: {prompt}",
            run_id=run_id,
        )
        self._log.append(
            _OutputEntry(
                prompt=prompt, response=response, criteria=criteria, verdict=verdict
            )
        )
        return verdict

    def evaluate(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        criteria: str = "coherence,helpfulness,harmlessness",
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Batch judge over a sequence of (prompt, response) pairs.

        Returns a Polars DataFrame with one row per pair.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must be same length")
        run_id = run_id or f"llm_eval-{uuid.uuid4().hex[:12]}"
        rows: list[dict[str, Any]] = []
        for i, (p, r) in enumerate(zip(prompts, responses)):
            v = self.llm_as_judge(p, r, criteria=criteria, run_id=f"{run_id}-{i}")
            rows.append(
                {
                    "idx": i,
                    "score": v.score,
                    "mode": v.mode,
                    "latency_ms": v.latency_ms,
                    "criteria": v.criteria,
                    "judge_model": v.judge_model,
                    "rationale": v.rationale,
                }
            )
        return pl.DataFrame(rows)

    # ── Faithfulness (RAG grounding) ───────────────────────────────────

    def faithfulness(
        self,
        response: str,
        context: Sequence[str] | str,
        *,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Judge whether ``response`` is grounded in ``context``.

        Delegates to the Kaizen judge with a strict grounding criterion.
        Returns a one-row Polars DataFrame. Multi-chunk context is joined
        with explicit chunk markers so the judge can cite.
        """
        if isinstance(context, str):
            context_blob = context
        else:
            context_blob = "\n\n".join(
                f"[chunk {i}] {c}" for i, c in enumerate(context)
            )
        verdict = self._judge.score(
            response,
            criteria="faithfulness,grounded_in_context,no_fabrication",
            context=f"Retrieved context:\n{context_blob}",
            run_id=run_id or f"faithful-{uuid.uuid4().hex[:12]}",
        )
        row = {
            "faithfulness": verdict.score,
            "mode": verdict.mode,
            "judge_model": verdict.judge_model,
            "latency_ms": verdict.latency_ms,
            "rationale": verdict.rationale,
        }
        self._faithful_log.append(row)
        return pl.DataFrame([row])

    # ── Self-consistency / hallucination scan ──────────────────────────

    def self_consistency(
        self,
        responses: Sequence[str],
        *,
        prompt: str = "",
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Measure agreement across ``n`` samples from the same prompt.

        The caller samples the responses externally (typically via
        ``Delegate.run_sync`` called ``n`` times with a non-zero
        temperature). This method then:

            * normalises whitespace + lowercases tokens
            * computes pairwise ROUGE-L (simple Jaccard fallback when
              ``rouge-score`` is not installed)
            * flags hallucination candidates as the lowest-agreement sample

        Returns a Polars DataFrame with ``idx``, ``response`` (truncated),
        ``agreement`` (mean pairwise similarity), ``is_outlier``.
        """
        if len(responses) < 2:
            raise ValueError("self_consistency needs >= 2 samples")
        run_id = run_id or f"self_consistency-{uuid.uuid4().hex[:12]}"

        similarity_fn = _build_similarity_fn()
        agreements: list[float] = []
        for i, ri in enumerate(responses):
            others = [rj for j, rj in enumerate(responses) if j != i]
            sims = [similarity_fn(ri, rj) for rj in others]
            agreements.append(sum(sims) / max(len(sims), 1))

        mean_agreement = sum(agreements) / len(agreements)
        threshold = mean_agreement * 0.75

        rows = [
            {
                "idx": i,
                "response_preview": r[:120],
                "agreement": agreements[i],
                "is_outlier": agreements[i] < threshold,
            }
            for i, r in enumerate(responses)
        ]
        self._consistency_log.append(
            {
                "run_id": run_id,
                "prompt_preview": prompt[:120],
                "n_samples": len(responses),
                "mean_agreement": mean_agreement,
                "n_outliers": sum(1 for a in agreements if a < threshold),
            }
        )
        logger.info(
            "output.self_consistency",
            extra={
                "run_id": run_id,
                "n_samples": len(responses),
                "mean_agreement": mean_agreement,
                "source": "local_metric",
                "mode": "real",
            },
        )
        return pl.DataFrame(rows)

    # ── Refusal calibration ────────────────────────────────────────────

    def refusal_rate(
        self,
        responses: Iterable[str],
        *,
        label: str = "sample",
    ) -> float:
        """Fraction of responses that look like a refusal.

        This is a heuristic over already-generated outputs (see
        ``_REFUSAL_MARKERS``) — it is *not* agent decision-making, so the
        ``rules/agent-reasoning.md`` keyword-match prohibition does not
        apply. For nuanced calls use :meth:`llm_as_judge` with
        ``criteria="is_refusal"``.
        """
        responses = list(responses)
        if not responses:
            return 0.0
        refused = sum(1 for r in responses if _looks_like_refusal(r))
        rate = refused / len(responses)
        self._refusal_log.append(
            {"label": label, "n": len(responses), "refused": refused, "rate": rate}
        )
        logger.info(
            "output.refusal_rate",
            extra={"label": label, "rate": rate, "n": len(responses), "mode": "real"},
        )
        return rate

    def over_refusal(self, benign_responses: Iterable[str]) -> float:
        """Refusal rate on a benign set — any refusals are over-refusals."""
        return self.refusal_rate(benign_responses, label="benign")

    # ── Classical metrics (thin wrappers) ──────────────────────────────

    def rouge(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        rouge_type: str = "rougeL",
    ) -> pl.DataFrame:
        """ROUGE (default ``rougeL``) score per (prediction, reference) pair."""
        if len(predictions) != len(references):
            raise ValueError("predictions and references must be same length")
        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:  # pragma: no cover — optional extra
            raise ImportError(
                "rouge requires the rouge-score package. Install with "
                "`pip install rouge-score`."
            ) from exc
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        rows = []
        for i, (p, r) in enumerate(zip(predictions, references)):
            s = scorer.score(r, p)[rouge_type]
            rows.append(
                {
                    "idx": i,
                    "precision": s.precision,
                    "recall": s.recall,
                    "fmeasure": s.fmeasure,
                }
            )
        return pl.DataFrame(rows)

    def bleu(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        """Corpus-level BLEU via ``sacrebleu``."""
        if len(predictions) != len(references):
            raise ValueError("predictions and references must be same length")
        try:
            import sacrebleu
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "bleu requires sacrebleu. Install with `pip install sacrebleu`."
            ) from exc
        # sacrebleu expects list-of-references (list of corpora), not pairs.
        return float(sacrebleu.corpus_bleu(list(predictions), [list(references)]).score)

    def bertscore(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        lang: str = "en",
    ) -> pl.DataFrame:
        """BERTScore per pair (requires the ``bert-score`` extra)."""
        try:
            from bert_score import score as _bs
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "bertscore requires bert-score. Install with "
                "`pip install bert-score`."
            ) from exc
        P, R, F = _bs(list(predictions), list(references), lang=lang, verbose=False)
        return pl.DataFrame(
            {
                "idx": list(range(len(predictions))),
                "precision": [float(x) for x in P.tolist()],
                "recall": [float(x) for x in R.tolist()],
                "f1": [float(x) for x in F.tolist()],
            }
        )

    def perplexity(self, token_logprobs: Sequence[float]) -> float:
        """Perplexity from precomputed token log-probabilities (natural log).

        The caller supplies the logprobs (typically from the target model's
        ``logprobs`` API). This method does not call the model — it is a
        pure reduction so the judge can focus on grading, not scoring.
        """
        if not token_logprobs:
            return float("nan")
        mean = sum(token_logprobs) / len(token_logprobs)
        try:
            return float(math.exp(-mean))
        except OverflowError:
            return float("inf")

    # ── DataFrames ─────────────────────────────────────────────────────

    def results_df(self) -> pl.DataFrame:
        """One row per :meth:`llm_as_judge` call."""
        if not self._log:
            return pl.DataFrame(
                schema={
                    "prompt_preview": pl.Utf8,
                    "response_preview": pl.Utf8,
                    "criteria": pl.Utf8,
                    "score": pl.Float64,
                    "mode": pl.Utf8,
                    "latency_ms": pl.Float64,
                }
            )
        return pl.DataFrame(
            [
                {
                    "prompt_preview": e.prompt[:120],
                    "response_preview": e.response[:120],
                    "criteria": e.criteria,
                    "score": e.verdict.score,
                    "mode": e.verdict.mode,
                    "latency_ms": e.verdict.latency_ms,
                }
                for e in self._log
            ]
        )

    def refusal_df(self) -> pl.DataFrame:
        if not self._refusal_log:
            return pl.DataFrame(
                schema={
                    "label": pl.Utf8,
                    "n": pl.Int64,
                    "refused": pl.Int64,
                    "rate": pl.Float64,
                }
            )
        return pl.DataFrame(self._refusal_log)

    def faithfulness_df(self) -> pl.DataFrame:
        if not self._faithful_log:
            return pl.DataFrame(
                schema={
                    "faithfulness": pl.Float64,
                    "mode": pl.Utf8,
                    "judge_model": pl.Utf8,
                    "latency_ms": pl.Float64,
                }
            )
        return pl.DataFrame(self._faithful_log).drop("rationale", strict=False)

    # ── Plots ──────────────────────────────────────────────────────────

    def plot_output_dashboard(self) -> go.Figure:
        """Score distribution + refusal bars + faithfulness histogram."""
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Judge scores",
                "Faithfulness",
                "Refusal rate by label",
                "Score by criteria",
            ),
        )

        # (1,1) Judge scores histogram.
        results = self.results_df()
        if results.height:
            fig.add_trace(
                go.Histogram(
                    x=results["score"].to_list(),
                    marker_color=_plots.PRIMARY,
                    nbinsx=20,
                ),
                row=1,
                col=1,
            )
        # (1,2) Faithfulness histogram.
        faith = self.faithfulness_df()
        if faith.height:
            fig.add_trace(
                go.Histogram(
                    x=faith["faithfulness"].to_list(),
                    marker_color=_plots.WARN,
                    nbinsx=20,
                ),
                row=1,
                col=2,
            )
        # (2,1) Refusal bars.
        ref_df = self.refusal_df()
        if ref_df.height:
            fig.add_trace(
                go.Bar(
                    x=ref_df["label"].to_list(),
                    y=ref_df["rate"].to_list(),
                    marker_color=_plots.ACCENT,
                ),
                row=2,
                col=1,
            )
        # (2,2) Score by criteria.
        if results.height:
            agg = (
                results.group_by("criteria")
                .agg(pl.col("score").mean().alias("mean_score"))
                .sort("mean_score")
            )
            fig.add_trace(
                go.Bar(
                    x=agg["criteria"].to_list(),
                    y=agg["mean_score"].to_list(),
                    marker_color=_plots.PRIMARY,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Output Lens Dashboard (Stethoscope)",
            template=_plots.TEMPLATE,
            showlegend=False,
            height=640,
        )
        return fig

    # ── Report ─────────────────────────────────────────────────────────

    def report(self) -> str:
        """Auto-diagnosis in plain text. One line per finding."""
        out: list[str] = []
        results = self.results_df()
        if results.height:
            mean = float(results["score"].mean() or 0.0)
            real_frac = float((results["mode"] == "real").mean() or 0.0)
            out.append(
                f"judge: {results.height} calls, mean score={mean:.2f}, "
                f"real_mode={real_frac:.0%}"
            )
            low = results.filter(pl.col("score") < 0.5).height
            if low:
                out.append(f"  -> {low} calls scored below 0.5 — review rationales")
        faith = self.faithfulness_df()
        if faith.height:
            mean_f = float(faith["faithfulness"].mean() or 0.0)
            out.append(f"faithfulness: {faith.height} calls, mean={mean_f:.2f}")
            if mean_f < 0.7:
                out.append("  -> faithfulness below 0.7 — retrieval or grounding weak")
        ref = self.refusal_df()
        if ref.height:
            for row in ref.iter_rows(named=True):
                out.append(
                    f"refusal[{row['label']}]: {row['rate']:.0%} "
                    f"({row['refused']}/{row['n']})"
                )
        if self._consistency_log:
            last = self._consistency_log[-1]
            out.append(
                f"self-consistency: mean agreement={last['mean_agreement']:.2f}, "
                f"outliers={last['n_outliers']}"
            )
        if not out:
            return "output-lens: no readings recorded yet."
        return "output-lens:\n  " + "\n  ".join(out)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _looks_like_refusal(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _REFUSAL_MARKERS)


def _build_similarity_fn():
    """Return a ``(a, b) -> float`` similarity. ROUGE-L when available, else Jaccard."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        def _fn(a: str, b: str) -> float:
            return scorer.score(a, b)["rougeL"].fmeasure

        return _fn
    except ImportError:
        logger.info(
            "output.self_consistency.fallback_jaccard",
            extra={"reason": "rouge_score not installed"},
        )

        def _jaccard(a: str, b: str) -> float:
            ta = Counter(a.lower().split())
            tb = Counter(b.lower().split())
            if not ta and not tb:
                return 1.0
            inter = sum((ta & tb).values())
            union = sum((ta | tb).values())
            return inter / union if union else 0.0

        return _jaccard
