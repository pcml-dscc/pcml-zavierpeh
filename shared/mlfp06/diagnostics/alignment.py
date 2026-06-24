# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Lens 5 — Alignment Diagnostics (the ECG).

Question answered: *Is the fine-tuning signal rewarding the right thing?*

Three primary readings:

    1. **Pair evaluation** — given a ``base_policy`` and a ``tuned_policy``
       plus a preference set, compute KL(base || tuned), reward-margin,
       and pairwise win-rate.
    2. **Training-curve tracking** — pull per-step metrics from a Kailash
       Align :class:`~kailash_align.AlignmentPipeline` (or any iterable of
       ``{step, reward, kl, loss, ...}`` dicts) and record them.
    3. **Reward-hacking detection** — flag the canonical signature of
       sudden reward spike co-occurring with a KL blow-up.

This lens does **no model loading**. It consumes preference tuples,
log-prob arrays, and training metrics — the heavy training runs happen
in the Align framework, this lens observes them.

``trl``'s statistical helpers are used when available (``trl.trainer.utils``
for KL estimators); otherwise the lens falls back to closed-form
implementations (per ``rules/dependencies.md`` — fallback is permitted
because the loud optional-dep error is emitted only by the trl-exclusive
methods).

Quick start::

    from shared.mlfp06.diagnostics import AlignmentDiagnostics

    align = AlignmentDiagnostics()
    align.evaluate_pair(base_logprobs, tuned_logprobs, preferences)
    align.track_training(pipeline.metrics_stream())
    align.detect_reward_hacking(threshold=2.5)
    align.plot_alignment_dashboard().show()
    print(align.report())
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import plotly.graph_objects as go
import polars as pl

from . import _plots

logger = logging.getLogger(__name__)

__all__ = ["AlignmentDiagnostics"]


# ════════════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════════════


@dataclass
class _PairReading:
    label: str
    kl_divergence: float
    reward_margin: float
    win_rate: float
    n: int
    mode: str


@dataclass
class _TrainingStep:
    step: int
    reward: float
    kl: float
    loss: float
    extras: dict[str, Any]


@dataclass
class _HackFinding:
    step: int
    reward_zscore: float
    kl_value: float
    reward_value: float
    label: str


# ════════════════════════════════════════════════════════════════════════
# AlignmentDiagnostics
# ════════════════════════════════════════════════════════════════════════


class AlignmentDiagnostics:
    """Alignment-lens diagnostics — KL, reward margin, win-rate, hacking scan.

    Args:
        label: Short tag applied to every recorded reading (useful when
            comparing multiple runs side-by-side).
    """

    def __init__(self, *, label: str = "run") -> None:
        self._label = label
        self._pair_log: list[_PairReading] = []
        self._training_log: list[_TrainingStep] = []
        self._hack_findings: list[_HackFinding] = []
        logger.info("alignment_diagnostics.init", extra={"label": label})

    def __enter__(self) -> "AlignmentDiagnostics":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    # ── Pair evaluation ────────────────────────────────────────────────

    def evaluate_pair(
        self,
        base_policy: Sequence[Sequence[float]],
        tuned_policy: Sequence[Sequence[float]],
        preferences: Sequence[dict[str, Any]],
        *,
        label: str | None = None,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Compute KL(base || tuned), reward margin, and pair win-rate.

        Args:
            base_policy: Per-example token log-probabilities from the base
                model. Length ``N``, each element a sequence of log-probs.
            tuned_policy: Same shape, from the tuned model.
            preferences: Iterable of ``{chosen_reward, rejected_reward,
                chosen_won}`` dicts. ``chosen_won`` is a bool.
            label: Optional sub-label overriding the instance label.
            run_id: Correlation ID per ``rules/observability.md``.

        Returns:
            A one-row Polars DataFrame.
        """
        if len(base_policy) != len(tuned_policy):
            raise ValueError("base_policy and tuned_policy must be same length")
        run_id = run_id or f"align_pair-{uuid.uuid4().hex[:12]}"
        label = label or self._label

        kls = [_kl_from_logprobs(b, t) for b, t in zip(base_policy, tuned_policy)]
        kl_mean = _mean(kls)

        if preferences:
            margins = [
                float(p["chosen_reward"]) - float(p["rejected_reward"])
                for p in preferences
            ]
            wins = sum(1 for p in preferences if bool(p.get("chosen_won")))
            win_rate = wins / len(preferences)
            reward_margin = _mean(margins)
        else:
            win_rate = float("nan")
            reward_margin = float("nan")

        reading = _PairReading(
            label=label,
            kl_divergence=kl_mean,
            reward_margin=reward_margin,
            win_rate=win_rate,
            n=len(base_policy),
            mode="real",
        )
        self._pair_log.append(reading)
        logger.info(
            "alignment.evaluate_pair",
            extra={
                "run_id": run_id,
                "label": label,
                "kl": kl_mean,
                "reward_margin": reward_margin,
                "win_rate": win_rate,
                "n": len(base_policy),
                "mode": "real",
            },
        )
        return pl.DataFrame(
            [
                {
                    "label": reading.label,
                    "kl_divergence": reading.kl_divergence,
                    "reward_margin": reading.reward_margin,
                    "win_rate": reading.win_rate,
                    "n": reading.n,
                }
            ]
        )

    def kl_divergence(
        self,
        p_logprobs: Sequence[float],
        q_logprobs: Sequence[float],
    ) -> float:
        """KL(p || q) from token log-probabilities.

        When ``trl`` is installed and exposes
        ``trl.trainer.utils.kl_divergence``, that implementation is used;
        otherwise a closed-form estimator runs locally.
        """
        try:
            from trl.trainer.utils import kl_divergence as trl_kl  # type: ignore[import-not-found]
        except ImportError:
            return _kl_from_logprobs(p_logprobs, q_logprobs)
        try:
            import torch  # type: ignore[import-not-found]

            p = torch.tensor(list(p_logprobs))
            q = torch.tensor(list(q_logprobs))
            return float(trl_kl(p, q).mean().item())
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "alignment.trl_kl_error",
                extra={"error": str(exc), "mode": "real"},
            )
            return _kl_from_logprobs(p_logprobs, q_logprobs)

    def win_rate(self, preferences: Sequence[dict[str, Any]]) -> float:
        """Fraction of preference rows where the chosen policy won."""
        if not preferences:
            return float("nan")
        wins = sum(1 for p in preferences if bool(p.get("chosen_won")))
        return wins / len(preferences)

    # ── Training-curve tracking ────────────────────────────────────────

    def track_training(
        self,
        metrics: Iterable[dict[str, Any]] | Any,
        *,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Record a training-metrics stream from an Align pipeline.

        Accepts either:

            * an iterable of ``{step, reward, kl, loss, ...}`` dicts, or
            * a Kailash Align ``AlignmentPipeline`` exposing a
              ``metrics_stream()`` / ``.metrics`` attribute.

        Any missing field is defaulted to ``float("nan")`` so partial
        streams (e.g. SFT runs without a reward signal) still produce a
        usable DataFrame.
        """
        run_id = run_id or f"align_track-{uuid.uuid4().hex[:12]}"
        iterable = _resolve_metrics_iterable(metrics)
        rows: list[dict[str, Any]] = []
        for raw in iterable:
            step = int(raw.get("step", len(self._training_log)))
            reward = float(raw.get("reward", float("nan")))
            kl = float(raw.get("kl", raw.get("kl_divergence", float("nan"))))
            loss = float(raw.get("loss", float("nan")))
            extras = {
                k: v
                for k, v in raw.items()
                if k not in {"step", "reward", "kl", "kl_divergence", "loss"}
            }
            self._training_log.append(
                _TrainingStep(step=step, reward=reward, kl=kl, loss=loss, extras=extras)
            )
            rows.append({"step": step, "reward": reward, "kl": kl, "loss": loss})
        df = pl.DataFrame(rows) if rows else _empty_training_df()
        logger.info(
            "alignment.track_training",
            extra={
                "run_id": run_id,
                "steps": df.height,
                "source": "align_pipeline",
                "mode": "real",
            },
        )
        return df

    # ── Reward hacking detection ───────────────────────────────────────

    def detect_reward_hacking(
        self,
        history: Sequence[dict[str, Any]] | None = None,
        *,
        threshold: float = 2.5,
        label: str | None = None,
    ) -> pl.DataFrame:
        """Flag reward-spike + KL-blowup steps.

        Reward-hacking's canonical signature is a sudden jump in reward
        that coincides with a divergence blow-up — the model has learned
        a shortcut the reward model rewards but the base distribution
        doesn't support.

        Args:
            history: Optional pre-recorded training history. When
                ``None``, uses the history accumulated via
                :meth:`track_training`.
            threshold: Z-score above which a reward delta is flagged.
            label: Optional label applied to findings.

        Returns:
            DataFrame of findings (``step``, ``reward_zscore``,
            ``kl_value``, ``reward_value``).
        """
        if history is not None:
            series = [
                _TrainingStep(
                    step=int(h.get("step", i)),
                    reward=float(h.get("reward", float("nan"))),
                    kl=float(h.get("kl", float("nan"))),
                    loss=float(h.get("loss", float("nan"))),
                    extras={},
                )
                for i, h in enumerate(history)
            ]
        else:
            series = list(self._training_log)
        if len(series) < 4:
            return _empty_findings_df()

        rewards = [s.reward for s in series if not math.isnan(s.reward)]
        if len(rewards) < 4:
            return _empty_findings_df()
        mu = _mean(rewards)
        sigma = _stdev(rewards, mu) or 1e-9
        median_kl = _median([s.kl for s in series if not math.isnan(s.kl)])

        label = label or self._label
        findings: list[_HackFinding] = []
        for prev, cur in zip(series, series[1:]):
            if math.isnan(cur.reward) or math.isnan(prev.reward):
                continue
            delta = cur.reward - prev.reward
            z = delta / sigma
            if (
                z > threshold
                and not math.isnan(cur.kl)
                and cur.kl > max(median_kl * 1.5, 0.05)
            ):
                findings.append(
                    _HackFinding(
                        step=cur.step,
                        reward_zscore=z,
                        kl_value=cur.kl,
                        reward_value=cur.reward,
                        label=label,
                    )
                )

        self._hack_findings.extend(findings)
        if findings:
            logger.warning(
                "alignment.reward_hacking.detected",
                extra={
                    "label": label,
                    "n_findings": len(findings),
                    "threshold_z": threshold,
                },
            )
        return (
            pl.DataFrame(
                [
                    {
                        "step": f.step,
                        "reward_zscore": f.reward_zscore,
                        "kl_value": f.kl_value,
                        "reward_value": f.reward_value,
                        "label": f.label,
                    }
                    for f in findings
                ]
            )
            if findings
            else _empty_findings_df()
        )

    # ── DataFrames ─────────────────────────────────────────────────────

    def pair_df(self) -> pl.DataFrame:
        if not self._pair_log:
            return pl.DataFrame(
                schema={
                    "label": pl.Utf8,
                    "kl_divergence": pl.Float64,
                    "reward_margin": pl.Float64,
                    "win_rate": pl.Float64,
                    "n": pl.Int64,
                }
            )
        return pl.DataFrame(
            [
                {
                    "label": r.label,
                    "kl_divergence": r.kl_divergence,
                    "reward_margin": r.reward_margin,
                    "win_rate": r.win_rate,
                    "n": r.n,
                }
                for r in self._pair_log
            ]
        )

    def training_df(self) -> pl.DataFrame:
        if not self._training_log:
            return _empty_training_df()
        return pl.DataFrame(
            [
                {"step": s.step, "reward": s.reward, "kl": s.kl, "loss": s.loss}
                for s in self._training_log
            ]
        )

    def findings_df(self) -> pl.DataFrame:
        if not self._hack_findings:
            return _empty_findings_df()
        return pl.DataFrame(
            [
                {
                    "step": f.step,
                    "reward_zscore": f.reward_zscore,
                    "kl_value": f.kl_value,
                    "reward_value": f.reward_value,
                    "label": f.label,
                }
                for f in self._hack_findings
            ]
        )

    # ── Plots ──────────────────────────────────────────────────────────

    def plot_alignment_dashboard(self) -> go.Figure:
        """Reward curve, KL curve, win-rate bars; hacking findings highlighted."""
        if not self._training_log and not self._pair_log:
            return _plots.empty_figure("Alignment Lens Dashboard (ECG)")

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Reward over training steps",
                "KL divergence over training steps",
                "Pair win-rate",
                "Reward vs KL (hacking scan)",
            ),
        )

        train_df = self.training_df()
        if train_df.height:
            fig.add_trace(
                go.Scatter(
                    x=train_df["step"].to_list(),
                    y=train_df["reward"].to_list(),
                    mode="lines+markers",
                    line=dict(color=_plots.PRIMARY),
                    name="reward",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=train_df["step"].to_list(),
                    y=train_df["kl"].to_list(),
                    mode="lines+markers",
                    line=dict(color=_plots.WARN),
                    name="kl",
                ),
                row=1,
                col=2,
            )
            # Scatter colored by z-score flag.
            fig.add_trace(
                go.Scatter(
                    x=train_df["kl"].to_list(),
                    y=train_df["reward"].to_list(),
                    mode="markers",
                    marker=dict(
                        color=_plots.PRIMARY,
                        size=8,
                        line=dict(width=1, color=_plots.MUTED),
                    ),
                    name="steps",
                ),
                row=2,
                col=2,
            )
            findings_df = self.findings_df()
            if findings_df.height:
                fig.add_trace(
                    go.Scatter(
                        x=findings_df["kl_value"].to_list(),
                        y=findings_df["reward_value"].to_list(),
                        mode="markers",
                        marker=dict(
                            color=_plots.WARN, size=14, symbol="x", line=dict(width=2)
                        ),
                        name="hack flag",
                    ),
                    row=2,
                    col=2,
                )

        pair_df = self.pair_df()
        if pair_df.height:
            fig.add_trace(
                go.Bar(
                    x=pair_df["label"].to_list(),
                    y=pair_df["win_rate"].to_list(),
                    marker_color=_plots.PRIMARY,
                    name="win_rate",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title="Alignment Lens Dashboard (ECG)",
            template=_plots.TEMPLATE,
            showlegend=False,
            height=640,
        )
        return fig

    # ── Report ─────────────────────────────────────────────────────────

    def report(self) -> str:
        """Plain-text Prescription Pad for the alignment lens."""
        out: list[str] = []
        pair_df = self.pair_df()
        if pair_df.height:
            for row in pair_df.iter_rows(named=True):
                out.append(
                    f"pair[{row['label']}]: KL={row['kl_divergence']:.3f}, "
                    f"margin={row['reward_margin']:.3f}, "
                    f"win_rate={row['win_rate']:.0%} (n={row['n']})"
                )
                if row["kl_divergence"] > 1.0:
                    out.append("  -> KL above 1.0 — tuned model drifted far from base")
                if row["reward_margin"] < 0.05:
                    out.append("  -> margin below 0.05 — preference signal is too weak")

        train_df = self.training_df()
        if train_df.height:
            mean_r = float(train_df["reward"].mean() or 0.0)
            mean_kl = float(train_df["kl"].mean() or 0.0)
            out.append(
                f"training: {train_df.height} steps, mean reward={mean_r:.3f}, "
                f"mean KL={mean_kl:.3f}"
            )

        findings_df = self.findings_df()
        if findings_df.height:
            out.append(
                f"hacking scan: {findings_df.height} suspected step(s) — "
                f"reward spike + KL blow-up"
            )
            top = findings_df.row(0, named=True)
            out.append(
                f"  -> step {top['step']}: z={top['reward_zscore']:.2f}, "
                f"kl={top['kl_value']:.3f}"
            )

        if not out:
            return "alignment-lens: no readings recorded yet."
        return "alignment-lens:\n  " + "\n  ".join(out)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: Sequence[float], mu: float) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return 0.0
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _median(xs: Sequence[float]) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _kl_from_logprobs(
    p_logprobs: Sequence[float], q_logprobs: Sequence[float]
) -> float:
    """KL(P || Q) estimator from paired token log-probabilities.

    Given log p(x_t) and log q(x_t) for the same sequence, the Monte
    Carlo estimator is E_{x ~ P}[log p - log q] ≈ mean(p_logprobs -
    q_logprobs). We also clip the ratio to avoid inf.
    """
    n = min(len(p_logprobs), len(q_logprobs))
    if n == 0:
        return 0.0
    diffs = []
    for i in range(n):
        d = float(p_logprobs[i]) - float(q_logprobs[i])
        # Clip extreme values so a single bad token can't dominate.
        d = max(-50.0, min(50.0, d))
        diffs.append(d)
    return sum(diffs) / n


def _resolve_metrics_iterable(obj: Any) -> Iterable[dict[str, Any]]:
    """Extract a metrics iterable from an ``AlignmentPipeline`` or a list."""
    if obj is None:
        return []
    if hasattr(obj, "metrics_stream"):
        return obj.metrics_stream()
    if hasattr(obj, "metrics"):
        return list(obj.metrics)
    return obj  # assume iterable of dicts


def _empty_training_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "step": pl.Int64,
            "reward": pl.Float64,
            "kl": pl.Float64,
            "loss": pl.Float64,
        }
    )


def _empty_findings_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "step": pl.Int64,
            "reward_zscore": pl.Float64,
            "kl_value": pl.Float64,
            "reward_value": pl.Float64,
            "label": pl.Utf8,
        }
    )
