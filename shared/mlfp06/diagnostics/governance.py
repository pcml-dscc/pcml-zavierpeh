# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Lens 6 — Governance Diagnostics (the Flight Recorder).

Question answered: *Is the system operating inside its envelope, and is
the audit chain intact?*

Wraps a PACT ``GovernanceEngine`` (or the ``audit`` facet of a Kaizen
``GovernedSupervisor``). This lens is NEVER responsible for making
governance decisions — it only *reads* what the engine has already
recorded. Per ``rules/framework-first.md`` MANDATORY: policy / envelope
construction is the engine's job, not ours.

Typical use::

    from shared.mlfp06.diagnostics import GovernanceDiagnostics

    diag = GovernanceDiagnostics(governance=supervisor.audit)
    audit_df = diag.audit_snapshot(last_n=200)
    chain_ok = diag.verify_chain(audit_df)
    budget_df = diag.budget_consumption()
    diag.plot_governance_dashboard().show()
    print(diag.report())
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import plotly.graph_objects as go
import polars as pl

from . import _plots

logger = logging.getLogger(__name__)

__all__ = ["GovernanceDiagnostics"]


# ════════════════════════════════════════════════════════════════════════
# Internal row shape (what we harvest from the engine)
# ════════════════════════════════════════════════════════════════════════


@dataclass
class _DrillResult:
    scenario: str
    verdict: str
    reason: str
    mode: str


# Canonical envelope dimensions — matches PACT's envelope taxonomy.
_ENVELOPE_DIMENSIONS: tuple[str, ...] = (
    "financial",
    "temporal",
    "data_access",
    "communication",
    "operational",
)

_VERDICT_COLORS = {
    "allow": _plots.PRIMARY,
    "warn": _plots.ACCENT,
    "block": _plots.WARN,
    "escalate": "mediumpurple",
    "unknown": _plots.MUTED,
}


class GovernanceDiagnostics:
    """Governance-lens diagnostics — audit snapshot, chain verification, drills.

    Args:
        governance: The governance engine under observation. Accepts:

            * PACT ``GovernanceEngine`` (``.verify_action`` + audit log)
            * Kaizen ``GovernedSupervisor.audit`` facet
            * Any object exposing ``audit_entries`` / ``to_list`` or a
              callable ``verify_action``.

            When ``None``, every data method raises a loud error (per
            ``rules/observability.md`` §3 — no silent fake verdicts).
        run_id: Optional correlation ID.
    """

    def __init__(
        self,
        *,
        governance: Any = None,
        run_id: str | None = None,
    ) -> None:
        self._engine = governance
        self._run_id = run_id
        self._drill_results: list[_DrillResult] = []
        logger.info(
            "governance_diagnostics.init",
            extra={
                "engine_type": type(governance).__name__ if governance else "none",
                "run_id": run_id,
            },
        )

    def __enter__(self) -> "GovernanceDiagnostics":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release any engine handles (best-effort)."""
        # The engine itself is owned by the caller — do not close it.
        self._drill_results = []

    # ── Helpers ────────────────────────────────────────────────────────

    def _require_engine(self, op: str) -> Any:
        if self._engine is None:
            raise RuntimeError(
                f"GovernanceDiagnostics.{op}() requires a governance engine. "
                "Pass governance=PACT GovernanceEngine or "
                "governance=supervisor.audit to the constructor. Silent "
                "degradation to fake verdicts is BLOCKED per "
                "rules/observability.md §3."
            )
        return self._engine

    def _extract_audit_rows(self, engine: Any) -> list[dict[str, Any]]:
        """Normalise whatever the engine exposes into a flat list of dicts.

        Recognised shapes (checked in order):

            * ``engine.audit_entries`` (PACT GovernanceEngine canonical)
            * ``engine.to_list()`` (Kaizen audit facet)
            * ``engine.entries`` (generic list)
            * iterable of dicts / dataclasses
        """
        source: Iterable[Any]
        if hasattr(engine, "audit_entries"):
            source = (
                engine.audit_entries()
                if callable(engine.audit_entries)
                else engine.audit_entries
            )
        elif hasattr(engine, "to_list"):
            source = engine.to_list()
        elif hasattr(engine, "entries"):
            source = engine.entries
        elif isinstance(engine, (list, tuple)):
            source = engine
        else:
            raise TypeError(
                f"GovernanceDiagnostics cannot read audit log from "
                f"{type(engine).__name__}. Expected .audit_entries, "
                f".to_list(), .entries, or a list."
            )

        rows: list[dict[str, Any]] = []
        for raw in source:
            if isinstance(raw, dict):
                rows.append(raw)
            elif hasattr(raw, "__dict__"):
                # Dataclass or plain object.
                rows.append(
                    {k: v for k, v in vars(raw).items() if not k.startswith("_")}
                )
            else:
                logger.warning(
                    "governance.audit.skip_unknown_row",
                    extra={"row_type": type(raw).__name__},
                )
        return rows

    # ── Audit snapshot ─────────────────────────────────────────────────

    def audit_snapshot(self, last_n: int = 100) -> pl.DataFrame:
        """Return the last ``last_n`` envelope decisions as a Polars frame.

        Columns (best-effort, missing fields become nulls):

            * ``timestamp`` — ISO-8601 str or float epoch, whatever the
              engine records.
            * ``subject`` — Director / Tenant / Role (PACT D/T/R).
            * ``action`` — The proposed action string.
            * ``verdict`` — ``allow`` / ``warn`` / ``block`` / ``escalate``.
            * ``reason`` — Short rationale from the engine.
            * ``hash`` / ``prev_hash`` — If the engine records them.
        """
        engine = self._require_engine("audit_snapshot")
        rows = self._extract_audit_rows(engine)
        if last_n > 0 and len(rows) > last_n:
            rows = rows[-last_n:]

        # Canonicalise the key names we care about. Accept several aliases.
        canonical: list[dict[str, Any]] = []
        for r in rows:
            canonical.append(
                {
                    "timestamp": r.get("timestamp") or r.get("ts") or r.get("time"),
                    "subject": r.get("subject") or r.get("director") or r.get("actor"),
                    "action": r.get("action") or r.get("operation"),
                    "verdict": (r.get("verdict") or r.get("decision") or "unknown"),
                    "reason": r.get("reason") or r.get("rationale") or "",
                    "hash": r.get("hash"),
                    "prev_hash": r.get("prev_hash"),
                }
            )

        logger.info(
            "governance.audit_snapshot",
            extra={
                "run_id": self._run_id,
                "rows": len(canonical),
                "source": "governance_engine",
                "mode": "real",
            },
        )
        if not canonical:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Utf8,
                    "subject": pl.Utf8,
                    "action": pl.Utf8,
                    "verdict": pl.Utf8,
                    "reason": pl.Utf8,
                    "hash": pl.Utf8,
                    "prev_hash": pl.Utf8,
                }
            )
        # Polars infers types; coerce timestamps / subjects to string for display.
        return pl.DataFrame(canonical).with_columns(
            [
                pl.col("timestamp").cast(pl.Utf8, strict=False),
                pl.col("subject").cast(pl.Utf8, strict=False),
                pl.col("action").cast(pl.Utf8, strict=False),
                pl.col("verdict").cast(pl.Utf8, strict=False),
                pl.col("reason").cast(pl.Utf8, strict=False),
                pl.col("hash").cast(pl.Utf8, strict=False),
                pl.col("prev_hash").cast(pl.Utf8, strict=False),
            ]
        )

    # ── Chain integrity ────────────────────────────────────────────────

    def verify_chain(
        self, audit_log: pl.DataFrame | Sequence[dict[str, Any]]
    ) -> pl.DataFrame:
        """Check hash-chain integrity.

        Contract: each row's ``hash`` must equal ``sha256(prev_hash +
        canonical_json(row_data))``. Rows lacking ``hash`` / ``prev_hash``
        are skipped and flagged as ``unchecked``.

        Returns a Polars DataFrame with one row per audit entry and a
        per-row ``ok`` / ``broken`` / ``unchecked`` verdict. The caller
        can ``filter(pl.col("integrity") == "broken")`` to surface tampering.
        """
        if isinstance(audit_log, pl.DataFrame):
            rows = audit_log.to_dicts()
        else:
            rows = list(audit_log)

        report_rows: list[dict[str, Any]] = []
        n_broken = 0
        n_ok = 0
        n_unchecked = 0
        expected_prev: str | None = None

        for idx, r in enumerate(rows):
            h = r.get("hash")
            prev = r.get("prev_hash")
            if not h:
                integrity = "unchecked"
                n_unchecked += 1
            else:
                payload = {k: v for k, v in r.items() if k not in {"hash"}}
                # Canonical JSON — sort keys, no whitespace, default=str so
                # datetimes round-trip.
                serialised = json.dumps(payload, sort_keys=True, default=str).encode(
                    "utf-8"
                )
                recomputed = hashlib.sha256(
                    ((prev or "").encode("utf-8") + serialised)
                ).hexdigest()
                chain_ok = recomputed == h
                link_ok = (expected_prev is None) or (prev == expected_prev)
                if chain_ok and link_ok:
                    integrity = "ok"
                    n_ok += 1
                else:
                    integrity = "broken"
                    n_broken += 1
                expected_prev = h

            report_rows.append(
                {
                    "idx": idx,
                    "timestamp": r.get("timestamp"),
                    "hash": h,
                    "prev_hash": prev,
                    "integrity": integrity,
                }
            )

        logger.info(
            "governance.verify_chain",
            extra={
                "run_id": self._run_id,
                "rows": len(rows),
                "ok": n_ok,
                "broken": n_broken,
                "unchecked": n_unchecked,
                "source": "local_metric",
                "mode": "real",
            },
        )
        if n_broken:
            logger.warning(
                "governance.chain_broken",
                extra={"run_id": self._run_id, "broken": n_broken, "total": len(rows)},
            )

        if not report_rows:
            return pl.DataFrame(
                schema={
                    "idx": pl.Int64,
                    "timestamp": pl.Utf8,
                    "hash": pl.Utf8,
                    "prev_hash": pl.Utf8,
                    "integrity": pl.Utf8,
                }
            )
        return pl.DataFrame(report_rows).with_columns(
            [
                pl.col("timestamp").cast(pl.Utf8, strict=False),
                pl.col("hash").cast(pl.Utf8, strict=False),
                pl.col("prev_hash").cast(pl.Utf8, strict=False),
            ]
        )

    # ── Budget consumption ─────────────────────────────────────────────

    def budget_consumption(self) -> pl.DataFrame:
        """Aggregate envelope usage per dimension into a Polars DataFrame.

        Columns:

            * ``dimension`` — one of ``financial``, ``temporal``,
              ``data_access``, ``communication``, ``operational``.
            * ``limit`` — envelope cap as declared by the engine.
            * ``consumed`` — amount consumed so far.
            * ``remaining`` — ``limit - consumed`` (clipped at 0).
            * ``pct_used`` — ``consumed / limit`` in ``[0, 1]`` (``nan`` if
              the limit is unset).
        """
        engine = self._require_engine("budget_consumption")

        # Canonical shape 1: engine exposes ``.budgets`` dict
        # {"financial": {"limit": X, "consumed": Y}, ...}.
        budgets: dict[str, dict[str, float]] = {}
        if hasattr(engine, "budgets"):
            raw = engine.budgets() if callable(engine.budgets) else engine.budgets
            if isinstance(raw, dict):
                budgets = {
                    str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)
                }

        # Canonical shape 2: derive from audit log — count rows by dimension
        # with a "consumed" / "amount" field.
        if not budgets:
            rows = self._extract_audit_rows(engine)
            tallies: dict[str, float] = {d: 0.0 for d in _ENVELOPE_DIMENSIONS}
            for r in rows:
                dim = r.get("dimension") or r.get("envelope_dimension")
                amount = r.get("amount") or r.get("consumed") or 0.0
                if dim in tallies:
                    try:
                        tallies[dim] += float(amount)
                    except (TypeError, ValueError):
                        continue
            budgets = {
                d: {"limit": float("nan"), "consumed": tallies[d]}
                for d in _ENVELOPE_DIMENSIONS
            }

        rows_out: list[dict[str, Any]] = []
        for dim in _ENVELOPE_DIMENSIONS:
            b = budgets.get(dim, {})
            limit = float(b.get("limit", float("nan")))
            consumed = float(b.get("consumed", 0.0))
            remaining = max(0.0, limit - consumed) if limit == limit else float("nan")
            pct = (consumed / limit) if (limit == limit and limit > 0) else float("nan")
            rows_out.append(
                {
                    "dimension": dim,
                    "limit": limit,
                    "consumed": consumed,
                    "remaining": remaining,
                    "pct_used": pct,
                }
            )

        logger.info(
            "governance.budget_consumption",
            extra={
                "run_id": self._run_id,
                "dimensions": len(rows_out),
                "source": "governance_engine",
                "mode": "real",
            },
        )
        return pl.DataFrame(rows_out)

    # ── Negative drills ────────────────────────────────────────────────

    def negative_drills(self, scenarios: Sequence[dict[str, Any]]) -> pl.DataFrame:
        """Run red-team scenarios through ``GovernanceEngine.verify_action``.

        Each scenario dict is passed as ``**kwargs`` to the engine's
        ``verify_action`` (or ``verify``) method. The returned object's
        ``verdict`` / ``decision`` field is recorded.

        Args:
            scenarios: List of kwarg dicts describing the action to test.
                e.g. ``[{"subject": "agent-1", "action": "transfer",
                "amount": 1_000_000, "dimension": "financial"}]``.

        Returns:
            A Polars DataFrame with one row per scenario: ``scenario``,
            ``verdict``, ``reason``, ``mode``.
        """
        engine = self._require_engine("negative_drills")
        verify = getattr(engine, "verify_action", None) or getattr(
            engine, "verify", None
        )
        if verify is None or not callable(verify):
            raise AttributeError(
                "Governance engine does not expose verify_action(). "
                "Pass a PACT GovernanceEngine or a GovernedSupervisor.audit "
                "that supports verification."
            )

        results: list[_DrillResult] = []
        for i, sc in enumerate(scenarios):
            label = sc.get("label") or sc.get("action") or f"drill_{i}"
            try:
                outcome = verify(**sc) if isinstance(sc, dict) else verify(sc)
            except Exception as exc:
                logger.exception(
                    "governance.drill.error",
                    extra={"scenario": label, "error": str(exc)},
                )
                results.append(
                    _DrillResult(
                        scenario=label,
                        verdict="error",
                        reason=str(exc),
                        mode="real",
                    )
                )
                continue

            verdict = _extract_field(
                outcome, ("verdict", "decision"), default="unknown"
            )
            reason = _extract_field(outcome, ("reason", "rationale"), default="")
            results.append(
                _DrillResult(
                    scenario=label,
                    verdict=str(verdict).lower(),
                    reason=str(reason),
                    mode="real",
                )
            )

        self._drill_results.extend(results)
        logger.info(
            "governance.negative_drills",
            extra={
                "run_id": self._run_id,
                "scenarios": len(scenarios),
                "source": "governance_engine",
                "mode": "real",
            },
        )
        return pl.DataFrame(
            [
                {
                    "scenario": r.scenario,
                    "verdict": r.verdict,
                    "reason": r.reason,
                    "mode": r.mode,
                }
                for r in results
            ]
        )

    # ── Plot ───────────────────────────────────────────────────────────

    def plot_governance_dashboard(self) -> go.Figure:
        """2x2 dashboard: verdict-over-time, budget bars, drill heatmap, chain timeline."""
        from plotly.subplots import make_subplots

        if self._engine is None:
            return _plots.empty_figure(
                "Governance Lens Dashboard (Flight Recorder)",
                note="no governance engine",
            )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Verdicts over time (stacked)",
                "Envelope budget consumption",
                "Negative-drill verdict mix",
                "Chain integrity timeline",
            ),
        )

        # (1,1) Verdict-over-time: stacked by verdict.
        try:
            snap = self.audit_snapshot(last_n=500)
        except Exception as exc:  # engine refused
            logger.warning(
                "governance.plot.snapshot_skipped", extra={"error": str(exc)}
            )
            snap = pl.DataFrame()
        if snap.height:
            verdicts_seen = snap["verdict"].unique().to_list()
            for verdict in verdicts_seen:
                sub = snap.filter(pl.col("verdict") == verdict)
                fig.add_trace(
                    go.Scatter(
                        x=(
                            sub["timestamp"].to_list()
                            if "timestamp" in sub.columns
                            else list(range(sub.height))
                        ),
                        y=[1] * sub.height,
                        mode="markers",
                        marker=dict(
                            color=_VERDICT_COLORS.get(verdict, _plots.MUTED),
                            size=8,
                        ),
                        name=verdict,
                        stackgroup="verdicts",
                    ),
                    row=1,
                    col=1,
                )

        # (1,2) Budget consumption bars.
        try:
            budget_df = self.budget_consumption()
        except Exception as exc:
            logger.warning("governance.plot.budget_skipped", extra={"error": str(exc)})
            budget_df = pl.DataFrame()
        if budget_df.height:
            fig.add_trace(
                go.Bar(
                    x=budget_df["dimension"].to_list(),
                    y=budget_df["consumed"].to_list(),
                    marker_color=_plots.PRIMARY,
                    name="consumed",
                ),
                row=1,
                col=2,
            )

        # (2,1) Negative drill verdict mix.
        if self._drill_results:
            counts: dict[str, int] = {}
            for r in self._drill_results:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
            fig.add_trace(
                go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=[_VERDICT_COLORS.get(v, _plots.MUTED) for v in counts],
                    name="drills",
                ),
                row=2,
                col=1,
            )

        # (2,2) Chain integrity timeline.
        if snap.height:
            chain_df = self.verify_chain(snap)
            color_map = {
                "ok": _plots.PRIMARY,
                "broken": _plots.WARN,
                "unchecked": _plots.MUTED,
            }
            fig.add_trace(
                go.Scatter(
                    x=chain_df["idx"].to_list(),
                    y=[1] * chain_df.height,
                    mode="markers",
                    marker=dict(
                        color=[
                            color_map.get(v, _plots.MUTED)
                            for v in chain_df["integrity"].to_list()
                        ],
                        size=10,
                        symbol="square",
                    ),
                    name="chain",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Governance Lens Dashboard (Flight Recorder)",
            template=_plots.TEMPLATE,
            showlegend=True,
            height=640,
        )
        return fig

    # ── Report ─────────────────────────────────────────────────────────

    def report(self) -> str:
        """Plain-text Prescription Pad for the governance lens."""
        if self._engine is None:
            return (
                "governance-lens: no engine configured — pass "
                "governance=GovernanceEngine(...) to read envelope state."
            )

        out: list[str] = []
        try:
            snap = self.audit_snapshot(last_n=200)
        except Exception as exc:
            return f"governance-lens: snapshot error — {exc}"

        if snap.height:
            verdict_counts = (
                snap.group_by("verdict")
                .agg(pl.len().alias("n"))
                .sort("n", descending=True)
            )
            parts = [
                f"{row['verdict']}={row['n']}"
                for row in verdict_counts.iter_rows(named=True)
            ]
            out.append(f"audit: {snap.height} entries — {', '.join(parts)}")

            chain_df = self.verify_chain(snap)
            n_broken = chain_df.filter(pl.col("integrity") == "broken").height
            n_unchecked = chain_df.filter(pl.col("integrity") == "unchecked").height
            if n_broken:
                out.append(
                    f"  -> CHAIN BROKEN at {n_broken} row(s) — possible tampering"
                )
            elif n_unchecked == chain_df.height:
                out.append("  -> chain unchecked (no hash/prev_hash columns present)")
            else:
                out.append(
                    f"  -> chain ok ({chain_df.height - n_broken - n_unchecked} linked)"
                )

        try:
            budget_df = self.budget_consumption()
        except Exception as exc:
            budget_df = None
            out.append(f"budget: unavailable — {exc}")
        if budget_df is not None and budget_df.height:
            for row in budget_df.iter_rows(named=True):
                pct = row["pct_used"]
                if pct == pct and pct >= 0.8:  # not NaN and >= 80%
                    out.append(
                        f"  -> envelope[{row['dimension']}] at {pct:.0%} — "
                        "approaching cap"
                    )

        if self._drill_results:
            blocked = sum(
                1 for r in self._drill_results if r.verdict in {"block", "escalate"}
            )
            total = len(self._drill_results)
            out.append(
                f"drills: {total} scenarios — {blocked}/{total} blocked/escalated"
            )
            if blocked < total:
                out.append(
                    "  -> some drills passed — inspect scenarios that were allowed"
                )

        if not out:
            return "governance-lens: no readings recorded yet."
        return "governance-lens:\n  " + "\n  ".join(out)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _extract_field(obj: Any, keys: Sequence[str], *, default: Any = None) -> Any:
    """Pull the first matching key from a dict / dataclass / attr object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        return default
    for k in keys:
        if hasattr(obj, k):
            return getattr(obj, k)
    return default
