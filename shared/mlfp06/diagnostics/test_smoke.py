# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the six-lens LLM Observatory.

Runs WITHOUT API keys, real Kaizen delegates, transformer_lens, or PACT
engines — uses ``unittest.mock`` fakes only. The goal is to prove that:

    1. ``LLMObservatory`` instantiates with zero real dependencies.
    2. Every lens is wired (six named attributes present).
    3. ``report()`` returns the expected six-key dict.
    4. ``plot_dashboard()`` returns a Plotly ``Figure`` even with no data.
    5. Every lens report carries a canonical ``severity`` field.
    6. Context-manager ``__enter__`` / ``__exit__`` are wired.
    7. Governance lens raises loudly (not silently) when called without
       an engine (per rules/observability.md §3.1 — no silent fake
       verdicts).

Run directly::

    .venv/bin/python shared/mlfp06/diagnostics/test_smoke.py

Or via pytest::

    .venv/bin/python -m pytest shared/mlfp06/diagnostics/test_smoke.py -x
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# Set a dummy judge model so resolve_judge_model() doesn't blow up during
# lazy construction. JudgeCallable resolves lazily, so construction is
# safe, but pytest-run tests that trigger attention model defaults need
# no env vars.
os.environ.setdefault("OPENAI_JUDGE_MODEL", "gpt-smoke-test")


# ─────────────────────────────────────────────────────────────────────
# Imports under test
# ─────────────────────────────────────────────────────────────────────

from shared.mlfp06.diagnostics import (  # noqa: E402
    AgentDiagnostics,
    AlignmentDiagnostics,
    GovernanceDiagnostics,
    InterpretabilityDiagnostics,
    LLMDiagnostics,
    LLMObservatory,
    RAGDiagnostics,
)

# ─────────────────────────────────────────────────────────────────────
# Canonical severity set per observatory.py
# ─────────────────────────────────────────────────────────────────────

SEVERITIES = {"HEALTHY", "WARNING", "CRITICAL", "UNKNOWN"}
LENS_NAMES = {"output", "attention", "retrieval", "agent", "alignment", "governance"}


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_observatory_instantiates() -> None:
    """LLMObservatory() with no args must wire all six lenses."""
    obs = LLMObservatory()
    assert obs.output is not None and isinstance(obs.output, LLMDiagnostics)
    assert obs.attention is not None and isinstance(
        obs.attention, InterpretabilityDiagnostics
    )
    assert obs.retrieval is not None and isinstance(obs.retrieval, RAGDiagnostics)
    assert obs.agent is not None and isinstance(obs.agent, AgentDiagnostics)
    assert obs.alignment is not None and isinstance(obs.alignment, AlignmentDiagnostics)
    assert obs.governance is not None and isinstance(
        obs.governance, GovernanceDiagnostics
    )
    obs.close()


def test_observatory_with_mocks() -> None:
    """LLMObservatory should accept mock delegate + governance without crashing."""
    fake_delegate = MagicMock(name="FakeDelegate")
    fake_governance = MagicMock(name="FakeGovernanceEngine")
    # Give the fake governance engine a minimal audit_entries() contract.
    fake_governance.audit_entries.return_value = []
    fake_governance.budgets = {}

    obs = LLMObservatory(
        delegate=fake_delegate,
        governance=fake_governance,
        run_id="smoke-run",
        max_judge_calls=5,
    )
    assert obs.output is not None
    assert obs.governance is not None
    obs.close()


def test_report_returns_dict() -> None:
    """report() must return a dict with exactly the six lens keys."""
    obs = LLMObservatory()
    result = obs.report()
    assert isinstance(result, dict), f"expected dict, got {type(result)}"
    assert (
        set(result.keys()) == LENS_NAMES
    ), f"expected keys {LENS_NAMES}, got {set(result.keys())}"
    obs.close()


def test_dashboard_returns_figure() -> None:
    """plot_dashboard() must return a Plotly Figure, even with no data."""
    import plotly.graph_objects as go

    obs = LLMObservatory()
    fig = obs.plot_dashboard()
    assert isinstance(fig, go.Figure), f"expected go.Figure, got {type(fig)}"
    obs.close()


def test_each_lens_report_has_severity() -> None:
    """Every lens report dict must have a canonical severity field."""
    obs = LLMObservatory()
    r = obs.report()
    for lens_name, lens_report in r.items():
        assert isinstance(
            lens_report, dict
        ), f"{lens_name} report is {type(lens_report)}, expected dict"
        assert "severity" in lens_report, (
            f"{lens_name} report missing 'severity' key; got keys: "
            f"{list(lens_report.keys())}"
        )
        assert (
            lens_report["severity"] in SEVERITIES
        ), f"{lens_name} severity={lens_report['severity']!r} not in {SEVERITIES}"
        assert "summary" in lens_report
        assert isinstance(lens_report["summary"], str)
    obs.close()


def test_context_manager() -> None:
    """with LLMObservatory() as obs: must return the observatory itself."""
    with LLMObservatory() as obs:
        assert isinstance(obs, LLMObservatory)
        assert obs.output is not None
    # After __exit__, close has been called; subsequent .report() should
    # still be defensible (no exception — lenses tolerate post-close).


def test_governance_loud_when_no_engine() -> None:
    """Governance lens must raise (not silently fake) when no engine is supplied."""
    obs = LLMObservatory()
    try:
        obs.governance.audit_snapshot(last_n=10)
    except RuntimeError as exc:
        # Expected — loud error per rules/observability.md §3.1.
        assert "governance engine" in str(exc).lower()
    else:
        raise AssertionError(
            "GovernanceDiagnostics.audit_snapshot should have raised "
            "RuntimeError when no engine is configured"
        )
    finally:
        obs.close()


def test_governance_report_with_no_engine() -> None:
    """Governance report MUST degrade gracefully (text) when no engine is set."""
    obs = LLMObservatory()
    r = obs.report()
    gov = r["governance"]
    assert gov["severity"] == "UNKNOWN"
    assert "no engine" in gov["summary"].lower()
    obs.close()


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────


def _main() -> int:
    tests = [
        test_observatory_instantiates,
        test_observatory_with_mocks,
        test_report_returns_dict,
        test_dashboard_returns_figure,
        test_each_lens_report_has_severity,
        test_context_manager,
        test_governance_loud_when_no_engine,
        test_governance_report_with_no_engine,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as exc:  # noqa: BLE001 — smoke runner wants the traceback
            failures += 1
            print(f"  FAIL {t.__name__}: {exc!r}")
            import traceback

            traceback.print_exc()
    if failures:
        print(f"\n{failures} SMOKE TEST(S) FAILED")
        return 1
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
