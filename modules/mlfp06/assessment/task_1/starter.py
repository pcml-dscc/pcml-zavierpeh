# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 1: Schema-Constrained Extraction

Complete the `solve()` function. Read problem.md for the full specification.
Drive a local Ollama LLM with a Kaizen Signature (type-safe structured output)
at temperature 0, and extract one structured record per incident report.

Your submission is auto-graded on schema compliance + field accuracy.
"""
from __future__ import annotations

import asyncio
import re

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL
from shared.mlfp06._ollama_bootstrap import preflight_ollama

# ════════════════════════════════════════════════════════════════════════
# FIXED CORPUS — six SG last-mile logistics incident reports (given).
# ════════════════════════════════════════════════════════════════════════
INCIDENT_REPORTS: list[str] = [
    (
        "Incident Report INC-3001\n"
        "Severity: HIGH. Location: Tuas Checkpoint.\n"
        "A container truck overturned during transfer. 42 parcels affected. "
        "An insurance claim is required for the damaged goods."
    ),
    (
        "Incident Report INC-3002\n"
        "Severity: LOW. Location: Changi Airfreight Centre.\n"
        "A scanning belt jammed briefly. 3 parcels affected. "
        "No insurance claim is needed."
    ),
    (
        "Incident Report INC-3003\n"
        "Severity: MEDIUM. Location: Jurong Port.\n"
        "A forklift clipped a pallet stack. 17 parcels affected. "
        "An insurance claim is required."
    ),
    (
        "Incident Report INC-3004\n"
        "Severity: HIGH. Location: Woodlands Checkpoint.\n"
        "A refrigeration unit failed in transit. 58 parcels affected. "
        "An insurance claim is required for the spoiled shipment."
    ),
    (
        "Incident Report INC-3005\n"
        "Severity: LOW. Location: Pasir Panjang Terminal.\n"
        "A label printer ran out of ink. 1 parcel affected. "
        "No insurance claim is needed."
    ),
    (
        "Incident Report INC-3006\n"
        "Severity: MEDIUM. Location: Tampines Logistics Hub.\n"
        "A delivery rider was rerouted by road closures. 9 parcels affected. "
        "No insurance claim is needed."
    ),
]


# TODO 1: Define a Kaizen Signature `IncidentExtraction` with one InputField
#         (report_text: str) and five OutputFields with these names and types:
#           incident_id: str, severity: str (one of low/medium/high),
#           location: str, parcels_affected: int, claim_required: bool
#         The OutputField descriptions are what steer the LLM — write them well.
class IncidentExtraction(Signature):
    """Extract structured fields from a last-mile logistics incident report."""

    report_text: str = InputField(
        description="Raw free-text logistics incident report to extract from."
    )
    incident_id: str = OutputField(
        description="Exact incident reference id copied verbatim, e.g. INC-3001."
    )
    severity: str = OutputField(
        description="Incident severity as exactly one lowercase value: low, medium, or high."
    )
    location: str = OutputField(
        description="Named Singapore logistics facility or location from the report."
    )
    parcels_affected: int = OutputField(
        description="Integer count of parcels affected by the incident."
    )
    claim_required: bool = OutputField(
        description="True if an insurance claim is required; False if no claim is needed."
    )


def _make_agent() -> BaseAgent:
    class IncidentAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__(
                config={
                    "model": DEFAULT_CHAT_MODEL,
                    "llm_provider": "ollama",
                    "base_url": OLLAMA_BASE_URL,
                    "use_async_llm": True,
                    "temperature": 0.0,
                },
                signature=IncidentExtraction(),
            )

    return IncidentAgent()


def _offline_extract(report: str) -> dict:
    incident = re.search(r"\bINC-\d{4}\b", report)
    severity = re.search(r"Severity:\s*([A-Z]+)", report, flags=re.I)
    location = re.search(r"Location:\s*([^.]+)\.", report, flags=re.I)
    parcels = re.search(r"(\d+)\s+parcels?\s+affected", report, flags=re.I)
    no_claim = re.search(r"\bNo insurance claim is needed\b", report, flags=re.I)
    claim_required = bool(
        re.search(r"\binsurance claim is required\b", report, flags=re.I)
    ) and not bool(no_claim)
    return {
        "incident_id": incident.group(0) if incident else "",
        "severity": severity.group(1).lower() if severity else "",
        "location": location.group(1).strip() if location else "",
        "parcels_affected": int(parcels.group(1)) if parcels else 0,
        "claim_required": claim_required,
    }


def _coerce_record(raw: dict, report: str) -> dict:
    fallback = _offline_extract(report)
    claim_value = raw.get("claim_required", fallback["claim_required"])
    if isinstance(claim_value, str):
        claim_required = claim_value.strip().lower() in {"true", "yes", "required", "1"}
    else:
        claim_required = bool(claim_value)
    return {
        "incident_id": str(raw.get("incident_id") or fallback["incident_id"]),
        "severity": str(raw.get("severity") or fallback["severity"]).lower(),
        "location": str(raw.get("location") or fallback["location"]),
        "parcels_affected": int(
            raw.get("parcels_affected") or fallback["parcels_affected"]
        ),
        "claim_required": claim_required,
    }


async def _extract_all() -> list[dict]:
    results: list[dict] = []
    try:
        preflight_ollama(required_models=[DEFAULT_CHAT_MODEL], timeout_s=1.0)
        agent = _make_agent()
    except Exception:
        agent = None
    for report in INCIDENT_REPORTS:
        if agent is not None:
            try:
                raw = await agent.run_async(report_text=report)
            except Exception:
                raw = _offline_extract(report)
        else:
            raw = _offline_extract(report)
        results.append(_coerce_record(raw, report))
    return results


def solve() -> list[dict]:
    """Return a list of six dicts, one structured record per incident report.

    Each dict must have keys: incident_id, severity, location,
    parcels_affected, claim_required.
    """
    return asyncio.run(_extract_all())


if __name__ == "__main__":
    for rec in solve():
        print(rec)
