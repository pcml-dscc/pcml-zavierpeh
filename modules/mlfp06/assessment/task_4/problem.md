# MLFP06 — Task 4: PACT Governance for a Production Agent Fleet

**Weight**: 30 marks · **Difficulty**: Hard · **Framework**: PACT `GovernanceEngine`
· **Dataset**: canonical SG FinTech org (`shared.mlfp06.ex_7`)

## Scenario

A Singapore digital bank is putting an autonomous agent fleet into production.
Before go-live, MAS TRM and the bank's own risk committee require **structural
governance**: every agent runs inside an _operating envelope_ (a dollar budget

- an allow-listed action set), every access decision is checked by a governance
  engine, and **privilege escalation is impossible by construction** — not merely
  "unlikely at runtime".

Your job is to compile the organisation, attach least-privilege envelopes to
four agent roles, exercise the engine's decision function across allow **and**
deny paths, and prove that a rogue re-delegation is rejected at envelope time.

This task is **100% deterministic** — no LLM calls. The governance engine is a
pure decision function. Implement `solve() -> dict`.

## Step 1 — Compile the organisation

Use `compile_governance()` from `shared.mlfp06.ex_7`. It returns
`(engine, org)`. Report the org counters in `org_stats`:
`n_agents`, `n_delegations`, `n_departments`.

## Step 2 — Attach least-privilege envelopes (build EXACTLY these)

For each role below, build a `ConstraintEnvelopeConfig` (all five canonical
dimensions populated — Financial, Operational, Temporal, Data Access,
Communication) and attach it to the engine with a `RoleEnvelope` using the
addresses in the table. The financial cap and the allowed-action list are the
two dimensions graded.

| Role (address)                 | Delegator (address)          | Clearance  | Max spend (USD) | Allowed actions                                                   |
| ------------------------------ | ---------------------------- | ---------- | --------------- | ----------------------------------------------------------------- |
| `data_analyst` `D1-R1-T1-R1`   | `chief_ml_officer` `D1-R1`   | RESTRICTED | 20.0            | `read_data`, `summarise_data`, `generate_report`                  |
| `model_trainer` `D1-R1-T2-R1`  | `chief_ml_officer` `D1-R1`   | RESTRICTED | 100.0           | `train_model`, `evaluate_model`, `read_data`                      |
| `risk_assessor` `D2-R1-T1-R1`  | `chief_risk_officer` `D2-R1` | RESTRICTED | 200.0           | `read_data`, `audit_model`, `generate_report`, `access_audit_log` |
| `customer_agent` `D3-R1-T1-R1` | `vp_customer` `D3-R1`        | PUBLIC     | 5.0             | `answer_question`, `search_faq`                                   |

## Step 3 — Exercise `engine.verify_action()` across these 10 cases

Call `engine.verify_action(role_address=..., action=..., context={"cost": ...})`
for each case **in this exact order**, and collect `verdict.allowed` (a bool)
into the `verdicts` list:

| #   | Role             | Action             | Cost (USD) |
| --- | ---------------- | ------------------ | ---------- |
| 0   | `data_analyst`   | `read_data`        | 0.10       |
| 1   | `data_analyst`   | `deploy_model`     | 0.10       |
| 2   | `data_analyst`   | `read_data`        | 50.0       |
| 3   | `model_trainer`  | `train_model`      | 5.0        |
| 4   | `model_trainer`  | `deploy_model`     | 1.0        |
| 5   | `risk_assessor`  | `audit_model`      | 0.50       |
| 6   | `risk_assessor`  | `access_audit_log` | 1.0        |
| 7   | `customer_agent` | `search_faq`       | 0.01       |
| 8   | `customer_agent` | `read_data`        | 0.10       |
| 9   | `customer_agent` | `answer_question`  | 100.0      |

## Step 4 — Prove privilege escalation is rejected

Build a department-head **parent** envelope for `vp_customer` (clearance
CONFIDENTIAL, max spend 50.0, allowed actions `answer_question`, `search_faq`).
Then build a **rogue child** envelope that tries to escalate
(clearance RESTRICTED, max spend 1000.0, allowed actions that add `read_data`
and `deploy_model`). Call `RoleEnvelope.validate_tightening(parent_envelope=...,
child_envelope=...)`; it MUST raise `MonotonicTighteningError`. Set
`escalation_caught = True` when (and only when) that error is raised.

## Return contract

```python
def solve() -> dict:
    return {
        "org_stats": {"n_agents": int, "n_delegations": int, "n_departments": int},
        "verdicts": [bool, ...],   # exactly 10, in case order 0..9
        "escalation_caught": bool,
    }
```

## Visible sanity checks

- `org_stats == {"n_agents": 6, "n_delegations": 6, "n_departments": 3}`
- `verdicts == [True, False, False, True, False, True, True, True, False, False]`
- `escalation_caught == True`

## Grading (10 automated checks, all must pass)

return type is dict · `n_agents` correct · `n_delegations` correct ·
`n_departments` correct · 10 verdicts returned · allow-path verdicts correct ·
deny-by-action verdicts correct · deny-by-budget verdicts correct · all 10
verdicts match the reference exactly · privilege escalation caught structurally.

## Rules

- **PACT only** — use `GovernanceEngine.verify_action()` and
  `RoleEnvelope.validate_tightening()`. No hand-rolled `if cost > cap` checks.
- Deterministic — no LLM, no randomness.
- Build all five envelope dimensions explicitly (least privilege is structural).
