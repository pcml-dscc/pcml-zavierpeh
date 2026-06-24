# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP Module 6 — LLMs and Agentic Workflows helpers.

Exercise-specific infrastructure (LLM setup, datasets, metrics, classifiers)
that technique files import. Each exercise gets its own submodule:

    from shared.mlfp06.ex_1 import load_sst2, zero_shot_classify, compute_metrics
    ...

Available after `uv sync` from any directory.
"""

# ── Offline fine-tuning: disable wandb ─────────────────────────────────
# trl's SFT/DPO trainers default to auto-detecting wandb and call
# wandb.init() during training. MLFP fine-tuning runs fully local/offline
# (Ollama only, no external SaaS — see rules/independence.md), so an
# unconfigured wandb raises "No API key configured" and aborts training.
# Force wandb into disabled mode before any trl trainer constructs. Set via
# setdefault so an instructor who genuinely wants wandb can export WANDB_MODE
# themselves. WANDB_SILENT quiets the startup banner.
import os as _os

_os.environ.setdefault("WANDB_MODE", "disabled")
_os.environ.setdefault("WANDB_SILENT", "true")

# ── Python 3.13 aiosqlite teardown fix ─────────────────────────────────
# kailash's ExperimentTracker / agent stores run on aiosqlite, whose worker
# Thread (in this version) is created WITHOUT daemon=True. On Python 3.13,
# Py_FinalizeEx joins non-daemon threads BEFORE atexit handlers run, so a
# script that touched the tracker hangs forever at exit on the idle worker
# stuck in queue.get(). Marking the worker daemon lets the interpreter exit
# cleanly once the exercise finishes — the worker is idle at exit (all
# writes already committed), so there is no data loss. Notebooks run inside
# a kernel and never Py_Finalize, so they are unaffected. See
# rules/patterns.md § "Async Resource Cleanup".
try:  # pragma: no cover - environment shim
    import aiosqlite.core as _aiocore

    _OrigThread = _aiocore.Thread
    if not getattr(_OrigThread, "_mlfp_daemonized", False):

        class _DaemonAioThread(_OrigThread):
            _mlfp_daemonized = True

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.daemon = True

        _aiocore.Thread = _DaemonAioThread

    # The aiosqlite Connection.__del__ emits a ResourceWarning via warn();
    # during interpreter shutdown `warn` is already None, so the finalizer
    # raises "TypeError: 'NoneType' object is not callable" and prints a
    # scary traceback AFTER the exercise's clean output. Wrap it to swallow
    # shutdown-time failures (normal-operation behaviour is preserved).
    if not getattr(_aiocore.Connection.__del__, "_mlfp_quiet", False):
        _orig_del = _aiocore.Connection.__del__

        def _quiet_del(self, _orig=_orig_del):
            try:
                _orig(self)
            except Exception:
                pass

        _quiet_del._mlfp_quiet = True
        _aiocore.Connection.__del__ = _quiet_del
except Exception:  # aiosqlite absent or API changed — nothing to patch
    pass


# ── kaizen 2.28 bare-code-fence JSON extraction fix ────────────────────
# kaizen's strategy `parse_result()` strips a leading code fence only when it
# carries a language tag (regex `r"```json\s*"`) and a trailing fence only at
# end-of-string. Small local models (e.g. llama3.2:3b) reliably wrap their
# structured-output JSON in a BARE fence — a leading "```" with no "json"
# tag — which neither regex removes, so `json.loads()` fails and the strategy
# returns {"response": <text>, "error": "JSON_PARSE_FAILED"} with none of the
# typed Signature fields unpacked. Every ex_5 / ex_6 BaseAgent specialist then
# KeyErrors on `result['<field>']`.
#
# This shim wraps `parse_result` on all three strategy classes: when the
# original parse reports JSON_PARSE_FAILED, it retries `json.loads` after a
# robust fence-strip (any leading "```<lang>" + any trailing "```"). On
# success it returns the parsed dict (the same shape the strategy would have
# produced for clean JSON); otherwise it returns the original error result
# unchanged. No behaviour changes for providers that already emit clean JSON.
#
# This is a course-boundary robustness shim, not an SDK reimplementation —
# the upstream fix belongs in kaizen's `parse_result` regex. Tracked for the
# next stack /sync. Mirrors the aiosqlite shim style above.
try:  # pragma: no cover - environment shim
    import json as _json
    import re as _re

    _FENCE_LEAD = _re.compile(r"^```[a-zA-Z0-9_-]*\s*")
    _FENCE_TAIL = _re.compile(r"\s*```$")

    def _robust_json(content: str):
        stripped = _FENCE_TAIL.sub("", _FENCE_LEAD.sub("", content.strip())).strip()
        return _json.loads(stripped)

    def _make_fence_tolerant(cls):
        orig = cls.parse_result
        if getattr(orig, "_mlfp_fence_tolerant", False):
            return

        def parse_result(self, raw_result, _orig=orig):
            result = _orig(self, raw_result)
            if (
                isinstance(result, dict)
                and result.get("error") == "JSON_PARSE_FAILED"
                and isinstance(result.get("response"), str)
            ):
                try:
                    parsed = _robust_json(result["response"])
                except Exception:
                    return result
                if isinstance(parsed, dict):
                    return parsed
                return {"response": parsed, "raw_content": result["response"]}
            return result

        parse_result._mlfp_fence_tolerant = True
        cls.parse_result = parse_result

    from kaizen.strategies.async_single_shot import AsyncSingleShotStrategy
    from kaizen.strategies.multi_cycle import MultiCycleStrategy
    from kaizen.strategies.single_shot import SingleShotStrategy

    for _strategy_cls in (
        AsyncSingleShotStrategy,
        MultiCycleStrategy,
        SingleShotStrategy,
    ):
        _make_fence_tolerant(_strategy_cls)
except Exception:  # kaizen absent or strategy API changed — nothing to patch
    pass


# ── kaizen 2.28 Ollama native-JSON enforcement ─────────────────────────
# kaizen's Ollama provider ignores `response_format`/`structured_output_mode`
# entirely — it never forwards Ollama's native `format="json"` constraint, so
# the only structured-output steering a BaseAgent gets is a weak prompt suffix
# ("Please respond in JSON format."). A small local model (llama3.2:3b)
# routinely ignores that suffix on non-trivial input and returns free prose,
# which no fence-strip can rescue → JSON_PARSE_FAILED → KeyError on the typed
# Signature fields.
#
# Ollama supports grammar-constrained JSON via the `format="json"` kwarg on
# `Client.chat`, which forces well-formed JSON every call. This shim wraps the
# provider's `chat` to (a) pass `format="json"`, and (b) raise the
# `num_predict` (max output tokens) floor so structured replies are not
# truncated mid-object on longer documents. Both only activate for Ollama, and
# only strengthen — never weaken — the existing behaviour.
#
# Course-boundary robustness shim; the upstream fix is for kaizen's Ollama
# provider to honour `response_format`. Tracked for the next stack /sync.
# The provider's `chat()` only forwards `model`, `messages`, and `options` to
# the underlying ``ollama.Client.chat`` — it drops any other kwargs, so a
# `format=` kwarg cannot reach the client from above. We therefore patch
# `_get_client` to return the real client wrapped so that every `.chat()` call
# defaults `format="json"`, and we bump the `num_predict` output floor inside
# the same wrapper.
try:  # pragma: no cover - environment shim
    from kaizen.providers.llm.ollama import OllamaProvider as _OllamaProvider

    _orig_get_client = _OllamaProvider._get_client
    if not getattr(_orig_get_client, "_mlfp_json_enforced", False):

        class _JsonEnforcingClient:
            """Proxy that defaults Ollama's native JSON grammar + output floor."""

            def __init__(self, inner):
                self._inner = inner

            def chat(self, *args, **kwargs):
                kwargs.setdefault("format", "json")
                opts = dict(kwargs.get("options") or {})
                if not opts.get("num_predict") or opts["num_predict"] < 1024:
                    opts["num_predict"] = 1024
                kwargs["options"] = opts
                return self._inner.chat(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def _get_client_json(self, backend_config=None, _orig=_orig_get_client):
            return _JsonEnforcingClient(_orig(self, backend_config))

        _get_client_json._mlfp_json_enforced = True
        _OllamaProvider._get_client = _get_client_json
        # Reset any cached bare client so the proxy is used on next call.
        _OllamaProvider._client = None
except Exception:  # kaizen absent or provider API changed — nothing to patch
    pass
