# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Lens 2 — Interpretability (the X-Ray).

Question answered: *What does the model attend to, and what circuit produces the answer?*

Works with open-weight models only (Llama, Gemma, Phi, Mistral). API-only
models (GPT, Claude, Gemini) explicitly return ``{"mode": "not_applicable"}``
— the lens is honest about what it cannot do, per rules/zero-tolerance.md
Rule 2 (no fake readings).

Default model: ``google/gemma-2-2b`` (Gemma Scope SAE coverage for
:meth:`sae_features`). Loading happens lazily on first use so the class
is cheap to import.

The underlying libraries are ``transformer_lens`` (attention + activation
hooks), ``sae_lens`` (pre-trained Gemma Scope SAEs), and optional
``nnterp`` (unified 2025 interface). Students read the output, they do
not train the SAEs — per design doc §11 Non-goals.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Sequence

import plotly.graph_objects as go
import polars as pl

from . import _plots

logger = logging.getLogger(__name__)

__all__ = ["InterpretabilityDiagnostics"]

DEFAULT_MODEL = "google/gemma-2-2b"

# Models we know the lens cannot introspect (API-only).
_API_ONLY_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1-",
    "o3-",
    "o4-",
    "claude-",
    "gemini-",
    "deepseek-",  # API tier
)


class InterpretabilityDiagnostics:
    """X-ray lens — attention heatmaps, logit-lens, probes, SAE features.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``google/gemma-2-2b`` (Gemma Scope SAE coverage).
        device: Torch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
            Auto-detected when ``None``.
        dtype: Torch dtype for weights — ``"float16"`` saves VRAM.
        run_id_prefix: Prefix for auto-generated correlation IDs.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "float16",
        run_id_prefix: str = "attn",
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._dtype = dtype
        self._run_id_prefix = run_id_prefix
        self._model: Any = None
        self._attention_log: list[dict[str, Any]] = []
        self._logit_log: list[dict[str, Any]] = []
        self._sae_log: list[dict[str, Any]] = []
        self._probe_log: list[dict[str, Any]] = []
        logger.info(
            "interp_diagnostics.init",
            extra={"model": model_name, "device": device or "auto", "dtype": dtype},
        )

    def __enter__(self) -> "InterpretabilityDiagnostics":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the HookedTransformer + clear caches."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Applicability check ────────────────────────────────────────────

    def _is_api_only(self, model: str | None = None) -> bool:
        m = (model or self.model_name).lower()
        return any(m.startswith(p) for p in _API_ONLY_PREFIXES)

    def _not_applicable(self, method: str, model: str | None = None) -> dict[str, Any]:
        logger.info(
            "interp.not_applicable",
            extra={"method": method, "model": model or self.model_name, "mode": "real"},
        )
        return {
            "mode": "not_applicable",
            "method": method,
            "model": model or self.model_name,
            "reason": (
                "attention lens requires open-weight models (Llama, Gemma, Phi, "
                "Mistral). API-only models (GPT, Claude, Gemini) cannot be X-rayed."
            ),
        }

    # ── Model loading ─────────────────────────────────────────────────

    def _load_model(self) -> Any:
        """Lazy-load the HookedTransformer. First call only."""
        if self._model is not None:
            return self._model
        if self._is_api_only():
            raise RuntimeError(
                f"{self.model_name} is API-only; cannot load weights. Use an "
                "open-weight model such as google/gemma-2-2b or meta-llama/Llama-3.2-1B."
            )
        try:
            from transformer_lens import HookedTransformer
        except ImportError as exc:  # pragma: no cover — optional extra
            raise ImportError(
                "InterpretabilityDiagnostics requires transformer_lens. "
                "Install with `pip install transformer-lens`."
            ) from exc

        device = self._device or _auto_device()
        # HuggingFace token for gated Gemma/Llama weights.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        load_kwargs: dict[str, Any] = {"device": device}
        if self._dtype == "float16":
            import torch

            load_kwargs["torch_dtype"] = torch.float16
        if token:
            load_kwargs["token"] = token

        logger.info(
            "interp.load_model.start",
            extra={"model": self.model_name, "device": device, "dtype": self._dtype},
        )
        self._model = HookedTransformer.from_pretrained(self.model_name, **load_kwargs)
        logger.info(
            "interp.load_model.ok",
            extra={"model": self.model_name, "n_layers": self._model.cfg.n_layers},
        )
        return self._model

    # ── Attention heatmap ──────────────────────────────────────────────

    def attention_heatmap(
        self,
        prompt: str,
        *,
        layer: int = 0,
        head: int = 0,
        run_id: str | None = None,
    ) -> go.Figure:
        """Token-to-token attention weights at (layer, head) as a heatmap.

        Returns a Plotly Figure; also records the (tokens, matrix) reading
        for later dashboard aggregation.
        """
        if self._is_api_only():
            # For API-only models the plot is a labelled empty figure.
            _ = self._not_applicable("attention_heatmap")
            return _plots.empty_figure(
                f"Attention Heatmap — layer {layer}, head {head}",
                note="not applicable for API-only models",
            )

        run_id = run_id or f"{self._run_id_prefix}-{uuid.uuid4().hex[:12]}"
        model = self._load_model()
        import torch

        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        # transformer_lens stores attention as [batch, head, query, key]
        attn = cache["pattern", layer][0, head].to("cpu").float().numpy()
        labels = [model.to_string(t) for t in tokens[0]]
        labels = [lbl.replace("\n", "\\n") or "∅" for lbl in labels]

        fig = go.Figure(
            go.Heatmap(
                z=attn,
                x=labels,
                y=labels,
                colorscale="Viridis",
                colorbar=dict(title="attention"),
            )
        )
        fig.update_layout(
            title=f"Attention — {self.model_name} · L{layer} H{head}",
            xaxis_title="key token",
            yaxis_title="query token",
            template=_plots.TEMPLATE,
            height=600,
        )
        self._attention_log.append(
            {
                "run_id": run_id,
                "layer": layer,
                "head": head,
                "n_tokens": len(labels),
                "mode": "real",
            }
        )
        logger.info(
            "interp.attention_heatmap.ok",
            extra={
                "run_id": run_id,
                "layer": layer,
                "head": head,
                "n_tokens": len(labels),
                "mode": "real",
            },
        )
        return fig

    # ── Logit lens ─────────────────────────────────────────────────────

    def logit_lens(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Early-exit predictions per layer.

        Projects each layer's residual stream through the unembedding and
        records the top-``k`` tokens + probabilities. Returns a Polars
        DataFrame with columns ``layer``, ``rank``, ``token``, ``prob``.

        On API-only models, returns an empty DataFrame tagged
        ``mode="not_applicable"`` instead of raising.
        """
        if self._is_api_only():
            reading = self._not_applicable("logit_lens")
            self._logit_log.append(reading)
            return pl.DataFrame(
                schema={
                    "layer": pl.Int64,
                    "rank": pl.Int64,
                    "token": pl.Utf8,
                    "prob": pl.Float64,
                    "mode": pl.Utf8,
                }
            )

        run_id = run_id or f"{self._run_id_prefix}-ll-{uuid.uuid4().hex[:12]}"
        model = self._load_model()
        import torch

        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        rows: list[dict[str, Any]] = []
        last_tok = tokens.shape[-1] - 1
        for layer in range(model.cfg.n_layers):
            resid = cache["resid_post", layer][last_tok]
            normed = model.ln_final(resid)
            logits = model.unembed(normed)
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=top_k)
            for rank, (p, tok_id) in enumerate(
                zip(top.values.tolist(), top.indices.tolist())
            ):
                rows.append(
                    {
                        "layer": layer,
                        "rank": rank,
                        "token": model.to_string(tok_id).replace("\n", "\\n"),
                        "prob": float(p),
                        "mode": "real",
                    }
                )
        df = pl.DataFrame(rows)
        self._logit_log.append(
            {
                "run_id": run_id,
                "n_layers": model.cfg.n_layers,
                "top_k": top_k,
                "mode": "real",
            }
        )
        logger.info(
            "interp.logit_lens.ok",
            extra={
                "run_id": run_id,
                "n_layers": model.cfg.n_layers,
                "top_k": top_k,
                "mode": "real",
            },
        )
        return df

    def plot_logit_lens(self, df: pl.DataFrame) -> go.Figure:
        """Heatmap of top-1 logit-lens probability per layer."""
        if df.height == 0:
            return _plots.empty_figure("Logit Lens", note="no data or not applicable")
        top1 = df.filter(pl.col("rank") == 0).sort("layer")
        fig = go.Figure(
            go.Bar(
                x=top1["layer"].to_list(),
                y=top1["prob"].to_list(),
                text=top1["token"].to_list(),
                marker_color=_plots.PRIMARY,
            )
        )
        _plots.style(
            fig,
            f"Logit Lens — top-1 probability per layer ({self.model_name})",
            x="layer",
            y="probability",
        )
        return fig

    # ── Linear probe ───────────────────────────────────────────────────

    def probe(
        self,
        prompts: Sequence[str],
        labels: Sequence[int],
        *,
        layer: int,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Train a linear probe on layer activations.

        The caller supplies ``prompts`` and corresponding integer ``labels``.
        The method extracts the last-token residual stream at ``layer``,
        fits a logistic regression (scikit-learn), and reports CV accuracy.
        """
        if self._is_api_only():
            return self._not_applicable("probe")

        if len(prompts) != len(labels):
            raise ValueError("prompts and labels must be same length")
        if len(set(labels)) < 2:
            raise ValueError("probe needs at least 2 distinct labels")

        run_id = run_id or f"{self._run_id_prefix}-probe-{uuid.uuid4().hex[:12]}"
        model = self._load_model()
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        features: list[np.ndarray] = []
        for p in prompts:
            tokens = model.to_tokens(p)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            last = cache["resid_post", layer][-1].to("cpu").float().numpy()
            features.append(last)
        X = np.stack(features)
        y = np.asarray(labels)
        clf = LogisticRegression(max_iter=500)
        scores = cross_val_score(clf, X, y, cv=min(5, len(set(labels))))
        acc = float(scores.mean())
        row = {
            "run_id": run_id,
            "layer": layer,
            "n_prompts": len(prompts),
            "n_classes": len(set(labels)),
            "cv_accuracy": acc,
            "mode": "real",
        }
        self._probe_log.append(row)
        logger.info("interp.probe.ok", extra=row)
        return row

    # ── SAE features (Gemma Scope) ─────────────────────────────────────

    def sae_features(
        self,
        prompt: str,
        *,
        layer: int,
        top_k: int = 10,
        release: str | None = None,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Load a pre-trained SAE and return the top-``k`` active features.

        ``release`` is the :mod:`sae_lens` release identifier. When ``None``,
        the default Gemma Scope release matching ``self.model_name`` is used
        (``gemma-scope-2b-pt-res`` for gemma-2-2b).

        Returns a Polars DataFrame with ``feature_index``, ``activation``,
        and a ``mode`` column. Students do NOT train SAEs (design §11).
        """
        if self._is_api_only():
            reading = self._not_applicable("sae_features")
            self._sae_log.append(reading)
            return pl.DataFrame(
                schema={
                    "feature_index": pl.Int64,
                    "activation": pl.Float64,
                    "mode": pl.Utf8,
                }
            )

        try:
            from sae_lens import SAE
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sae_features requires sae-lens. Install with `pip install sae-lens`."
            ) from exc

        run_id = run_id or f"{self._run_id_prefix}-sae-{uuid.uuid4().hex[:12]}"
        model = self._load_model()
        import torch

        release = release or _default_sae_release(self.model_name)
        sae_id = f"layer_{layer}/width_16k/canonical"
        logger.info(
            "interp.sae_load.start",
            extra={"release": release, "sae_id": sae_id, "layer": layer},
        )
        sae, _cfg_dict, _sparsity = SAE.from_pretrained(release=release, sae_id=sae_id)
        sae = sae.to(model.cfg.device)

        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            hook_name = f"blocks.{layer}.hook_resid_post"
            resid = cache[hook_name][-1]
            acts = sae.encode(resid.to(sae.device))
        top = torch.topk(acts, k=min(top_k, acts.numel()))
        rows = [
            {
                "feature_index": int(idx),
                "activation": float(val),
                "mode": "real",
            }
            for idx, val in zip(top.indices.tolist(), top.values.tolist())
        ]
        df = pl.DataFrame(rows)
        self._sae_log.append(
            {
                "run_id": run_id,
                "layer": layer,
                "release": release,
                "top_k": top_k,
                "mode": "real",
            }
        )
        logger.info(
            "interp.sae_features.ok",
            extra={
                "run_id": run_id,
                "layer": layer,
                "top_k": top_k,
                "mode": "real",
            },
        )
        return df

    # ── Report ─────────────────────────────────────────────────────────

    def report(self) -> str:
        if self._is_api_only():
            return (
                f"interp-lens: {self.model_name} is API-only — attention, logit "
                "lens, probe, and SAE features are NOT APPLICABLE. Load an "
                "open-weight model (e.g. google/gemma-2-2b)."
            )
        parts: list[str] = []
        if self._attention_log:
            parts.append(f"attention: {len(self._attention_log)} heatmaps recorded")
        if self._logit_log:
            parts.append(f"logit_lens: {len(self._logit_log)} sweeps recorded")
        if self._probe_log:
            last = self._probe_log[-1]
            parts.append(
                f"probe: last CV accuracy={last['cv_accuracy']:.2f} on layer {last['layer']}"
            )
        if self._sae_log:
            parts.append(f"sae: {len(self._sae_log)} feature reads recorded")
        if not parts:
            return "interp-lens: no readings recorded yet."
        return "interp-lens:\n  " + "\n  ".join(parts)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _auto_device() -> str:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _default_sae_release(model_name: str) -> str:
    name = model_name.lower()
    if "gemma-2-2b" in name:
        return "gemma-scope-2b-pt-res"
    if "gemma-2-9b" in name:
        return "gemma-scope-9b-pt-res"
    # Fall back to gemma-2-2b's release — Gemma Scope has the best coverage.
    return "gemma-scope-2b-pt-res"
