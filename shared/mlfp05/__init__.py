# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP Module 5 — Deep Learning helpers.

Exercise-specific infrastructure (data loading, visualisation, training
loops) that technique files import. Each exercise gets its own submodule:

    from shared.mlfp05.ex_1 import load_fashion_mnist, train_variant
    from shared.mlfp05.ex_2 import load_cifar10, train_cnn
    ...

Available after `uv sync` from any directory.
"""

# ── Python 3.13 aiosqlite teardown fix (CLI-script counterpart of M4's
# teardown_engines) ────────────────────────────────────────────────────
# kailash's ExperimentTracker runs on aiosqlite, whose worker Thread (in
# this version) is created WITHOUT daemon=True. On Python 3.13,
# Py_FinalizeEx joins non-daemon threads BEFORE atexit handlers run, so a
# script that used the tracker hangs forever at exit on the idle worker
# stuck in queue.get() (observed: ex_1 autoencoders ran in ~30s then hung
# until the 600s watchdog killed them). Marking the worker daemon lets the
# interpreter exit cleanly once the exercise finishes — the worker is idle
# at exit (every `async with tracker.track(...)` block has already
# committed), so there is no data loss. Notebooks run inside a kernel and
# never Py_Finalize, so they are unaffected. See rules/patterns.md
# § "Async Resource Cleanup" and the M4 `teardown_engines()` helper.
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
