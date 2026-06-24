# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Unified data loading for MLFP course — supports local and Colab."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Google Drive shared folder containing all MLFP datasets
_DRIVE_FOLDER_ID = "16c3RkGmiwMWbjD7cJKbJx-JRZlgmQdws"

# Module subfolders on the shared Drive
_MODULES = {
    "mlfp01",
    "mlfp02",
    "mlfp03",
    "mlfp04",
    "mlfp05",
    "mlfp06",
    "mlfp_assessment",
}


def _is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _colab_data_root() -> Path:
    """Return the Drive-mounted mlfp_data path in Colab."""
    return Path("/content/drive/MyDrive/mlfp_data")


def _local_cache_dir() -> Path:
    """Return local cache directory for downloaded files."""
    cache = Path.cwd() / ".data_cache"
    cache.mkdir(exist_ok=True)
    return cache


def _download_from_drive(module: str, filename: str, dest: Path) -> Path:
    """Download a file from the shared Google Drive using gdown."""
    import gdown

    dest_dir = dest / module
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename

    if dest_file.exists():
        logger.debug("Using cached file: %s", dest_file)
        return dest_file

    # gdown can download from a folder by file path
    url = f"https://drive.google.com/drive/folders/{_DRIVE_FOLDER_ID}"
    logger.info("Downloading %s/%s from Google Drive...", module, filename)

    # Download the specific file from the shared folder
    try:
        gdown.download_folder(
            url=url,
            output=str(dest),
            quiet=True,
            remaining_ok=True,
        )
    except TypeError:
        # Older gdown versions don't support remaining_ok
        gdown.download_folder(
            url=url,
            output=str(dest),
            quiet=True,
        )

    if not dest_file.exists():
        # Try direct download if folder download didn't isolate the file
        for candidate in dest.rglob(filename):
            if candidate.is_file():
                if candidate != dest_file:
                    candidate.rename(dest_file)
                return dest_file

        msg = (
            f"File not found after download: {module}/{filename}. "
            f"Check that it exists in the mlfp_data shared Drive."
        )
        raise FileNotFoundError(msg)

    return dest_file


def _read_file(path: Path) -> pl.DataFrame:
    """Read a data file into a polars DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    elif suffix == ".parquet":
        return pl.read_parquet(path)
    elif suffix == ".json":
        return pl.read_json(path)
    elif suffix in (".p", ".pickle", ".pkl"):
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if isinstance(obj, pl.DataFrame):
            return obj
        raise TypeError(
            f"Cannot convert pickle object of type {type(obj)} to polars DataFrame. "
            f"Convert the pickle to parquet upstream: pl.from_pandas(obj).write_parquet('out.parquet')"
        )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .csv, .parquet, or .json"
        )


def _repo_data_dir() -> Path | None:
    """Find the repo-local data/ directory by walking up from cwd."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "data"
        if candidate.is_dir() and (parent / "pyproject.toml").exists():
            return candidate
    return None


class MLFPDataLoader:
    """Load MLFP course datasets with automatic source resolution.

    Resolution order:
    1. Colab: Drive mount at /content/drive/MyDrive/mlfp_data/
    2. Local repo data/ directory (committed datasets)
    3. Google Drive download via gdown (cached in .data_cache/)

    Usage:
        loader = MLFPDataLoader()
        df = loader.load("mlfp01", "hdbprices.csv")

    Shortcut:
        df = MLFPDataLoader.mlfp01("hdbprices.csv")
    """

    def __init__(self, cache_dir: Path | str | None = None):
        self._colab = _is_colab()
        if self._colab:
            self._root = _colab_data_root()
        else:
            self._local_data = _repo_data_dir()
            self._cache = Path(cache_dir) if cache_dir else _local_cache_dir()

    def load_raw(self, module: str, filename: str) -> Path:
        """Return the file path without reading into memory.

        Use this for image directories, audio files, or any data that torch/HF
        loads directly rather than via polars.

        Args:
            module: Module subfolder (e.g., "mlfp05")
            filename: File or directory name (e.g., "fashion_mnist", "cifar10")

        Returns:
            Path to the local file or directory.
        """
        if module not in _MODULES:
            raise ValueError(
                f"Unknown module '{module}'. Available: {sorted(_MODULES)}"
            )

        if self._colab:
            path = self._root / module / filename
        else:
            if self._local_data:
                local_path = self._local_data / module / filename
                if local_path.exists():
                    return local_path
            path = self._cache / module / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {module}/{filename}. "
                f"Run 'python scripts/fetch-real-data.py' to download."
            )
        return path

    @staticmethod
    def load_hf(
        dataset_name: str,
        split: str = "train",
        streaming: bool = False,
    ):
        """Load a HuggingFace dataset directly (not via polars).

        Use this for large datasets (millions of rows) or multimodal data
        (images, audio) that don't fit into a DataFrame.

        Args:
            dataset_name: HuggingFace dataset ID (e.g., "zalando-datasets/fashion_mnist")
            split: Dataset split ("train", "test", "validation")
            streaming: If True, returns an IterableDataset for memory-efficient processing

        Returns:
            HuggingFace Dataset or IterableDataset object.
        """
        from datasets import load_dataset

        logger.info(
            "Loading HuggingFace dataset: %s (split=%s, streaming=%s)",
            dataset_name,
            split,
            streaming,
        )
        return load_dataset(dataset_name, split=split, streaming=streaming)

    def load(self, module: str, filename: str) -> pl.DataFrame:
        """Load a dataset file as a polars DataFrame.

        Args:
            module: Module subfolder (e.g., "mlfp01", "mlfp_assessment")
            filename: File name within the module folder (e.g., "hdbprices.csv")

        Returns:
            polars DataFrame with the loaded data.
        """
        if module not in _MODULES:
            raise ValueError(
                f"Unknown module '{module}'. Available: {sorted(_MODULES)}"
            )

        if self._colab:
            path = self._root / module / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}. "
                    f"Ensure mlfp_data is accessible in your Google Drive."
                )
        else:
            # Check repo-local data/ first, then fall back to Drive download
            if self._local_data:
                module_dir = self._local_data / module
                local_path = module_dir / filename
                if local_path.exists():
                    path = local_path
                    logger.info(
                        "Loading %s/%s from local data/ (%s)", module, filename, path
                    )
                    return _read_file(path)
                # Bundled data is present for this module but the file is absent
                # → it is genuinely missing. Raise immediately instead of falling
                # back to a Drive download that would hang offline (and pulls the
                # whole folder online just to discover the file is not there).
                if module_dir.is_dir():
                    raise FileNotFoundError(
                        f"File not found: {module}/{filename} "
                        f"(not in local data/{module}/)."
                    )
            path = _download_from_drive(module, filename, self._cache)

        logger.info("Loading %s/%s (%s)", module, filename, path)
        return _read_file(path)

    def list_files(self, module: str) -> list[str]:
        """List available data files in a module folder."""
        if module not in _MODULES:
            raise ValueError(
                f"Unknown module '{module}'. Available: {sorted(_MODULES)}"
            )

        if self._colab:
            root = self._root / module
        else:
            root = self._cache / module

        if not root.exists():
            return []

        return sorted(f.name for f in root.iterdir() if f.is_file())

    # -- Module shortcuts --

    @classmethod
    def mlfp01(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp01 (Data Pipelines & Visualisation)."""
        return cls().load("mlfp01", filename)

    @classmethod
    def mlfp02(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp02 (Statistical Mastery)."""
        return cls().load("mlfp02", filename)

    @classmethod
    def mlfp03(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp03 (Supervised ML)."""
        return cls().load("mlfp03", filename)

    @classmethod
    def mlfp04(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp04 (Unsupervised ML)."""
        return cls().load("mlfp04", filename)

    @classmethod
    def mlfp05(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp05 (Deep Learning & Vision)."""
        return cls().load("mlfp05", filename)

    @classmethod
    def mlfp06(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp06 (LLMs, Agents & Transformation)."""
        return cls().load("mlfp06", filename)

    @classmethod
    def assessment(cls, filename: str) -> pl.DataFrame:
        """Load from mlfp_assessment (capstone datasets)."""
        return cls().load("mlfp_assessment", filename)
