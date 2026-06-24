# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Google Colab setup — paste this as the first cell in Colab notebooks.

This file is the reference source. The actual Colab notebooks embed this
as their first code cell.
"""

COLAB_SETUP_CELL = '''
# ══════════════════════════════════════════════════════════
# MLFP Course Setup (Google Colab)
# ══════════════════════════════════════════════════════════
# Run this cell first. It installs Kailash SDK and mounts
# your Google Drive to access the course datasets.
# ══════════════════════════════════════════════════════════

# Install Kailash SDK
!pip install -q kailash-ml polars plotly gdown python-dotenv

# Mount Google Drive (for dataset access)
from google.colab import drive
drive.mount("/content/drive")

# Verify data access
from pathlib import Path
data_root = Path("/content/drive/MyDrive/mlfp_data")
if data_root.exists():
    print(f"Data folder found: {sorted(p.name for p in data_root.iterdir())}")
else:
    print("WARNING: mlfp_data not found in Drive. Add the shared folder shortcut to My Drive.")

import polars as pl
print(f"polars {pl.__version__} ready")
'''

# Module 5-6 need additional setup for LLM agents
COLAB_AGENT_SETUP_CELL = '''
# Additional setup for AI agent exercises (Modules 5-6)
!pip install -q kailash-kaizen kaizen-agents

# Load API keys from Colab Secrets
from google.colab import userdata
import os

try:
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    os.environ["DEFAULT_LLM_MODEL"] = userdata.get("DEFAULT_LLM_MODEL")
    print("API keys loaded from Colab Secrets")
except Exception:
    print("WARNING: Set OPENAI_API_KEY in Colab Secrets (key icon in sidebar)")
'''

# Module 6 fine-tuning setup
COLAB_ALIGN_SETUP_CELL = '''
# Additional setup for fine-tuning exercises (Module 6)
!pip install -q kailash-align kailash-pact

import os
from google.colab import userdata
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HuggingFace token loaded")
except Exception:
    print("WARNING: Set HF_TOKEN in Colab Secrets for fine-tuning exercises")
'''
