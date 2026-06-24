# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP03 — Exercise 5: Class Imbalance & Calibration (R10 directory).

Five independently runnable technique files. Each file follows the 5-phase
R10 structure: Theory -> Build -> Train -> Visualise -> Apply.

Run order (later files depend on probability vectors saved by earlier ones):

    01_metrics_and_baseline.py      # baseline + complete metrics taxonomy
    02_sampling_strategies.py       # SMOTE vs cost-sensitive learning
    03_loss_functions.py            # focal loss + alpha weighting
    04_threshold_optimisation.py    # cost matrix + annual ROI
    05_calibration.py               # Platt + Isotonic + final comparison
"""
