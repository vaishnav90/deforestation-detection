"""
CONFIGURATION FILE FOR DEFORESTATION DETECTION PROJECT
=====================================================

This file contains all the configuration parameters used throughout the project.
Modify these values to adjust training behavior, data paths, and model settings.

Author: AI Assistant
Date: 2024
"""

import os
import torch

# ========== DATA PATHS ==========
# Root directory containing all satellite data
DATA_ROOT = "Claude AI Archive"

# Paths to satellite imagery patches (16 small patches from full images)
SENTINEL1_PATH = os.path.join(DATA_ROOT, "1_CLOUD_FREE_DATASET", "1_SENTINEL1", "IMAGE_16_GRID")
SENTINEL2_PATH = os.path.join(DATA_ROOT, "1_CLOUD_FREE_DATASET", "2_SENTINEL2", "IMAGE_16_GRID")
MASK_PATH = os.path.join(DATA_ROOT, "3_TRAINING_MASKS", "MASK_16_GRID")

# ========== DATASET CONFIGURATION ==========
NUM_PATCHES = 16        # Number of satellite image patches available
PATCH_SIZE = 2816       # Size of each patch in pixels (2816x2816)
TRAIN_SPLIT = 0.7       # Fraction of data used for training
VAL_SPLIT = 0.3         # Fraction of data used for validation

# ========== TRAINING CONFIGURATION ==========
BATCH_SIZE = 2          # Number of samples per training batch (small due to limited data)
LEARNING_RATE = 0.0005  # Learning rate for optimizer (AdamW)
NUM_EPOCHS = 50         # Maximum number of training epochs

# ========== DEVICE CONFIGURATION ==========
# Automatically detect and use GPU if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== DATA SPECIFICATIONS ==========
SENTINEL1_BANDS = 2     # Sentinel-1 radar bands (VV, VH polarizations)
SENTINEL2_BANDS = 4     # Sentinel-2 optical bands (Red, Green, Blue, NIR)
MASK_BANDS = 1          # Deforestation mask (binary: 0=forest, 1=deforested)
