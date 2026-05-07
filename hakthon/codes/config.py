"""
Configuration file for Tomato Disease Detection System
Contains all hyperparameters, paths, and settings for training and inference
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directories
BASE_DIR = Path("/home/smart-farming")
DATA_DIR = BASE_DIR / "tomato_extract/tomato_disease_knowledge.json-20260204T122943Z-3-001/PlantDiseased-20260204T122941Z-3-002/PlantDiseased"
KNOWLEDGE_FILE = BASE_DIR / "tomato_extract/tomato_disease_knowledge.json-20260204T122943Z-3-001/tomato_disease_knowledge.json/disease_update.json"

# Dataset paths
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# Output directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, HEATMAPS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Class names (10 tomato disease classes)
CLASS_NAMES = [
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites",
    "tomato_target_spot",
    "tomato_yellow_leaf_curl_virus"
]

NUM_CLASSES = len(CLASS_NAMES)

# Image settings
IMAGE_SIZE = 224  # Standard ViT input size
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Vision Transformer settings
MODEL_NAME = "vit_large_patch16_224"  # upgraded for better accuracy
# Alternative: "deit_base_patch16_224", "vit_large_patch16_224"

PRETRAINED = True
DROPOUT_RATE = 0.2

# Progressive fine-tuning settings
NUM_TOP_BLOCKS_STAGE2 = 6  # Number of top transformer blocks to unfreeze in stage 2

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Hardware
DEVICE = "cuda"  # Will auto-detect in code
NUM_WORKERS = 4
PIN_MEMORY = True

# Batch sizes
BATCH_SIZE_STAGE1 = 32
BATCH_SIZE_STAGE2 = 32
BATCH_SIZE_STAGE3 = 24

# Stage 1: Classifier head training (backbone frozen)
STAGE1_EPOCHS = 15
STAGE1_LR = 2e-3
STAGE1_WEIGHT_DECAY = 0.01
STAGE1_WARMUP_STEPS = 500

# Stage 2: Top blocks fine-tuning
STAGE2_EPOCHS = 20
STAGE2_LR = 5e-5
STAGE2_WEIGHT_DECAY = 0.01
STAGE2_WARMUP_STEPS = 300

# Stage 3: Full model fine-tuning
STAGE3_EPOCHS = 25
STAGE3_LR = 5e-6
STAGE3_WEIGHT_DECAY = 0.02
STAGE3_WARMUP_STEPS = 200

# Optimizer settings
OPTIMIZER = "adamw"
BETAS = (0.9, 0.999)
EPS = 1e-8

# Learning rate scheduler
SCHEDULER = "cosine"  # Options: "cosine", "step", "plateau"
MIN_LR = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 5e-5

# Gradient clipping
GRAD_CLIP_MAX_NORM = 0.5

# Mixed precision training
USE_AMP = True  # Automatic Mixed Precision

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

# Training augmentation probabilities (farm-condition simulation)
AUG_RANDOM_BRIGHTNESS_CONTRAST_P = 0.8
AUG_HUE_SATURATION_VALUE_P = 0.6
AUG_RANDOM_SHADOW_P = 0.4
AUG_MOTION_BLUR_P = 0.35
AUG_GAUSS_NOISE_P = 0.35
AUG_PERSPECTIVE_P = 0.35
AUG_COARSE_DROPOUT_P = 0.5
AUG_HORIZONTAL_FLIP_P = 0.5
AUG_VERTICAL_FLIP_P = 0.3
AUG_ROTATE90_P = 0.3
AUG_SHIFT_SCALE_ROTATE_P = 0.5
AUG_BLUR_P = 0.25
AUG_JPEG_COMPRESSION_P = 0.35

# Augmentation parameters
BRIGHTNESS_CONTRAST_LIMIT = 0.3
HUE_SHIFT_LIMIT = 25
SAT_SHIFT_LIMIT = 40
VAL_SHIFT_LIMIT = 30
SHADOW_NUM_SHADOWS_LOWER = 1
SHADOW_NUM_SHADOWS_UPPER = 2
MOTION_BLUR_LIMIT = 7
GAUSS_NOISE_VAR_LIMIT = (10.0, 50.0)
PERSPECTIVE_SCALE = 0.05
COARSE_DROPOUT_MAX_HOLES = 8
COARSE_DROPOUT_MAX_HEIGHT = 32
COARSE_DROPOUT_MAX_WIDTH = 32
ROTATE_LIMIT = 30
SHIFT_LIMIT = 0.1
SCALE_LIMIT = 0.2
JPEG_QUALITY_LOWER = 70

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

USE_CLASS_WEIGHTS = True
USE_BALANCED_SAMPLING = False  # Set to True if class weights aren't enough

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Tomato detection threshold
TOMATO_DETECTION_THRESHOLD = 0.5

# Confidence calibration
USE_TEMPERATURE_SCALING = True
TEMPERATURE = 1.0  # Will be calibrated on validation set

# Prediction threshold
MIN_CONFIDENCE_THRESHOLD = 0.3  # Below this, report "uncertain"

# ============================================================================
# EXPLAINABILITY CONFIGURATION
# ============================================================================

# Attention visualization
ATTENTION_ROLLOUT_DISCARD_RATIO = 0.9
ATTENTION_HEAD_FUSION = "mean"  # Options: "mean", "max", "min"

# Heatmap settings
HEATMAP_COLORMAP = "jet"  # OpenCV colormap
HEATMAP_ALPHA = 0.5  # Overlay transparency

# ============================================================================
# WEB INTERFACE CONFIGURATION
# ============================================================================

# Server settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = False

# Upload settings
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_INTERVAL = 1  # Save every N epochs
VERBOSE = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# TOMATO VS NON-TOMATO CLASSIFIER
# ============================================================================

# Binary classifier settings
BINARY_MODEL_NAME = "vit_base_patch16_224"
BINARY_EPOCHS = 10
BINARY_LR = 1e-4
BINARY_BATCH_SIZE = 32

# Negative samples (non-tomato images)
# These will be downloaded from ImageNet or other sources
NUM_NEGATIVE_SAMPLES = 5000
