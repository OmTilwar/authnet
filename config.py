"""
AuthNet Configuration
Central configuration for all hyperparameters, paths, and device settings.
"""

import os
import torch

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
BACKBONE = "resnet18"           # Pre-trained backbone
EMBEDDING_DIM = 128             # Output embedding dimension
PRETRAINED = True               # Use ImageNet pre-trained weights
FREEZE_LAYERS = ["layer1", "layer2"]  # Freeze early layers

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 64                 # Per-GPU batch size
NUM_EPOCHS = 40                 # Training epochs
LR_BACKBONE = 1e-4              # Learning rate for backbone (fine-tuning)
LR_HEAD = 1e-3                  # Learning rate for embedding head
WEIGHT_DECAY = 1e-4             # L2 regularization
MARGIN = 0.2                    # Triplet loss margin
MINING_TYPE = "semihard"        # Triplet mining strategy: "semihard", "hard", "all"
SAMPLES_PER_CLASS = 4           # For MPerClassSampler (min images per class in batch)
PATIENCE = 7                    # Early stopping patience (epochs)

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
IMAGE_SIZE = 224                # Input image size (224x224)
NUM_WORKERS = 4                 # DataLoader workers
TRAIN_SPLIT = 0.8               # Train/validation split ratio

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
RECALL_K = [1, 2, 4, 8]        # Recall@K values to compute
AUTH_THRESHOLD = 0.75           # Cosine similarity threshold for authentication

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DTD_DIR = os.path.join(DATA_DIR, "dtd")
MVTEC_DIR = os.path.join(DATA_DIR, "mvtec")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
EMBED_DIR = os.path.join(OUTPUT_DIR, "embeddings")
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmarks")

# ONNX
ONNX_FP32_PATH = os.path.join(MODEL_DIR, "authnet_fp32.onnx")
ONNX_INT8_PATH = os.path.join(MODEL_DIR, "authnet_int8.onnx")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "authnet_best.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "authnet_last.pth")

# MVTec AD categories to use (texture-based only)
MVTEC_CATEGORIES = ["leather", "carpet", "wood", "grid", "tile"]

# ──────────────────────────────────────────────
# Create directories
# ──────────────────────────────────────────────
for _dir in [DATA_DIR, DTD_DIR, MVTEC_DIR, COMBINED_DIR,
             OUTPUT_DIR, MODEL_DIR, VIZ_DIR, EMBED_DIR, BENCHMARK_DIR]:
    os.makedirs(_dir, exist_ok=True)
