import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Radar parameters
NUM_BEAMS = 64
MAX_ANGLE_DEG = 90
D_BY_LAMBDA = 0.5

# Gesture formatting
GESTURES_PER_BATCH = 20
EPSILON = 1e-10  # For log10 stability

# ML Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0005  # Change from 0.001 to 0.0005
EPOCHS = 30             # Give it slightly more time to train since LR is lower
TEST_SPLIT = 0.2
RANDOM_STATE = 42