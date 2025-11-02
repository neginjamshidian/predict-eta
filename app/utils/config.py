
# utils/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root (predict-eta/)
MOCK_DATA_DIR = BASE_DIR / "mock_data"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"
