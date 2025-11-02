
# utils/data_loading.py
from __future__ import annotations
from typing import Tuple
import json
import pandas as pd
from .config import MOCK_DATA_DIR
from .preprocess import finalize_dataframe

def load_training_dataframe() -> pd.DataFrame:
    with open(MOCK_DATA_DIR / "training.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    return finalize_dataframe(df)

def load_inference_dataframe() -> pd.DataFrame:
    with open(MOCK_DATA_DIR / "inference.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    return finalize_dataframe(df)
