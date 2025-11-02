# data_utils.py
from pathlib import Path
import pandas as pd

def get_or_make_mock_csv(path: str, columns=('feature1','feature2','target'), rows=64) -> str:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({c: (list(range(rows)) if c != 'target' else [i % 2 for i in range(rows)]) for c in columns})
        df.to_csv(p, index=False)
        print(f"[i] Created mock CSV at {p} with columns={list(columns)}")
    return str(p)
