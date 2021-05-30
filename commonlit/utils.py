from pathlib import Path

import pandas as pd

DATA_DIR = Path(".") / "data"


def load_data(name: str) -> pd.DataFrame:
    fpath = DATA_DIR / f"{name}.csv"
    return pd.read_csv(fpath.resolve())
