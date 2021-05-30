from pathlib import Path

import pandas as pd
import spacy
from loguru import logger
from sklearn.linear_model import LassoCV

from .train import preprocess
from .utils import load_data

nlp = spacy.load("en_core_web_sm")

OUTPUT_DIR = Path(".") / "output"

def load_test_data() -> pd.DataFrame:
    df = load_data("test")
    return df[["id", "excerpt"]]


def predict(m: LassoCV, df: pd.DataFrame, exclude_features: list[str]) -> pd.DataFrame:
    X_test = preprocess(df)[[f for f in df.columns if f not in exclude_features]]
    preds = m.predict(X_test)
    preds_df = pd.DataFrame({"id": df["id"], "target": preds})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    foutput = OUTPUT_DIR / "submission.csv"
    preds_df.to_csv(foutput.resolve(), index=False)
    logger.info(f"Submission written to {foutput}")
