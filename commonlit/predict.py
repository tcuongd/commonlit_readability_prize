import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LassoCV

from .config import MODEL_OUTPUT, OUTPUT_DIR, PREDICT_OUTPUT
from .features import process_texts, scale_features
from .utils import load_data


def load_test_data() -> pd.DataFrame:
    df = load_data("test")
    return df[["id", "excerpt"]]


def predict(m: LassoCV, df: pd.DataFrame) -> pd.DataFrame:
    features = process_texts(df["excerpt"].tolist())
    preds = m.predict(np.array(scale_features(features)))
    preds_df = pd.DataFrame({"id": df["id"], "target": preds})
    return preds_df


def main():
    df = load_test_data()
    try:
        with open(MODEL_OUTPUT, "rb") as f:
            m = pickle.load(f)
    except FileNotFoundError as exc:
        raise Exception(f"Model binary not found in {MODEL_OUTPUT}") from exc
    logger.info(f"Predicing on test data")
    preds_df = predict(m, df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing submission to {PREDICT_OUTPUT}")
    preds_df.to_csv(PREDICT_OUTPUT.resolve(), index=False)

if __name__ == "__main__":
    main()
