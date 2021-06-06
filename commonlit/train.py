import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import COEFS_OUTPUT, FEATURES_OUTPUT, MODEL_OUTPUT, OUTPUT_DIR
from .features import process_texts
from .utils import load_data


def load_train_data() -> pd.DataFrame:
    df = load_data("train")
    return df[["id", "excerpt", "target", "standard_error"]]


def train(features_df: pd.DataFrame, target: np.array) -> LassoCV:
    train_array = np.array(features_df)
    X_train, X_test, y_train, y_test = train_test_split(train_array, target, test_size=0.1)
    m = LassoCV(
        n_alphas=100, fit_intercept=True, normalize=False, max_iter=5000, tol=0.0000001, cv=10
    )
    m = m.fit(X_train, y_train)
    oos_preds = m.predict(X_test)
    oos_rmse = mean_squared_error(y_test, oos_preds, squared=False)
    logger.info(f"Out-of-sample RMSE: {oos_rmse:.2f}")
    return m


def model_coefficients(m: LassoCV, features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"feature": features, "coef": m.coef_})
    df["coef_abs"] = df["coef"].abs()
    return df.sort_values(["coef_abs"], ascending=False)


def main():
    df = load_train_data()
    logger.info("Processing texts")
    features = process_texts(df["excerpt"].tolist())
    logger.info("Training model")
    m = train(features, df["target"].values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dumping model to {MODEL_OUTPUT}")
    with open(MODEL_OUTPUT.resolve(), "wb") as f:
        pickle.dump(m, f, protocol=3)

    logger.info(f"Dumping model coefficients to {COEFS_OUTPUT}")
    coefs = model_coefficients(m, features.columns)
    coefs.to_csv(COEFS_OUTPUT.resolve(), index=False)

    logger.info(f"Dumping features table to {FEATURES_OUTPUT}")
    max_rows = 100000
    features.iloc[:max_rows].to_csv(FEATURES_OUTPUT.resolve(), index=False)


if __name__ == "__main__":
    main()
