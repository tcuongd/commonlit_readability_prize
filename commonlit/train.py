from multiprocessing import Pipe
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import COEFS_OUTPUT, FEATURES_OUTPUT, MODEL_OUTPUT, OUTPUT_DIR
from .features import process_texts
from .utils import load_data


def load_train_data() -> pd.DataFrame:
    df = load_data("train")
    return df[["id", "excerpt", "target", "standard_error"]]


def train(
    features_df: pd.DataFrame,
    target: np.array,
    standard_error: np.array,
    samples_per_record: int = 20,
) -> Pipeline:
    train_array = np.array(features_df)
    X_train, X_test, y_train, y_test, std_train, std_test = train_test_split(
        train_array, target, standard_error, test_size=0.1
    )

    X_train_resampled = np.repeat(X_train, repeats=samples_per_record, axis=0)
    y_train_resampled = np.hstack(
        [
            np.random.normal(loc=t, scale=s, size=samples_per_record)
            for t, s in zip(y_train, std_train)
        ]
    )

    lasso = LassoCV(
        n_alphas=100,
        fit_intercept=True,
        normalize=False,
        max_iter=10000,
        tol=0.0000001,
        cv=10,
        n_jobs=5,
    )
    pipe = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
    pipe.fit(X_train_resampled, y_train_resampled)

    oos_preds = pipe.predict(X_test)
    oos_rmse = mean_squared_error(y_test, oos_preds, squared=False)
    logger.info(f"Out-of-sample RMSE: {oos_rmse:.2f}")
    return pipe


def model_coefficients(m: LassoCV, features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"feature": features, "coef": m.coef_})
    df["coef_abs"] = df["coef"].abs()
    return df.sort_values(["coef_abs"], ascending=False)


def main():
    df = load_train_data()
    logger.info("Processing texts")
    features = process_texts(df["excerpt"].tolist())
    logger.info("Training model")
    pipe = train(features, df["target"].values, df["standard_error"].values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dumping model to {MODEL_OUTPUT}")
    with open(MODEL_OUTPUT.resolve(), "wb") as f:
        pickle.dump(pipe, f, protocol=3)

    logger.info(f"Dumping model coefficients to {COEFS_OUTPUT}")
    coefs = model_coefficients(pipe.named_steps['lasso'], features.columns)
    coefs.to_csv(COEFS_OUTPUT.resolve(), index=False)

    logger.info(f"Dumping features table to {FEATURES_OUTPUT}")
    max_rows = 100000
    features.iloc[:max_rows].to_csv(FEATURES_OUTPUT.resolve(), index=False)


if __name__ == "__main__":
    main()
