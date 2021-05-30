import numpy as np
import pandas as pd
import spacy
from loguru import logger
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

from .utils import load_data

nlp = spacy.load("en_core_web_sm")


def load_train_data() -> pd.DataFrame:
    df = load_data("train")
    return df[["id", "excerpt", "target", "standard_error"]]


def mean_vectors(excerpt: str) -> np.array:
    doc = nlp(excerpt)
    all_vectors = np.vstack([token.vector for token in doc])
    return all_vectors.mean(axis=0)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    word_vecs = df["excerpt"].apply(mean_vectors)
    word_vecs_array = np.array(word_vecs.tolist())
    cols = [f"vec{i}" for i in range(word_vecs_array.shape[1])]
    df[cols] = word_vecs_array
    return df


def train(df: pd.DataFrame, target: str, exclude_features: list[str]) -> LassoCV:
    train_array = np.array(df[[f for f in df.columns if f not in exclude_features + [target]]])
    X_train, X_test, y_train, y_test = train_test_split(train_array, df[target], test_size=0.1)
    m = LassoCV(n_alphas=100, fit_intercept=True, normalize=False, max_iter=5000, tol=0.0000001)
    m = m.fit(X_train, y_train)
    oos_rmse = m.score(X_test, y_test)
    logger.info(f"Out-of-sample RMSE: {oos_rmse:.2f}")
    return m
