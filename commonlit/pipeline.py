from .predict import load_test_data, predict
from .train import load_train_data, preprocess, train


def main():
    train_df = load_train_data()
    model = train(
        preprocess(train_df), "target", exclude_features=["id", "excerpt", "standard_error"]
    )
    test_df = load_test_data()
    predict(model, test_df, exclude_features=["id", "excerpt"])


if __name__ == "__main__":
    main()
