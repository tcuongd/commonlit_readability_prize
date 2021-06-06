## Rating the complexity of literary passages for classroom use.

[Kaggle Competition - CommonLit](https://www.kaggle.com/c/commonlitreadabilityprize)

### Setup

System requirements:

* Python 3.9
* [`pipenv`](https://pipenv.pypa.io/en/latest/)
* [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials set up

From this directory:

```sh
pipenv install
pipenv run get_nlp_models
pipenv run get_data
```

Run the full pipeline and produce a sample submission file:

```sh
pipenv run pipeline
```
