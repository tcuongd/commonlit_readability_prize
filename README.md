## Rating the complexity of literary passages for classroom use.

[Kaggle Competition - CommonLit](https://www.kaggle.com/c/commonlitreadabilityprize)

### Setup

System requirements:

* Python 3.9
* `pipenv`
* Kaggle API credentials set up (see https://github.com/Kaggle/kaggle-api)

From this directory:

```sh
pipenv install
pipenv run kaggle competitions download -c commonlitreadabilityprize
unzip commonlitreadabilityprize.zip -d data/
```

Run the full pipeline and product a sample submission file:

```sh
pipenv run pipeline
```