# Bank Term Deposit Prediction

A reproducible machine-learning project that estimates whether a bank customer will subscribe to a term deposit. It compares logistic regression and random forest baselines, reports held-out performance, and saves plots, a reusable model, and a Word report.

## Project layout

- `bank-full.csv` — expected input dataset (not bundled; see below).
- `bank_prediction.py` — command-line training and batch-prediction entry point.
- `src/data_preprocessing.py` — CSV loading, target encoding, and pipeline preprocessors.
- `src/eda.py` — exploratory visualizations (target distribution, age, job subscription rates).
- `src/train_model.py` — train, tune, compare classifiers, and save the winning pipeline.
- `src/evaluate_model.py` — model evaluation, confusion matrix, feature importance, and Word report generation.
- `src/prediction.py` — load model artifact and run batch/single customer inference.
- `notebooks/bank_term_deposit_prediction.ipynb` — interactive project walkthrough.
- `models/final_model.pkl` — trained model pipeline bundle.
- `outputs/` — EDA charts, evaluation figures, and model comparison table.
- `report/project_report.docx` — comprehensive project report with metrics and figures.

## Dataset

Place the UCI Bank Marketing `bank-full.csv` file in this project directory. The original file is semicolon-delimited and uses `y` as the target; delimiter detection is enabled.

## Setup

Use Python 3.10 or newer. From this directory, install the packages listed in `requirements.txt` in your preferred virtual environment.

## Train and evaluate

Run `python bank_prediction.py`. To choose another CSV or target column, provide `--data` and/or `--target`. The script uses a stratified 80/20 split, compares logistic regression and random forest by ROC-AUC, and creates all model, visualization, and report artifacts. Set a fixed random state for repeatable results.

## Predict new records

After training, provide a CSV containing the same feature columns used during training (do not include the target column). The command-line entry point writes predictions to the console or to the path passed to `--output`. Predictions include a binary class and estimated subscription probability.

## Important modeling note

The source dataset's `duration` field is the length of the last contact. It is not available before a call and can be highly predictive because it is measured during the campaign interaction. For a genuine pre-call deployment, remove `duration` from the feature table during both training and inference; the default project follows the full provided feature set for reproducibility.
