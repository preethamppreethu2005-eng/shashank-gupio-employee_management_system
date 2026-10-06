"""Load a saved model and score new customer records."""
from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd


def load_model_bundle(path: str | Path) -> dict[str, object]:
    """Load a trusted model bundle created by this project."""
    with Path(path).open("rb") as file:
        bundle = pickle.load(file)  # Only load pickle files from trusted sources.
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("The model file does not contain a recognized model bundle.")
    return bundle


def predict_records(
    records: pd.DataFrame, bundle: dict[str, object]
) -> pd.DataFrame:
    """Return predicted class and subscription probability for each row."""
    columns = bundle["feature_columns"]
    if not isinstance(columns, list):
        raise ValueError("The saved model bundle has no valid feature column list.")
    missing = set(columns).difference(records.columns)
    if missing:
        raise ValueError(f"Input is missing required features: {sorted(missing)}")
    model = bundle["model"]
    result = records.loc[:, columns].copy()
    result["predicted_subscription"] = model.predict(result).astype(int)
    result["subscription_probability"] = model.predict_proba(result)[:, 1]
    return result
