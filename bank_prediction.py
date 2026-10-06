"""Run training/evaluation or score a CSV with the saved model."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_preprocessing import load_dataset
from src.eda import create_eda_plots
from src.evaluate_model import create_project_report, evaluate_model
from src.prediction import load_model_bundle, predict_records
from src.train_model import train_and_select

ROOT = Path(__file__).resolve().parent


def run_training(data_path: Path, target: str) -> None:
    frame, labels = load_dataset(data_path, target)
    create_eda_plots(frame.assign(**{target: labels.map({0: "no", 1: "yes"})}), target, ROOT / "outputs")
    model, comparison, X_test, y_test = train_and_select(
        frame, labels, ROOT / "models", ROOT / "outputs"
    )
    report = evaluate_model(model, X_test, y_test, ROOT / "outputs")
    create_project_report(
        comparison, report, ROOT / "report" / "project_report.docx", data_path.name, ROOT / "outputs"
    )
    print(f"Selected model: {comparison.iloc[0]['model']}")
    print(comparison.to_string(index=False))
    print("\nTest classification report:")
    print(pd.DataFrame(report).T.to_string())
    print(f"\nSaved model and plots under {ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "bank-full.csv", help="Training CSV path")
    parser.add_argument("--target", default="y", help="Binary target column (default: y)")
    parser.add_argument("--predict", type=Path, help="Score a CSV using models/final_model.pkl")
    parser.add_argument("--output", type=Path, help="Optional path for prediction results CSV")
    args = parser.parse_args()

    if args.predict:
        bundle = load_model_bundle(ROOT / "models" / "final_model.pkl")
        results = predict_records(pd.read_csv(args.predict), bundle)
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print(results.to_string(index=False))
    else:
        run_training(args.data, args.target)


if __name__ == "__main__":
    main()
