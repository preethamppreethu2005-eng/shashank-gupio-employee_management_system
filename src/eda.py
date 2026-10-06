"""Generate the project's exploratory-analysis figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")


def create_eda_plots(frame: pd.DataFrame, target: str, output_dir: str | Path) -> None:
    """Write target, age, and job subscription plots to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = frame[target].astype(str).str.lower()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=labels, order=["no", "yes"], ax=ax)
    ax.set(title="Term-deposit subscription", xlabel="Subscribed", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)

    if "age" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_data = frame.assign(_subscription=labels)
        sns.boxplot(data=plot_data, x="_subscription", y="age", order=["no", "yes"], ax=ax)
        ax.set(title="Age by subscription outcome", xlabel="Subscribed", ylabel="Age")
        fig.tight_layout()
        fig.savefig(output_dir / "age_vs_subscription.png", dpi=160)
        plt.close(fig)

    if "job" in frame.columns:
        rates = frame.assign(_subscription=labels).groupby("job", observed=False)[
            "_subscription"
        ].apply(lambda values: values.eq("yes").mean()).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        rates.mul(100).plot.barh(ax=ax, color=sns.color_palette("Set2", len(rates)))
        ax.set(title="Subscription rate by job", xlabel="Subscribed customers (%)", ylabel="Job")
        fig.tight_layout()
        fig.savefig(output_dir / "job_vs_subscription.png", dpi=160)
        plt.close(fig)
