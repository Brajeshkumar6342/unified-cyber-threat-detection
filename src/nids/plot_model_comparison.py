
"""Plot validation metrics for Random Forest configurations."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "random_forest_configuration_comparison.csv"
)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Comparison results not found: {RESULTS_PATH}\n"
            "Run src/nids/compare_random_forest.py first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "experiment",
        "f1_attack",
        "recall_attack",
        "false_positive_rate",
    }
    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Convert model names into readable labels.
    labels = results["experiment"].str.replace("RF_", "RF ", regex=False)

    metrics = [
        ("f1_attack", "Attack F1-score", "attack_f1_comparison.png"),
        ("recall_attack", "Attack Recall", "attack_recall_comparison.png"),
        (
            "false_positive_rate",
            "False-Positive Rate",
            "false_positive_rate_comparison.png",
        ),
    ]

    for column, title, filename in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))

        bars = ax.bar(labels, results[column])
        ax.set_title(f"{title} — Validation Results")
        ax.set_ylabel("Score")
        ax.set_ylim(0, min(1.0, max(results[column].max() * 1.15, 0.1)))
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, value in zip(bars, results[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        output_path = FIGURES_DIR / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")

    print("\nVisualization complete.")
    print("These figures show validation results, not official test results.")


if __name__ == "__main__":
    main()
