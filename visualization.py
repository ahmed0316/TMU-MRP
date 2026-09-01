import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class ModelVisualizer:

    def __init__(self, predictions):

        self.predictions = predictions.copy()

        self.predictions["date"] = pd.to_datetime(
            self.predictions["date"]
        )

    def actual_vs_predicted(self):

        Path("outputs/figures").mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            self.predictions["date"],
            self.predictions["actual_fvi"],
            label="Actual FVI"
        )

        plt.plot(
            self.predictions["date"],
            self.predictions["linear_regression"],
            label="Linear Regression"
        )

        plt.plot(
            self.predictions["date"],
            self.predictions["random_forest"],
            label="Random Forest"
        )

        plt.xlabel("Date")
        plt.ylabel("Financial Vulnerability Index")
        plt.title(
            "Actual vs Predicted Financial Vulnerability Index"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            "outputs/figures/actual_vs_predicted_fvi.png",
            dpi=300
        )

        plt.close()

        print(
            "Saved: outputs/figures/"
            "actual_vs_predicted_fvi.png"
        )