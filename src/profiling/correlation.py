import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)


class CorrelationAnalysis:

    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
        bool_cols: list[str],
        excl_cols: list[str],
        output_dir: str = "artifacts/eda/correlation",
    ):
        self.df = df
        self.num_cols = num_cols
        self.bool_cols = bool_cols
        self.excl_cols = excl_cols
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    def ask_target(self):

        print("\n" + "=" * 60)
        print("CORRELATION / MULTICOLLINEARITY ANALYSIS")
        print("=" * 60)

        target = input(
            "\nEnter target column:\n> "
        ).strip()

        if target not in self.df.columns:
            raise ValueError(
                f"Target '{target}' not found."
            )

        return target

    # --------------------------------------------------
    # Correlation columns
    # --------------------------------------------------

    def get_correlation_columns(
        self,
        target,
    ):

        cols = (
            self.num_cols
            + self.bool_cols
        )

        # Remove excluded columns
        cols = [
            col
            for col in cols
            if col not in self.excl_cols
        ]

        # Remove target first to avoid duplicate
        cols = [
            col
            for col in cols
            if col != target
        ]

        # Add target if numeric
        if pd.api.types.is_numeric_dtype(
            self.df[target]
        ):
            cols.append(target)

        # Remove duplicates while preserving order
        cols = list(dict.fromkeys(cols))

        return cols

    # --------------------------------------------------
    # Correlation
    # --------------------------------------------------

    def correlation_matrix(
        self,
        target,
    ):

        cols = self.get_correlation_columns(
            target
        )

        corr = (
            self.df[cols]
            .corr()
        )

        return corr

    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------

    def plot_heatmap(
        self,
        corr,
    ):

        n_features = len(corr.columns)

        size = max(
            10,
            n_features * 0.7,
        )

        plt.figure(
            figsize=(size, size)
        )

        # Hide duplicate upper triangle
        mask = np.triu(
            np.ones_like(
                corr,
                dtype=bool,
            )
        )

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={
                "shrink": 0.8
            },
        )

        plt.title(
            "Feature Correlation Matrix",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "correlation_heatmap.png",
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        print(
            f"\n[EDA] Correlation heatmap saved to {path}"
        )

        plt.show()
        plt.close()

    # --------------------------------------------------
    # Target correlations
    # --------------------------------------------------

    def print_target_correlations(
        self,
        corr,
        target,
    ):

        if target not in corr.columns:
            return

        target_corr = (
            corr[target]
            .drop(target)
            .sort_values(
                key=abs,
                ascending=False,
            )
        )

        print("\n" + "=" * 60)
        print(f"CORRELATION WITH TARGET: {target}")
        print("=" * 60)

        print(
            target_corr.round(3)
        )

    # --------------------------------------------------
    # VIF
    # --------------------------------------------------

    def calculate_vif(
        self,
        target,
    ):

        # Continuous numerical predictors only
        vif_cols = [
            col
            for col in self.num_cols
            if col != target
            and col not in self.excl_cols
        ]

        X = (
            self.df[vif_cols]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .copy()
        )

        # Remove constant columns
        constant_cols = [
            col
            for col in X.columns
            if X[col].nunique() <= 1
        ]

        if constant_cols:

            print(
                "\n[VIF] Removing constant columns:"
            )

            print(constant_cols)

            X = X.drop(
                columns=constant_cols
            )

        if X.shape[1] < 2:

            print(
                "\n[VIF] Not enough numerical "
                "predictors to calculate VIF."
            )

            return pd.DataFrame()

        vif = pd.DataFrame({
            "feature": X.columns,
            "VIF": [
                variance_inflation_factor(
                    X.values,
                    i,
                )
                for i in range(X.shape[1])
            ],
        })

        vif = (
            vif
            .sort_values(
                "VIF",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return vif

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self):

        target = self.ask_target()

        # Correlation
        corr = self.correlation_matrix(
            target
        )

        self.print_target_correlations(
            corr,
            target,
        )

        self.plot_heatmap(
            corr
        )

        # VIF
        vif = self.calculate_vif(
            target
        )

        if not vif.empty:

            print("\n" + "=" * 60)
            print("VARIANCE INFLATION FACTOR")
            print("=" * 60)

            print(
                vif.round(3).to_string(
                    index=False
                )
            )

        return target, corr, vif