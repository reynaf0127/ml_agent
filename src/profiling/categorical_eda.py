import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class CategoricalEDA:

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        num_cols: list[str],
        output_dir: str = "artifacts/eda/categorical",
    ):
        self.df = df
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Seaborn formatting
        sns.set_theme(
            style="whitegrid",
            context="notebook",
        )

    # --------------------------------------------------
    # Terminal selection
    # --------------------------------------------------

    def ask_columns(self) -> list[str]:
        """
        Ask which categorical columns should be analyzed.

        Press Enter to skip categorical EDA.
        """

        if not self.cat_cols:
            print("\n[EDA] No categorical columns available.")
            return []

        print("\n" + "=" * 60)
        print("CATEGORICAL EDA")
        print("=" * 60)

        print("\nAvailable categorical columns:")

        for i, col in enumerate(self.cat_cols, start=1):
            print(f"  {i}. {col}")

        user_input = input(
            "\nSelect categorical columns "
            "(comma separated, or press Enter to skip):\n> "
        ).strip()

        if not user_input:
            print("[EDA] Categorical EDA skipped.")
            return []

        selected_cols = [
            col.strip()
            for col in user_input.split(",")
            if col.strip()
        ]

        invalid = [
            col
            for col in selected_cols
            if col not in self.cat_cols
        ]

        if invalid:
            raise ValueError(
                f"Invalid categorical columns: {invalid}"
            )

        return selected_cols

    # --------------------------------------------------
    # Target selection
    # --------------------------------------------------

    def ask_target(self) -> str | None:
        """
        Ask for an optional numerical target.

        Press Enter to skip target-based charts.
        """

        if not self.num_cols:
            return None

        print("\nAvailable numerical columns:")

        for i, col in enumerate(self.num_cols, start=1):
            print(f"  {i}. {col}")

        target = input(
            "\nEnter target column for group mean "
            "(or press Enter to skip):\n> "
        ).strip()

        if not target:
            print("[EDA] Target analysis skipped.")
            return None

        if target not in self.num_cols:
            raise ValueError(
                f"'{target}' is not a valid numerical column."
            )

        return target

    # --------------------------------------------------
    # Value counts
    # --------------------------------------------------

    def value_counts(
        self,
        cat_col: str,
    ) -> pd.DataFrame:

        result = (
            self.df[cat_col]
            .value_counts(
                normalize=True,
                dropna=False,
            )
            .rename("proportion")
            .reset_index()
        )

        result["percentage"] = (
            result["proportion"] * 100
        )

        return result

    # --------------------------------------------------
    # Group means
    # --------------------------------------------------

    def group_means(
        self,
        cat_col: str,
    ) -> pd.DataFrame:

        return (
            self.df
            .groupby(
                cat_col,
                dropna=False,
            )[self.num_cols]
            .mean()
            .reset_index()
        )

    # --------------------------------------------------
    # Plot value counts
    # --------------------------------------------------

    def plot_value_counts(
        self,
        cat_col: str,
    ) -> None:

        plot_df = self.value_counts(cat_col)

        plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            data=plot_df,
            x=cat_col,
            y="percentage",
            hue=cat_col,
            palette="viridis",
            legend=False,
        )

        ax.set_title(
            f"Distribution of {cat_col}",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel(
            cat_col.replace("_", " ").title()
        )

        ax.set_ylabel("Percentage (%)")

        # Percentage labels
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f%%",
                padding=3,
                fontsize=9,
            )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        sns.despine()

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            f"{cat_col}_distribution.png",
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()

    # --------------------------------------------------
    # Plot target mean
    # --------------------------------------------------

    def plot_target_mean(
        self,
        cat_col: str,
        target: str,
    ) -> None:

        plot_df = (
            self.df
            .groupby(
                cat_col,
                dropna=False,
            )[target]
            .mean()
            .reset_index()
            .sort_values(
                target,
                ascending=False,
            )
        )

        plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            data=plot_df,
            x=cat_col,
            y=target,
            hue=cat_col,
            palette="mako",
            legend=False,
        )

        ax.set_title(
            f"Mean {target} by {cat_col}",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel(
            cat_col.replace("_", " ").title()
        )

        ax.set_ylabel(
            f"Mean {target.replace('_', ' ').title()}"
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.2f",
                padding=3,
                fontsize=9,
            )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        sns.despine()

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            f"{cat_col}_by_{target}.png",
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()

    # --------------------------------------------------
    # Main workflow
    # --------------------------------------------------

    def run(self):

        if not self.cat_cols:
            print("\n[EDA] No categorical columns available.")
            return

        # --------------------------------------------------
        # 1. Print ALL categorical value counts
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("CATEGORICAL DATA SUMMARY")
        print("=" * 60)

        for cat_col in self.cat_cols:

            print("\n" + "-" * 60)
            print(f"COLUMN: {cat_col}")
            print("-" * 60)

            summary = (
                self.df[cat_col]
                .value_counts(
                    dropna=False
                )
                .rename("count")
                .to_frame()
            )

            summary["percentage"] = (
                self.df[cat_col]
                .value_counts(
                    normalize=True,
                    dropna=False,
                )
                * 100
            )

            summary["percentage"] = (
                summary["percentage"]
                .round(2)
            )

            print(summary)

        # --------------------------------------------------
        # 2. Ask which categorical columns to chart
        # --------------------------------------------------

        selected_cols = self.ask_columns()

        if not selected_cols:
            print("\n[EDA] No charts selected.")
            return

        # --------------------------------------------------
        # 3. Distribution charts
        # --------------------------------------------------

        print("\n[EDA] Creating categorical distribution charts...")

        for cat_col in selected_cols:

            self.plot_value_counts(
                cat_col
            )

        # --------------------------------------------------
        # 4. Ask for optional target
        # --------------------------------------------------

        target = self.ask_target()

        if target is None:
            return

        # --------------------------------------------------
        # 5. Print target means
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print(f"MEAN {target.upper()} BY CATEGORY")
        print("=" * 60)

        for cat_col in selected_cols:

            print(f"\n[{cat_col}]")

            target_summary = (
                self.df
                .groupby(
                    cat_col,
                    dropna=False,
                )[target]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            print(target_summary)

        # --------------------------------------------------
        # 6. Target charts
        # --------------------------------------------------

        print(
            f"\n[EDA] Creating mean {target} charts..."
        )

        for cat_col in selected_cols:

            self.plot_target_mean(
                cat_col=cat_col,
                target=target,
            )