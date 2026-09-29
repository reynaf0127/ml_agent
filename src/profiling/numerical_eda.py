import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class NumericalEDA:

    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
        output_dir: str = "artifacts/eda/numerical",
    ):
        self.df = df
        self.num_cols = num_cols
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        sns.set_theme(
            style="whitegrid",
            context="notebook",
        )

    # --------------------------------------------------
    # Ask for optional group-by column
    # --------------------------------------------------

    def ask_groupby_column(self) -> str | None:

        print("\n" + "=" * 60)
        print("NUMERICAL EDA")
        print("=" * 60)

        group_col = input(
            "\nEnter a target/group column for numerical means "
            "(or press Enter for overall means):\n> "
        ).strip()

        if not group_col:
            return None

        if group_col not in self.df.columns:
            raise ValueError(
                f"Column '{group_col}' not found in dataset."
            )

        return group_col

    # --------------------------------------------------
    # Numerical means
    # --------------------------------------------------

    def print_means(
        self,
        group_col: str | None = None,
    ) -> None:

        print("\n" + "=" * 60)
        print("NUMERICAL MEANS")
        print("=" * 60)

        if group_col is None:

            means = (
                self.df[self.num_cols]
                .mean()
                .round(2)
            )

            print("\nOverall means:")
            print(means)

        else:

            # Don't aggregate group column itself if it
            # happens to also be numerical.
            analysis_cols = [
                col
                for col in self.num_cols
                if col != group_col
            ]

            means = (
                self.df
                .groupby(
                    group_col,
                    dropna=False,
                )[analysis_cols]
                .mean()
                .round(2)
            )

            print(
                f"\nMean numerical values grouped by "
                f"'{group_col}':"
            )

            print(means)

    def numerical_diagnostics(self):

        rows = []

        for col in self.num_cols:

            s = self.df[col].dropna()

            skew = s.skew()

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask = (
                (s < lower) |
                (s > upper)
            )

            outlier_pct = (
                outlier_mask.mean() * 100
            )

            if skew > 0:
                direction = "right"
            elif skew < 0:
                direction = "left"
            else:
                direction = "symmetric"

            rows.append({
                "column": col,
                "mean": s.mean(),
                "median": s.median(),
                "skewness": skew,
                "skew_direction": direction,
                "outlier_pct": outlier_pct,
                "min": s.min(),
                "q01": s.quantile(0.01),
                "q25": q1,
                "q50": s.median(),
                "q75": q3,
                "q99": s.quantile(0.99),
                "max": s.max(),
            })

        diagnostics = pd.DataFrame(rows)

        print("\n" + "=" * 90)
        print("NUMERICAL DIAGNOSTICS")
        print("=" * 90)

        print(
            diagnostics.round(3).to_string(
                index=False
            )
        )

        return diagnostics

    def classify_skew(skew):

        abs_skew = abs(skew)

        if abs_skew < 0.5:
            return "low"

        elif abs_skew < 1:
            return "moderate"

        return "high"

    # --------------------------------------------------
    # Boxplots
    # --------------------------------------------------

    def plot_boxplots(self) -> None:

        if not self.num_cols:
            print(
                "\n[EDA] No numerical columns available "
                "for boxplots."
            )
            return

        n_cols = 2

        n_rows = math.ceil(
            len(self.num_cols) / n_cols
        )

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(
                14,
                4 * n_rows,
            ),
        )

        # Make axes always iterable
        axes = (
            axes.flatten()
            if hasattr(axes, "flatten")
            else [axes]
        )

        for i, col in enumerate(self.num_cols):

            ax = axes[i]

            sns.boxplot(
                data=self.df,
                x=col,
                ax=ax,
            )

            ax.set_title(
                col.replace("_", " ").title(),
                fontsize=13,
                fontweight="bold",
            )

            ax.set_xlabel(
                col.replace("_", " ").title()
            )

            ax.set_ylabel("")

        # Remove unused subplot(s)
        for i in range(
            len(self.num_cols),
            len(axes),
        ):
            fig.delaxes(axes[i])

        fig.suptitle(
            "Numerical Feature Distributions",
            fontsize=18,
            fontweight="bold",
            y=1.01,
        )

        sns.despine()

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "numerical_boxplots.png",
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        print(
            f"\n[EDA] Boxplots saved to: {path}"
        )

        plt.show()
        plt.close()

    # --------------------------------------------------
    # Main workflow
    # --------------------------------------------------

    def run(self):

        if not self.num_cols:
            print(
                "\n[EDA] No numerical columns available."
            )
            return

        # --------------------------------------------------
        # 1. Numerical diagnostics
        # --------------------------------------------------

        diagnostics = self.numerical_diagnostics()

        # --------------------------------------------------
        # 2. Ask optional group-by column
        # --------------------------------------------------

        group_col = self.ask_groupby_column()

        # --------------------------------------------------
        # 3. Print numerical means
        # --------------------------------------------------

        self.print_means(
            group_col=group_col
        )

        # --------------------------------------------------
        # 4. Boxplots
        # --------------------------------------------------

        self.plot_boxplots()

        return diagnostics