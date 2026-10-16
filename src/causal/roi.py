import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CausalROIAnalyzer:
    """
    Convert predicted causal uplift into business value.

    Core idea:

        Expected incremental outcomes
            = predicted uplift

        Expected incremental value
            = predicted uplift * outcome value

        Expected net value
            = expected incremental value - treatment cost

    Treatment is economically worthwhile when:

        predicted uplift * outcome value > treatment cost

    Therefore the break-even uplift is:

        treatment cost / outcome value

    Supports:
    - Constant outcome value
    - Constant treatment cost
    - Optional row-level value column
    - Optional row-level cost column
    - Budget constraints
    - ROI targeting curve
    - Economic treatment recommendations

    Important:
    This module does NOT estimate causal effects.
    It consumes uplift / CATE estimates produced by
    the causal model.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/roi",
    ):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate(
        self,
        df,
        uplift_col,
        outcome_value=None,
        treatment_cost=None,
        outcome_value_col=None,
        treatment_cost_col=None,
    ):

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if uplift_col not in df.columns:

            raise ValueError(
                "Uplift column not found: "
                f"{uplift_col}"
            )

        uplift = pd.to_numeric(
            df[uplift_col],
            errors="coerce",
        )

        if uplift.isna().any():

            raise ValueError(
                "Uplift column contains missing "
                "or non-numeric values."
            )

        # ----------------------------------------------------
        # Outcome value
        # ----------------------------------------------------

        if outcome_value_col is not None:

            if outcome_value_col not in df.columns:

                raise ValueError(
                    "Outcome value column not found: "
                    f"{outcome_value_col}"
                )

        elif outcome_value is None:

            raise ValueError(
                "Provide either outcome_value "
                "or outcome_value_col."
            )

        # ----------------------------------------------------
        # Treatment cost
        # ----------------------------------------------------

        if treatment_cost_col is not None:

            if treatment_cost_col not in df.columns:

                raise ValueError(
                    "Treatment cost column not found: "
                    f"{treatment_cost_col}"
                )

        elif treatment_cost is None:

            raise ValueError(
                "Provide either treatment_cost "
                "or treatment_cost_col."
            )

    # ========================================================
    # BUSINESS VALUE VECTORS
    # ========================================================

    def _get_business_values(
        self,
        df,
        outcome_value=None,
        treatment_cost=None,
        outcome_value_col=None,
        treatment_cost_col=None,
    ):

        # ----------------------------------------------------
        # Outcome value
        # ----------------------------------------------------

        if outcome_value_col is not None:

            value = pd.to_numeric(
                df[outcome_value_col],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

        else:

            value = np.full(
                len(df),
                float(outcome_value),
                dtype=float,
            )

        # ----------------------------------------------------
        # Treatment cost
        # ----------------------------------------------------

        if treatment_cost_col is not None:

            cost = pd.to_numeric(
                df[treatment_cost_col],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

        else:

            cost = np.full(
                len(df),
                float(treatment_cost),
                dtype=float,
            )

        if np.isnan(value).any():

            raise ValueError(
                "Outcome value contains missing "
                "or non-numeric values."
            )

        if np.isnan(cost).any():

            raise ValueError(
                "Treatment cost contains missing "
                "or non-numeric values."
            )

        if (value < 0).any():

            raise ValueError(
                "Outcome value cannot be negative."
            )

        if (cost < 0).any():

            raise ValueError(
                "Treatment cost cannot be negative."
            )

        return (
            value,
            cost,
        )

    # ========================================================
    # BREAK-EVEN UPLIFT
    # ========================================================

    @staticmethod
    def calculate_break_even_uplift(
        outcome_value,
        treatment_cost,
    ):

        outcome_value = np.asarray(
            outcome_value,
            dtype=float,
        )

        treatment_cost = np.asarray(
            treatment_cost,
            dtype=float,
        )

        return np.divide(
            treatment_cost,
            outcome_value,
            out=np.full_like(
                treatment_cost,
                np.inf,
                dtype=float,
            ),
            where=(
                outcome_value > 0
            ),
        )

    # ========================================================
    # OBSERVATION-LEVEL ECONOMICS
    # ========================================================

    def calculate_row_economics(
        self,
        df,
        uplift_col,
        outcome_value=None,
        treatment_cost=None,
        outcome_value_col=None,
        treatment_cost_col=None,
    ):

        result = df.copy()

        uplift = pd.to_numeric(
            result[uplift_col],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        (
            value,
            cost,
        ) = self._get_business_values(
            df=result,
            outcome_value=outcome_value,
            treatment_cost=treatment_cost,
            outcome_value_col=outcome_value_col,
            treatment_cost_col=treatment_cost_col,
        )

        break_even = (
            self.calculate_break_even_uplift(
                outcome_value=value,
                treatment_cost=cost,
            )
        )

        expected_incremental_value = (
            uplift
            * value
        )

        expected_net_value = (
            expected_incremental_value
            - cost
        )

        result[
            "causal_outcome_value"
        ] = value

        result[
            "causal_treatment_cost"
        ] = cost

        result[
            "break_even_uplift"
        ] = break_even

        result[
            "expected_incremental_outcome"
        ] = uplift

        result[
            "expected_incremental_value"
        ] = expected_incremental_value

        result[
            "expected_net_value"
        ] = expected_net_value

        result[
            "economically_positive"
        ] = (
            expected_net_value
            > 0
        )

        return result

    # ========================================================
    # UNCONSTRAINED ECONOMIC POLICY
    # ========================================================

    def apply_economic_policy(
        self,
        df,
    ):

        result = df.copy()

        result[
            "roi_recommended_treatment"
        ] = (
            result[
                "expected_net_value"
            ]
            > 0
        ).astype(int)

        return result

    # ========================================================
    # BUDGET-CONSTRAINED POLICY
    # ========================================================

    def apply_budget_constraint(
        self,
        df,
        budget,
    ):
        """
        Select economically positive observations while
        respecting the supplied treatment budget.

        Ranking metric:

            expected_net_value / treatment_cost

        For zero-cost treatment, economically positive
        observations are prioritized by expected net value.
        """

        if budget is None:
            return self.apply_economic_policy(
                df
            )

        budget = float(
            budget
        )

        if budget < 0:

            raise ValueError(
                "Budget cannot be negative."
            )

        result = df.copy()

        result[
            "roi_recommended_treatment"
        ] = 0

        positive = result.loc[
            result[
                "expected_net_value"
            ] > 0
        ].copy()

        if positive.empty:
            return result

        cost = (
            positive[
                "causal_treatment_cost"
            ]
            .astype(float)
        )

        net_value = (
            positive[
                "expected_net_value"
            ]
            .astype(float)
        )

        positive[
            "_roi_priority"
        ] = np.where(
            cost > 0,
            net_value / cost,
            np.inf,
        )

        positive = (
            positive
            .sort_values(
                [
                    "_roi_priority",
                    "expected_net_value",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        cumulative_cost = (
            positive[
                "causal_treatment_cost"
            ]
            .cumsum()
        )

        selected = (
            cumulative_cost
            <= budget
        )

        selected_index = (
            positive.loc[
                selected
            ].index
        )

        result.loc[
            selected_index,
            "roi_recommended_treatment",
        ] = 1

        return result

    # ========================================================
    # POLICY SUMMARY
    # ========================================================

    def summarize_policy(
        self,
        df,
    ):

        target = (
            df[
                "roi_recommended_treatment"
            ]
            == 1
        )

        targeted = (
            df.loc[
                target
            ]
        )

        total_rows = len(
            df
        )

        targeted_rows = len(
            targeted
        )

        target_fraction = (
            targeted_rows
            / total_rows
            if total_rows > 0
            else np.nan
        )

        incremental_outcomes = (
            targeted[
                "expected_incremental_outcome"
            ].sum()
        )

        incremental_value = (
            targeted[
                "expected_incremental_value"
            ].sum()
        )

        total_cost = (
            targeted[
                "causal_treatment_cost"
            ].sum()
        )

        net_value = (
            targeted[
                "expected_net_value"
            ].sum()
        )

        roi = (
            net_value
            / total_cost
            if total_cost > 0
            else np.nan
        )

        mean_uplift = (
            targeted[
                "expected_incremental_outcome"
            ].mean()
            if targeted_rows > 0
            else np.nan
        )

        summary = pd.DataFrame([
            {
                "observations":
                    total_rows,

                "recommended_treatments":
                    targeted_rows,

                "recommended_target_fraction":
                    target_fraction,

                "mean_targeted_uplift":
                    mean_uplift,

                "expected_incremental_outcomes":
                    incremental_outcomes,

                "expected_incremental_value":
                    incremental_value,

                "expected_treatment_cost":
                    total_cost,

                "expected_net_value":
                    net_value,

                "expected_roi":
                    roi,
            }
        ])

        return summary

    # ========================================================
    # TARGETING CURVE
    # ========================================================

    def targeting_curve(
        self,
        df,
        step=0.05,
    ):
        """
        Evaluate business value when targeting the top
        observations ranked by expected net value.

        This is useful for visualizing where marginal
        targeting becomes economically unattractive.
        """

        if (
            step <= 0
            or step > 1
        ):

            raise ValueError(
                "step must be between 0 and 1."
            )

        ranked = (
            df.copy()
            .sort_values(
                "expected_net_value",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        fractions = np.arange(
            step,
            1.0 + step,
            step,
        )

        fractions = np.clip(
            fractions,
            0,
            1,
        )

        fractions = np.unique(
            fractions
        )

        rows = []

        n = len(
            ranked
        )

        for fraction in fractions:

            target_n = int(
                np.ceil(
                    n
                    * fraction
                )
            )

            target_n = min(
                target_n,
                n,
            )

            subset = (
                ranked.iloc[
                    :target_n
                ]
            )

            incremental_outcomes = (
                subset[
                    "expected_incremental_outcome"
                ].sum()
            )

            incremental_value = (
                subset[
                    "expected_incremental_value"
                ].sum()
            )

            treatment_cost = (
                subset[
                    "causal_treatment_cost"
                ].sum()
            )

            net_value = (
                subset[
                    "expected_net_value"
                ].sum()
            )

            roi = (
                net_value
                / treatment_cost
                if treatment_cost > 0
                else np.nan
            )

            rows.append({
                "target_fraction":
                    fraction,

                "target_pct":
                    fraction * 100,

                "observations":
                    target_n,

                "mean_uplift":
                    subset[
                        "expected_incremental_outcome"
                    ].mean(),

                "expected_incremental_outcomes":
                    incremental_outcomes,

                "expected_incremental_value":
                    incremental_value,

                "treatment_cost":
                    treatment_cost,

                "expected_net_value":
                    net_value,

                "expected_roi":
                    roi,
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # PLOT NET VALUE
    # ========================================================

    def plot_net_value_curve(
        self,
        curve_df,
    ):

        path = os.path.join(
            self.output_dir,
            "net_value_by_target_pct.png",
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            curve_df[
                "target_pct"
            ],
            curve_df[
                "expected_net_value"
            ],
            marker="o",
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        plt.xlabel(
            "Targeted Population (%)"
        )

        plt.ylabel(
            "Expected Net Incremental Value"
        )

        plt.title(
            "Expected Net Value by Targeting Fraction"
        )

        plt.grid(
            alpha=0.2
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[ROI] Saved net value curve:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    def print_configuration(
        self,
        df,
    ):

        print("\n" + "=" * 70)
        print("MARGINAL ROI CONFIGURATION")
        print("=" * 70)

        value = (
            df[
                "causal_outcome_value"
            ]
        )

        cost = (
            df[
                "causal_treatment_cost"
            ]
        )

        break_even = (
            df[
                "break_even_uplift"
            ]
        )

        print(
            "\nOutcome value:"
        )

        if value.nunique() == 1:

            print(
                f"  {value.iloc[0]:,.4f}"
            )

        else:

            print(
                "  Row-level values"
            )

            print(
                f"  Mean: "
                f"{value.mean():,.4f}"
            )

        print(
            "\nTreatment cost:"
        )

        if cost.nunique() == 1:

            print(
                f"  {cost.iloc[0]:,.4f}"
            )

        else:

            print(
                "  Row-level costs"
            )

            print(
                f"  Mean: "
                f"{cost.mean():,.4f}"
            )

        print(
            "\nBreak-even uplift:"
        )

        if (
            np.isfinite(
                break_even
            ).all()
            and break_even.nunique() == 1
        ):

            print(
                f"  "
                f"{break_even.iloc[0]:.4%}"
            )

        else:

            finite = (
                break_even[
                    np.isfinite(
                        break_even
                    )
                ]
            )

            if len(finite) > 0:

                print(
                    f"  Mean: "
                    f"{finite.mean():.4%}"
                )

                print(
                    f"  Median: "
                    f"{finite.median():.4%}"
                )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    def print_summary(
        self,
        summary,
    ):

        row = (
            summary.iloc[0]
        )

        print("\n" + "=" * 70)
        print("MARGINAL ROI RESULTS")
        print("=" * 70)

        print(
            "\nRecommended target fraction: "
            f"{row['recommended_target_fraction']:.2%}"
        )

        print(
            "Recommended observations: "
            f"{int(row['recommended_treatments']):,}"
        )

        if pd.notna(
            row[
                "mean_targeted_uplift"
            ]
        ):

            print(
                "Mean targeted uplift: "
                f"{row['mean_targeted_uplift']:.4%}"
            )

        print(
            "\nExpected incremental outcomes: "
            f"{row['expected_incremental_outcomes']:,.2f}"
        )

        print(
            "Expected incremental value: "
            f"{row['expected_incremental_value']:,.2f}"
        )

        print(
            "Expected treatment cost: "
            f"{row['expected_treatment_cost']:,.2f}"
        )

        print(
            "Expected net value: "
            f"{row['expected_net_value']:,.2f}"
        )

        if pd.notna(
            row[
                "expected_roi"
            ]
        ):

            print(
                "Expected ROI: "
                f"{row['expected_roi']:.2%}"
            )

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        df,
        uplift_col="uplift_score",
        outcome_value=None,
        treatment_cost=None,
        outcome_value_col=None,
        treatment_cost_col=None,
        budget=None,
        targeting_step=0.05,
    ):

        print("\n" + "=" * 70)
        print("MARGINAL ROI ANALYSIS")
        print("=" * 70)

        print(
            f"\nRows: "
            f"{len(df):,}"
        )

        print(
            f"Uplift column: "
            f"{uplift_col}"
        )

        # ====================================================
        # VALIDATE
        # ====================================================

        self._validate(
            df=df,
            uplift_col=uplift_col,
            outcome_value=outcome_value,
            treatment_cost=treatment_cost,
            outcome_value_col=outcome_value_col,
            treatment_cost_col=treatment_cost_col,
        )

        # ====================================================
        # ECONOMIC VALUE
        # ====================================================

        result = (
            self.calculate_row_economics(
                df=df,
                uplift_col=uplift_col,
                outcome_value=outcome_value,
                treatment_cost=treatment_cost,
                outcome_value_col=outcome_value_col,
                treatment_cost_col=treatment_cost_col,
            )
        )

        self.print_configuration(
            result
        )

        # ====================================================
        # APPLY POLICY
        # ====================================================

        if budget is None:

            result = (
                self.apply_economic_policy(
                    result
                )
            )

            print(
                "\nBudget constraint: None"
            )

        else:

            result = (
                self.apply_budget_constraint(
                    df=result,
                    budget=budget,
                )
            )

            print(
                "\nBudget constraint: "
                f"{budget:,.2f}"
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary = (
            self.summarize_policy(
                result
            )
        )

        self.print_summary(
            summary
        )

        # ====================================================
        # TARGETING CURVE
        # ====================================================

        curve = (
            self.targeting_curve(
                df=result,
                step=targeting_step,
            )
        )

        print("\n" + "=" * 70)
        print("ROI TARGETING CURVE")
        print("=" * 70)

        print(
            "\n"
            + curve
            .round(4)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # SAVE
        # ====================================================

        scores_path = os.path.join(
            self.output_dir,
            "causal_roi_scores.csv",
        )

        summary_path = os.path.join(
            self.output_dir,
            "causal_roi_summary.csv",
        )

        curve_path = os.path.join(
            self.output_dir,
            "causal_roi_curve.csv",
        )

        result.to_csv(
            scores_path,
            index=False,
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        curve.to_csv(
            curve_path,
            index=False,
        )

        plot_path = (
            self.plot_net_value_curve(
                curve
            )
        )

        print(
            "\n[ROI] Saved scoring table:"
            f"\n{scores_path}"
        )

        print(
            "[ROI] Saved summary:"
            f"\n{summary_path}"
        )

        print(
            "[ROI] Saved targeting curve:"
            f"\n{curve_path}"
        )

        print("\n" + "=" * 70)
        print("MARGINAL ROI ANALYSIS COMPLETE")
        print("=" * 70)

        print(
            "\nThe economically optimal policy "
            "targets observations whose expected "
            "incremental value exceeds treatment cost."
        )

        return {
            "scores":
                result,

            "summary":
                summary,

            "curve":
                curve,

            "scores_path":
                scores_path,

            "summary_path":
                summary_path,

            "curve_path":
                curve_path,

            "plot_path":
                plot_path,
        }