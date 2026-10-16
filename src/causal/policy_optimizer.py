import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CausalPolicyOptimizer:
    """
    Optimize causal targeting policy.

    The optimizer assumes observations have already been scored
    by an uplift / CATE model.

    It evaluates policies such as:

        target top 5%
        target top 10%
        target top 15%
        ...
        target top 100%

    For each candidate targeting fraction it calculates:

        - number targeted
        - mean predicted uplift
        - expected incremental outcomes
        - expected incremental value, if available
        - treatment cost, if available
        - expected net value, if available
        - ROI, if available

    The optimizer can select a policy according to:

        incremental_outcomes
        incremental_outcomes_per_target
        net_value
        roi

    No dataset-specific feature names are hardcoded.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/policy_optimization",
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
                f"Uplift column not found: "
                f"{uplift_col}"
            )

        if len(df) == 0:

            raise ValueError(
                "Policy optimization DataFrame "
                "is empty."
            )

        uplift = pd.to_numeric(
            df[uplift_col],
            errors="coerce",
        )

        if uplift.isna().any():

            raise ValueError(
                f"{uplift_col} contains "
                "missing/non-numeric values."
            )

    # ========================================================
    # TARGET FRACTIONS
    # ========================================================

    def build_target_fractions(
        self,
        start=0.05,
        stop=1.00,
        step=0.05,
    ):

        if start <= 0:

            raise ValueError(
                "start must be > 0."
            )

        if stop > 1:

            raise ValueError(
                "stop must be <= 1."
            )

        if start > stop:

            raise ValueError(
                "start cannot exceed stop."
            )

        if step <= 0:

            raise ValueError(
                "step must be > 0."
            )

        fractions = np.arange(
            start,
            stop + step / 2,
            step,
        )

        fractions = np.clip(
            fractions,
            0,
            1,
        )

        fractions = np.unique(
            np.round(
                fractions,
                10,
            )
        )

        return fractions

    # ========================================================
    # EVALUATE SINGLE POLICY
    # ========================================================

    def evaluate_fraction(
        self,
        ranked_df,
        uplift_col,
        target_fraction,
    ):

        n_total = len(
            ranked_df
        )

        n_target = int(
            np.ceil(
                n_total
                * target_fraction
            )
        )

        n_target = max(
            1,
            min(
                n_target,
                n_total,
            ),
        )

        targeted = (
            ranked_df
            .iloc[
                :n_target
            ]
            .copy()
        )

        mean_uplift = (
            targeted[
                uplift_col
            ]
            .mean()
        )

        median_uplift = (
            targeted[
                uplift_col
            ]
            .median()
        )

        minimum_uplift = (
            targeted[
                uplift_col
            ]
            .min()
        )

        maximum_uplift = (
            targeted[
                uplift_col
            ]
            .max()
        )

        expected_incremental_outcomes = (
            targeted[
                uplift_col
            ]
            .clip(
                lower=0
            )
            .sum()
        )

        incremental_outcomes_per_target = (
            expected_incremental_outcomes
            / n_target
        )

        row = {
            "target_fraction":
                target_fraction,

            "target_pct":
                target_fraction * 100,

            "observations":
                n_target,

            "population_size":
                n_total,

            "mean_predicted_uplift":
                mean_uplift,

            "median_predicted_uplift":
                median_uplift,

            "minimum_predicted_uplift":
                minimum_uplift,

            "maximum_predicted_uplift":
                maximum_uplift,

            "expected_incremental_outcomes":
                expected_incremental_outcomes,

            "incremental_outcomes_per_target":
                incremental_outcomes_per_target,
        }

        # ====================================================
        # OPTIONAL BUSINESS VALUE
        # ====================================================

        if (
            "expected_incremental_value"
            in targeted.columns
        ):

            incremental_value = (
                targeted[
                    "expected_incremental_value"
                ]
                .sum()
            )

            row[
                "expected_incremental_value"
            ] = incremental_value

        # ====================================================
        # OPTIONAL TREATMENT COST
        # ====================================================

        if (
            "treatment_cost"
            in targeted.columns
        ):

            treatment_cost = (
                targeted[
                    "treatment_cost"
                ]
                .sum()
            )

            row[
                "expected_treatment_cost"
            ] = treatment_cost

        # ====================================================
        # OPTIONAL NET VALUE
        # ====================================================

        if (
            "expected_net_value"
            in targeted.columns
        ):

            net_value = (
                targeted[
                    "expected_net_value"
                ]
                .sum()
            )

            row[
                "expected_net_value"
            ] = net_value

        # ====================================================
        # ROI
        # ====================================================

        if (
            "expected_incremental_value"
            in row
            and
            "expected_treatment_cost"
            in row
        ):

            cost = row[
                "expected_treatment_cost"
            ]

            value = row[
                "expected_incremental_value"
            ]

            if cost > 0:

                row[
                    "expected_roi"
                ] = (
                    value
                    - cost
                ) / cost

            else:

                row[
                    "expected_roi"
                ] = np.nan

        return row

    # ========================================================
    # POLICY GRID
    # ========================================================

    def evaluate_grid(
        self,
        df,
        uplift_col,
        target_fractions,
    ):

        ranked_df = (
            df
            .sort_values(
                uplift_col,
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        rows = []

        for fraction in target_fractions:

            row = (
                self.evaluate_fraction(
                    ranked_df=ranked_df,
                    uplift_col=uplift_col,
                    target_fraction=float(
                        fraction
                    ),
                )
            )

            rows.append(
                row
            )

        results = pd.DataFrame(
            rows
        )

        return results

    # ========================================================
    # OBJECTIVE VALIDATION
    # ========================================================

    def validate_objective(
        self,
        results,
        objective,
    ):

        supported = [
            "incremental_outcomes",
            "incremental_outcomes_per_target",
            "net_value",
            "roi",
        ]

        if objective not in supported:

            raise ValueError(
                "Unknown optimization objective: "
                f"{objective}. "
                f"Supported objectives: "
                f"{supported}"
            )

        objective_column_map = {
            "incremental_outcomes":
                "expected_incremental_outcomes",

            "incremental_outcomes_per_target":
                "incremental_outcomes_per_target",

            "net_value":
                "expected_net_value",

            "roi":
                "expected_roi",
        }

        objective_col = (
            objective_column_map[
                objective
            ]
        )

        if objective_col not in results.columns:

            if objective in [
                "net_value",
                "roi",
            ]:

                raise ValueError(
                    f"Cannot optimize '{objective}'. "
                    "Business value and treatment "
                    "cost are required."
                )

            raise ValueError(
                "Optimization metric missing: "
                f"{objective_col}"
            )

        return objective_col

    # ========================================================
    # SELECT BEST POLICY
    # ========================================================

    def select_best_policy(
        self,
        results,
        objective,
        minimum_mean_uplift=None,
        maximum_target_fraction=None,
    ):

        objective_col = (
            self.validate_objective(
                results=results,
                objective=objective,
            )
        )

        candidates = (
            results.copy()
        )

        # ----------------------------------------------------
        # Optional minimum uplift requirement
        # ----------------------------------------------------

        if minimum_mean_uplift is not None:

            candidates = (
                candidates[
                    candidates[
                        "mean_predicted_uplift"
                    ]
                    >= float(
                        minimum_mean_uplift
                    )
                ]
            )

        # ----------------------------------------------------
        # Optional targeting budget / capacity constraint
        # ----------------------------------------------------

        if maximum_target_fraction is not None:

            maximum_target_fraction = float(
                maximum_target_fraction
            )

            if not (
                0
                < maximum_target_fraction
                <= 1
            ):

                raise ValueError(
                    "maximum_target_fraction "
                    "must be between 0 and 1."
                )

            candidates = (
                candidates[
                    candidates[
                        "target_fraction"
                    ]
                    <= maximum_target_fraction
                ]
            )

        if candidates.empty:

            raise ValueError(
                "No candidate targeting policy "
                "satisfies the optimization "
                "constraints."
            )

        candidates = (
            candidates
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna(
                subset=[
                    objective_col
                ]
            )
        )

        if candidates.empty:

            raise ValueError(
                "No valid optimization values "
                "remain after filtering."
            )

        best_index = (
            candidates[
                objective_col
            ]
            .idxmax()
        )

        best_policy = (
            candidates
            .loc[
                best_index
            ]
            .copy()
        )

        best_policy[
            "optimization_objective"
        ] = objective

        return (
            best_policy,
            objective_col,
        )

    # ========================================================
    # PRINT GRID
    # ========================================================

    def print_grid(
        self,
        results,
    ):

        print("\n" + "=" * 70)
        print("TARGETING POLICY COMPARISON")
        print("=" * 70)

        display_cols = [
            "target_pct",
            "observations",
            "mean_predicted_uplift",
            "expected_incremental_outcomes",
            "incremental_outcomes_per_target",
        ]

        optional_cols = [
            "expected_incremental_value",
            "expected_treatment_cost",
            "expected_net_value",
            "expected_roi",
        ]

        for col in optional_cols:

            if col in results.columns:

                display_cols.append(
                    col
                )

        print(
            "\n"
            + results[
                display_cols
            ]
            .round(6)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # PRINT RECOMMENDATION
    # ========================================================

    def print_recommendation(
        self,
        best_policy,
        objective,
    ):

        print("\n" + "=" * 70)
        print("RECOMMENDED TARGETING POLICY")
        print("=" * 70)

        print(
            "\nOptimization objective: "
            f"{objective}"
        )

        print(
            "\nRecommended target fraction: "
            f"{best_policy['target_fraction']:.2%}"
        )

        print(
            "Recommended observations: "
            f"{int(best_policy['observations']):,}"
        )

        print(
            "Mean predicted uplift: "
            f"{best_policy['mean_predicted_uplift']:.4%}"
        )

        print(
            "Expected incremental outcomes: "
            f"{best_policy['expected_incremental_outcomes']:,.2f}"
        )

        print(
            "Incremental outcomes per target: "
            f"{best_policy['incremental_outcomes_per_target']:.6f}"
        )

        if (
            "expected_incremental_value"
            in best_policy.index
        ):

            print(
                "Expected incremental value: "
                f"{best_policy['expected_incremental_value']:,.2f}"
            )

        if (
            "expected_treatment_cost"
            in best_policy.index
        ):

            print(
                "Expected treatment cost: "
                f"{best_policy['expected_treatment_cost']:,.2f}"
            )

        if (
            "expected_net_value"
            in best_policy.index
        ):

            print(
                "Expected net value: "
                f"{best_policy['expected_net_value']:,.2f}"
            )

        if (
            "expected_roi"
            in best_policy.index
            and
            pd.notna(
                best_policy[
                    "expected_roi"
                ]
            )
        ):

            print(
                "Expected ROI: "
                f"{best_policy['expected_roi']:.2%}"
            )

    # ========================================================
    # PLOT INCREMENTAL OUTCOMES
    # ========================================================

    def plot_incremental_outcomes(
        self,
        results,
        best_policy,
    ):

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            results[
                "target_pct"
            ],
            results[
                "expected_incremental_outcomes"
            ],
            marker="o",
        )

        plt.axvline(
            best_policy[
                "target_pct"
            ],
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "Targeted Population (%)"
        )

        plt.ylabel(
            "Expected Incremental Outcomes"
        )

        plt.title(
            "Policy Optimization: "
            "Incremental Outcomes"
        )

        plt.grid(
            alpha=0.2
        )

        path = os.path.join(
            self.output_dir,
            "incremental_outcomes_by_target_pct.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "\n[POLICY OPTIMIZER] "
            "Saved incremental outcome plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # PLOT EFFICIENCY
    # ========================================================

    def plot_efficiency(
        self,
        results,
        best_policy,
    ):

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            results[
                "target_pct"
            ],
            results[
                "incremental_outcomes_per_target"
            ],
            marker="o",
        )

        plt.axvline(
            best_policy[
                "target_pct"
            ],
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "Targeted Population (%)"
        )

        plt.ylabel(
            "Incremental Outcomes per Target"
        )

        plt.title(
            "Policy Optimization: "
            "Targeting Efficiency"
        )

        plt.grid(
            alpha=0.2
        )

        path = os.path.join(
            self.output_dir,
            "targeting_efficiency.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        return path

    # ========================================================
    # BUSINESS VALUE PLOT
    # ========================================================

    def plot_business_objective(
        self,
        results,
        best_policy,
        objective_col,
    ):

        if objective_col not in [
            "expected_net_value",
            "expected_roi",
        ]:

            return None

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            results[
                "target_pct"
            ],
            results[
                objective_col
            ],
            marker="o",
        )

        plt.axvline(
            best_policy[
                "target_pct"
            ],
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "Targeted Population (%)"
        )

        if (
            objective_col
            == "expected_net_value"
        ):

            ylabel = (
                "Expected Net Value"
            )

            title = (
                "Policy Optimization: "
                "Expected Net Value"
            )

            filename = (
                "expected_net_value.png"
            )

        else:

            ylabel = (
                "Expected ROI"
            )

            title = (
                "Policy Optimization: "
                "Expected ROI"
            )

            filename = (
                "expected_roi.png"
            )

        plt.ylabel(
            ylabel
        )

        plt.title(
            title
        )

        plt.grid(
            alpha=0.2
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        return path

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    def save_results(
        self,
        results,
        best_policy,
    ):

        grid_path = os.path.join(
            self.output_dir,
            "policy_optimization_grid.csv",
        )

        best_path = os.path.join(
            self.output_dir,
            "recommended_policy.csv",
        )

        results.to_csv(
            grid_path,
            index=False,
        )

        pd.DataFrame([
            best_policy
        ]).to_csv(
            best_path,
            index=False,
        )

        print(
            "\n[POLICY OPTIMIZER] "
            "Saved optimization grid:"
            f"\n{grid_path}"
        )

        print(
            "[POLICY OPTIMIZER] "
            "Saved recommended policy:"
            f"\n{best_path}"
        )

        return {
            "grid_path":
                grid_path,

            "best_policy_path":
                best_path,
        }

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        df,
        uplift_col="uplift_score",
        objective="incremental_outcomes",
        target_fractions=None,
        start=0.05,
        stop=1.00,
        step=0.05,
        minimum_mean_uplift=None,
        maximum_target_fraction=None,
    ):

        print("\n" + "=" * 70)
        print("CAUSAL POLICY OPTIMIZATION")
        print("=" * 70)

        self._validate(
            df=df,
            uplift_col=uplift_col,
        )

        print(
            f"\nRows: {len(df):,}"
        )

        print(
            f"Uplift column: "
            f"{uplift_col}"
        )

        print(
            f"Optimization objective: "
            f"{objective}"
        )

        # ====================================================
        # TARGET GRID
        # ====================================================

        if target_fractions is None:

            target_fractions = (
                self.build_target_fractions(
                    start=start,
                    stop=stop,
                    step=step,
                )
            )

        else:

            target_fractions = np.asarray(
                target_fractions,
                dtype=float,
            )

        print(
            "\nTargeting fractions:"
        )

        print(
            [
                f"{x:.0%}"
                for x in target_fractions
            ]
        )

        # ====================================================
        # EVALUATE POLICIES
        # ====================================================

        results = (
            self.evaluate_grid(
                df=df,
                uplift_col=uplift_col,
                target_fractions=(
                    target_fractions
                ),
            )
        )

        self.print_grid(
            results=results
        )

        # ====================================================
        # OPTIMIZE
        # ====================================================

        (
            best_policy,
            objective_col,
        ) = (
            self.select_best_policy(
                results=results,
                objective=objective,

                minimum_mean_uplift=(
                    minimum_mean_uplift
                ),

                maximum_target_fraction=(
                    maximum_target_fraction
                ),
            )
        )

        self.print_recommendation(
            best_policy=best_policy,
            objective=objective,
        )

        # ====================================================
        # PLOTS
        # ====================================================

        incremental_plot = (
            self.plot_incremental_outcomes(
                results=results,
                best_policy=best_policy,
            )
        )

        efficiency_plot = (
            self.plot_efficiency(
                results=results,
                best_policy=best_policy,
            )
        )

        business_plot = (
            self.plot_business_objective(
                results=results,
                best_policy=best_policy,
                objective_col=objective_col,
            )
        )

        # ====================================================
        # SAVE
        # ====================================================

        paths = (
            self.save_results(
                results=results,
                best_policy=best_policy,
            )
        )

        print("\n" + "=" * 70)
        print("POLICY OPTIMIZATION COMPLETE")
        print("=" * 70)

        return {
            "grid":
                results,

            "recommended_policy":
                best_policy,

            "objective":
                objective,

            "objective_column":
                objective_col,

            "incremental_plot":
                incremental_plot,

            "efficiency_plot":
                efficiency_plot,

            "business_plot":
                business_plot,

            **paths,
        }