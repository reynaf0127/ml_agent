# src/causal/policy.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CausalPolicy:
    """
    Convert estimated uplift / CATE into an actionable
    treatment or targeting policy.

    Core idea
    ---------
    For observation i:

        uplift_i
            = estimated incremental probability of outcome

        incremental_value_i
            = uplift_i * value_of_outcome_i

        expected_net_value_i
            = incremental_value_i - treatment_cost_i

    A treatment can then be recommended when:

        expected_net_value_i > 0

    or according to another user-selected policy such as:

        - positive uplift
        - uplift threshold
        - top-K targeting
        - positive expected net value

    This module intentionally does NOT fit a causal model.
    It consumes causal estimates produced by the uplift model.

    Expected input
    --------------
    A DataFrame containing at minimum:

        uplift_score

    Optional columns:

        predicted_outcome_treated
        predicted_outcome_control
        propensity_score
        observed_treatment
        observed_outcome
        causal_group

    Business value and treatment cost can be supplied as:

        1. constants, or
        2. existing DataFrame columns

    No dataset-specific column names are hardcoded.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/policy",
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
                "Uplift column not found: "
                f"{uplift_col}"
            )

        uplift = pd.to_numeric(
            df[uplift_col],
            errors="coerce",
        )

        if uplift.isna().any():

            missing = int(
                uplift
                .isna()
                .sum()
            )

            raise ValueError(
                f"{uplift_col} contains "
                f"{missing:,} missing or "
                "non-numeric values."
            )

        if len(df) == 0:

            raise ValueError(
                "Policy input DataFrame is empty."
            )

    # ========================================================
    # RESOLVE VALUE INPUT
    # ========================================================

    def _resolve_value(
        self,
        df,
        constant=None,
        column=None,
        default=None,
        label="value",
    ):
        """
        Resolve a business quantity from either:

            - a DataFrame column
            - a scalar constant
            - a default value

        Column takes precedence over constant.
        """

        if column is not None:

            if column not in df.columns:

                raise ValueError(
                    f"{label} column not found: "
                    f"{column}"
                )

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if values.isna().any():

                missing = int(
                    values
                    .isna()
                    .sum()
                )

                raise ValueError(
                    f"{label} column "
                    f"'{column}' contains "
                    f"{missing:,} missing or "
                    "non-numeric values."
                )

            return values.astype(float)

        if constant is not None:

            try:

                constant = float(
                    constant
                )

            except (
                TypeError,
                ValueError,
            ):

                raise ValueError(
                    f"{label} constant must "
                    "be numeric."
                )

            return pd.Series(
                constant,
                index=df.index,
                dtype=float,
            )

        if default is not None:

            return pd.Series(
                float(default),
                index=df.index,
                dtype=float,
            )

        return None

    # ========================================================
    # UPLIFT RANK
    # ========================================================

    def add_uplift_rank(
        self,
        df,
        uplift_col,
    ):

        result = df.copy()

        result[
            "uplift_rank"
        ] = (
            result[
                uplift_col
            ]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(int)
        )

        result[
            "uplift_percentile"
        ] = (
            result[
                "uplift_rank"
            ]
            / len(result)
        )

        return result

    # ========================================================
    # UPLIFT DECILES
    # ========================================================

    def add_uplift_deciles(
        self,
        df,
        uplift_col,
        n_bins=10,
    ):

        result = (
            df.copy()
            .sort_values(
                uplift_col,
                ascending=False,
            )
        )

        n_bins = min(
            int(n_bins),
            len(result),
        )

        if n_bins < 1:

            raise ValueError(
                "n_bins must be >= 1."
            )

        result[
            "uplift_decile"
        ] = (
            pd.qcut(
                np.arange(
                    len(result)
                ),
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

        result[
            "uplift_decile"
        ] = (
            result[
                "uplift_decile"
            ]
            .astype(int)
        )

        return (
            result
            .sort_index()
        )

    # ========================================================
    # BUSINESS VALUE
    # ========================================================

    def calculate_business_value(
        self,
        df,
        uplift_col,
        outcome_value_constant=None,
        outcome_value_col=None,
        treatment_cost_constant=None,
        treatment_cost_col=None,
    ):
        """
        Calculate expected incremental business value.

        incremental value:

            uplift * outcome value

        expected net value:

            incremental value - treatment cost

        Outcome value and treatment cost can each be
        observation-specific or constant.
        """

        result = df.copy()

        outcome_value = (
            self._resolve_value(
                df=result,
                constant=(
                    outcome_value_constant
                ),
                column=(
                    outcome_value_col
                ),
                default=None,
                label="Outcome value",
            )
        )

        treatment_cost = (
            self._resolve_value(
                df=result,
                constant=(
                    treatment_cost_constant
                ),
                column=(
                    treatment_cost_col
                ),
                default=None,
                label="Treatment cost",
            )
        )

        # ----------------------------------------------------
        # If no business value supplied, leave business
        # columns unavailable rather than invent assumptions.
        # ----------------------------------------------------

        if outcome_value is None:

            return result

        result[
            "outcome_value"
        ] = outcome_value

        result[
            "expected_incremental_value"
        ] = (
            result[
                uplift_col
            ].astype(float)
            * result[
                "outcome_value"
            ]
        )

        if treatment_cost is not None:

            result[
                "treatment_cost"
            ] = treatment_cost

            result[
                "expected_net_value"
            ] = (
                result[
                    "expected_incremental_value"
                ]
                - result[
                    "treatment_cost"
                ]
            )

            # -----------------------------------------------
            # Break-even uplift
            #
            # cost / conversion value
            # -----------------------------------------------

            result[
                "break_even_uplift"
            ] = np.where(
                result[
                    "outcome_value"
                ] > 0,

                result[
                    "treatment_cost"
                ]
                / result[
                    "outcome_value"
                ],

                np.nan,
            )

        return result

    # ========================================================
    # POSITIVE UPLIFT POLICY
    # ========================================================

    def positive_uplift_policy(
        self,
        df,
        uplift_col,
    ):

        return (
            df[
                uplift_col
            ]
            .astype(float)
            > 0
        )

    # ========================================================
    # UPLIFT THRESHOLD POLICY
    # ========================================================

    def uplift_threshold_policy(
        self,
        df,
        uplift_col,
        threshold,
    ):

        threshold = float(
            threshold
        )

        return (
            df[
                uplift_col
            ]
            .astype(float)
            >= threshold
        )

    # ========================================================
    # TOP-K POLICY
    # ========================================================

    def top_k_policy(
        self,
        df,
        uplift_col,
        target_fraction,
    ):

        target_fraction = float(
            target_fraction
        )

        if not (
            0
            < target_fraction
            <= 1
        ):

            raise ValueError(
                "target_fraction must be "
                "between 0 and 1."
            )

        n_target = int(
            np.ceil(
                len(df)
                * target_fraction
            )
        )

        ranked_index = (
            df[
                uplift_col
            ]
            .astype(float)
            .sort_values(
                ascending=False
            )
            .head(
                n_target
            )
            .index
        )

        recommendation = pd.Series(
            False,
            index=df.index,
        )

        recommendation.loc[
            ranked_index
        ] = True

        return recommendation

    # ========================================================
    # NET VALUE POLICY
    # ========================================================

    def net_value_policy(
        self,
        df,
    ):

        if (
            "expected_net_value"
            not in df.columns
        ):

            raise ValueError(
                "Net-value policy requires "
                "expected_net_value. "
                "Provide both outcome value "
                "and treatment cost."
            )

        return (
            df[
                "expected_net_value"
            ]
            > 0
        )

    # ========================================================
    # APPLY POLICY
    # ========================================================

    def apply_policy(
        self,
        df,
        uplift_col,
        policy="positive_uplift",
        uplift_threshold=0.0,
        target_fraction=0.10,
    ):

        result = df.copy()

        policy = (
            str(
                policy
            )
            .strip()
            .lower()
        )

        if policy == "positive_uplift":

            recommendation = (
                self.positive_uplift_policy(
                    df=result,
                    uplift_col=uplift_col,
                )
            )

        elif policy == "uplift_threshold":

            recommendation = (
                self.uplift_threshold_policy(
                    df=result,
                    uplift_col=uplift_col,
                    threshold=(
                        uplift_threshold
                    ),
                )
            )

        elif policy == "top_k":

            recommendation = (
                self.top_k_policy(
                    df=result,
                    uplift_col=uplift_col,
                    target_fraction=(
                        target_fraction
                    ),
                )
            )

        elif policy == "net_value":

            recommendation = (
                self.net_value_policy(
                    df=result
                )
            )

        else:

            raise ValueError(
                "Unknown causal policy: "
                f"{policy}. "
                "Supported policies are: "
                "positive_uplift, "
                "uplift_threshold, "
                "top_k, "
                "net_value."
            )

        result[
            "recommended_treatment"
        ] = (
            recommendation
            .astype(int)
        )

        result[
            "policy_name"
        ] = policy

        return result

    # ========================================================
    # POLICY SUMMARY
    # ========================================================

    def summarize_policy(
        self,
        df,
        uplift_col,
    ):

        recommended = (
            df[
                "recommended_treatment"
            ]
            == 1
        )

        not_recommended = (
            ~recommended
        )

        n_total = len(
            df
        )

        n_recommended = int(
            recommended.sum()
        )

        summary = {
            "observations":
                n_total,

            "recommended_observations":
                n_recommended,

            "recommended_pct":
                (
                    n_recommended
                    / n_total
                    if n_total
                    else np.nan
                ),

            "mean_uplift_all":
                df[
                    uplift_col
                ].mean(),

            "mean_uplift_recommended":
                (
                    df.loc[
                        recommended,
                        uplift_col,
                    ]
                    .mean()
                    if n_recommended
                    else np.nan
                ),

            "mean_uplift_not_recommended":
                (
                    df.loc[
                        not_recommended,
                        uplift_col,
                    ]
                    .mean()
                    if not_recommended.any()
                    else np.nan
                ),
        }

        # ----------------------------------------------------
        # Expected incremental outcomes
        # ----------------------------------------------------

        summary[
            "expected_incremental_outcomes"
        ] = (
            df.loc[
                recommended,
                uplift_col,
            ]
            .clip(
                lower=0
            )
            .sum()
        )

        # ----------------------------------------------------
        # Business value
        # ----------------------------------------------------

        if (
            "expected_incremental_value"
            in df.columns
        ):

            summary[
                "expected_incremental_value"
            ] = (
                df.loc[
                    recommended,
                    "expected_incremental_value",
                ]
                .sum()
            )

        if (
            "treatment_cost"
            in df.columns
        ):

            summary[
                "expected_treatment_cost"
            ] = (
                df.loc[
                    recommended,
                    "treatment_cost",
                ]
                .sum()
            )

        if (
            "expected_net_value"
            in df.columns
        ):

            summary[
                "expected_net_value"
            ] = (
                df.loc[
                    recommended,
                    "expected_net_value",
                ]
                .sum()
            )

        return pd.DataFrame([
            summary
        ])

    # ========================================================
    # DECILE POLICY SUMMARY
    # ========================================================

    def summarize_by_decile(
        self,
        df,
        uplift_col,
    ):

        if (
            "uplift_decile"
            not in df.columns
        ):

            return pd.DataFrame()

        rows = []

        for decile, subset in (
            df
            .groupby(
                "uplift_decile"
            )
        ):

            recommended = (
                subset[
                    "recommended_treatment"
                ]
                == 1
            )

            row = {
                "uplift_decile":
                    int(
                        decile
                    ),

                "observations":
                    len(
                        subset
                    ),

                "mean_uplift":
                    subset[
                        uplift_col
                    ]
                    .mean(),

                "recommended_observations":
                    int(
                        recommended.sum()
                    ),

                "recommended_pct":
                    recommended.mean(),

                "expected_incremental_outcomes":
                    (
                        subset.loc[
                            recommended,
                            uplift_col,
                        ]
                        .clip(
                            lower=0
                        )
                        .sum()
                    ),
            }

            if (
                "expected_incremental_value"
                in subset.columns
            ):

                row[
                    "expected_incremental_value"
                ] = (
                    subset.loc[
                        recommended,
                        "expected_incremental_value",
                    ]
                    .sum()
                )

            if (
                "treatment_cost"
                in subset.columns
            ):

                row[
                    "expected_treatment_cost"
                ] = (
                    subset.loc[
                        recommended,
                        "treatment_cost",
                    ]
                    .sum()
                )

            if (
                "expected_net_value"
                in subset.columns
            ):

                row[
                    "expected_net_value"
                ] = (
                    subset.loc[
                        recommended,
                        "expected_net_value",
                    ]
                    .sum()
                )

            rows.append(
                row
            )

        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "uplift_decile"
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # POLICY PLOT
    # ========================================================

    def plot_policy(
        self,
        df,
        uplift_col,
    ):

        plot_df = (
            df
            .sort_values(
                uplift_col,
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        plot_df[
            "population_fraction"
        ] = (
            np.arange(
                1,
                len(plot_df) + 1,
            )
            / len(plot_df)
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            plot_df[
                "population_fraction"
            ],
            plot_df[
                uplift_col
            ],
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1.2,
        )

        plt.xlabel(
            "Population Fraction "
            "(Highest Uplift First)"
        )

        plt.ylabel(
            "Predicted Uplift"
        )

        plt.title(
            "Causal Targeting Policy"
        )

        plt.grid(
            alpha=0.2
        )

        path = os.path.join(
            self.output_dir,
            "causal_policy_curve.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[POLICY] Saved policy plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # SAVE
    # ========================================================

    def save_results(
        self,
        policy_df,
        summary_df,
        decile_df,
    ):

        policy_path = os.path.join(
            self.output_dir,
            "causal_policy_scores.csv",
        )

        summary_path = os.path.join(
            self.output_dir,
            "causal_policy_summary.csv",
        )

        decile_path = os.path.join(
            self.output_dir,
            "causal_policy_deciles.csv",
        )

        policy_df.to_csv(
            policy_path,
            index=False,
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        decile_df.to_csv(
            decile_path,
            index=False,
        )

        print(
            "\n[POLICY] Saved policy scores:"
            f"\n{policy_path}"
        )

        print(
            "[POLICY] Saved policy summary:"
            f"\n{summary_path}"
        )

        print(
            "[POLICY] Saved policy deciles:"
            f"\n{decile_path}"
        )

        return {
            "policy_path":
                policy_path,

            "summary_path":
                summary_path,

            "decile_path":
                decile_path,
        }

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        df,
        uplift_col="uplift_score",
        policy="positive_uplift",
        uplift_threshold=0.0,
        target_fraction=0.10,
        outcome_value_constant=None,
        outcome_value_col=None,
        treatment_cost_constant=None,
        treatment_cost_col=None,
        n_deciles=10,
    ):

        print("\n" + "=" * 70)
        print("CAUSAL TARGETING POLICY")
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
            f"Policy: "
            f"{policy}"
        )

        # ====================================================
        # RANK
        # ====================================================

        result = (
            self.add_uplift_rank(
                df=df,
                uplift_col=uplift_col,
            )
        )

        result = (
            self.add_uplift_deciles(
                df=result,
                uplift_col=uplift_col,
                n_bins=n_deciles,
            )
        )

        # ====================================================
        # BUSINESS VALUE
        # ====================================================

        result = (
            self.calculate_business_value(
                df=result,
                uplift_col=uplift_col,

                outcome_value_constant=(
                    outcome_value_constant
                ),

                outcome_value_col=(
                    outcome_value_col
                ),

                treatment_cost_constant=(
                    treatment_cost_constant
                ),

                treatment_cost_col=(
                    treatment_cost_col
                ),
            )
        )

        # ====================================================
        # POLICY
        # ====================================================

        result = (
            self.apply_policy(
                df=result,
                uplift_col=uplift_col,
                policy=policy,

                uplift_threshold=(
                    uplift_threshold
                ),

                target_fraction=(
                    target_fraction
                ),
            )
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary_df = (
            self.summarize_policy(
                df=result,
                uplift_col=uplift_col,
            )
        )

        decile_df = (
            self.summarize_by_decile(
                df=result,
                uplift_col=uplift_col,
            )
        )

        print("\n" + "=" * 70)
        print("CAUSAL POLICY SUMMARY")
        print("=" * 70)

        print(
            "\n"
            + summary_df
            .round(6)
            .to_string(
                index=False
            )
        )

        print("\n" + "=" * 70)
        print("CAUSAL POLICY BY UPLIFT DECILE")
        print("=" * 70)

        print(
            "\n"
            + decile_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # PLOT
        # ====================================================

        plot_path = (
            self.plot_policy(
                df=result,
                uplift_col=uplift_col,
            )
        )

        # ====================================================
        # SAVE
        # ====================================================

        paths = (
            self.save_results(
                policy_df=result,
                summary_df=summary_df,
                decile_df=decile_df,
            )
        )

        print("\n" + "=" * 70)
        print("CAUSAL POLICY COMPLETE")
        print("=" * 70)

        print(
            "\nThe policy converts estimated "
            "incremental treatment effects into "
            "actionable treatment decisions."
        )

        return {
            "data":
                result,

            "summary":
                summary_df,

            "deciles":
                decile_df,

            "plot_path":
                plot_path,

            **paths,
        }