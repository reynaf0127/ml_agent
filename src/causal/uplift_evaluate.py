import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UpliftEvaluator:
    """
    Evaluate uplift / CATE model quality.

    Main responsibilities:
    - Validate uplift scoring data
    - Rank observations by predicted uplift
    - Estimate treatment effects by uplift bucket
    - Support propensity-weighted treatment-effect estimates
    - Calculate cumulative uplift / gain
    - Calculate AUUC
    - Calculate Qini curve and Qini coefficient
    - Evaluate top-K targeting policies
    - Compare predicted uplift with estimated observed uplift
    - Save evaluation tables and plots

    Important
    ---------
    Individual treatment effects are not directly observable.

    Therefore, uplift models cannot be evaluated using ordinary
    prediction metrics such as RMSE between predicted uplift and
    "actual uplift".

    Instead, we evaluate whether observations ranked as having
    high predicted uplift actually show stronger incremental
    treatment effects.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/uplift_evaluation",
        random_state=42,
    ):

        self.output_dir = output_dir
        self.random_state = random_state

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
        treatment_col,
        outcome_col,
        uplift_col,
        propensity_col=None,
        weight_col=None,
    ):

        required = [
            treatment_col,
            outcome_col,
            uplift_col,
        ]

        if propensity_col is not None:
            required.append(
                propensity_col
            )

        if weight_col is not None:
            required.append(
                weight_col
            )

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required uplift evaluation "
                f"columns: {missing}"
            )

        treatment_values = set(
            df[treatment_col]
            .dropna()
            .unique()
        )

        if not treatment_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "Treatment must be binary 0/1. "
                f"Found: {treatment_values}"
            )

        outcome_values = set(
            df[outcome_col]
            .dropna()
            .unique()
        )

        if not outcome_values.issubset(
            {0, 1}
        ):

            print(
                "\n[UPLIFT EVALUATION] Warning: "
                "outcome is not binary."
            )

            print(
                "Some uplift metrics are primarily "
                "designed for binary outcomes."
            )

        if (
            df[uplift_col]
            .isna()
            .any()
        ):

            raise ValueError(
                "Predicted uplift contains "
                "missing values."
            )

    # ========================================================
    # WEIGHTED MEAN
    # ========================================================

    @staticmethod
    def _weighted_mean(
        values,
        weights,
    ):

        values = np.asarray(
            values,
            dtype=float,
        )

        weights = np.asarray(
            weights,
            dtype=float,
        )

        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
        )

        values = values[
            valid
        ]

        weights = weights[
            valid
        ]

        if len(values) == 0:
            return np.nan

        denominator = np.sum(
            weights
        )

        if denominator <= 0:
            return np.nan

        return (
            np.sum(
                values
                * weights
            )
            / denominator
        )

    # ========================================================
    # EFFECT ESTIMATION
    # ========================================================

    def estimate_effect(
        self,
        df,
        treatment_col,
        outcome_col,
        weight_col=None,
    ):

        """
        Estimate treatment-control outcome difference.

        If weight_col is supplied, calculate weighted means.

        Otherwise use raw observed means.
        """

        treated = (
            df[treatment_col]
            == 1
        )

        control = (
            df[treatment_col]
            == 0
        )

        n_treated = int(
            treated.sum()
        )

        n_control = int(
            control.sum()
        )

        if (
            n_treated == 0
            or n_control == 0
        ):

            return {
                "treated_observations":
                    n_treated,

                "control_observations":
                    n_control,

                "treated_outcome_rate":
                    np.nan,

                "control_outcome_rate":
                    np.nan,

                "estimated_uplift":
                    np.nan,
            }

        if weight_col is None:

            treated_rate = (
                df.loc[
                    treated,
                    outcome_col,
                ]
                .astype(float)
                .mean()
            )

            control_rate = (
                df.loc[
                    control,
                    outcome_col,
                ]
                .astype(float)
                .mean()
            )

        else:

            treated_rate = (
                self._weighted_mean(
                    values=df.loc[
                        treated,
                        outcome_col,
                    ],
                    weights=df.loc[
                        treated,
                        weight_col,
                    ],
                )
            )

            control_rate = (
                self._weighted_mean(
                    values=df.loc[
                        control,
                        outcome_col,
                    ],
                    weights=df.loc[
                        control,
                        weight_col,
                    ],
                )
            )

        estimated_uplift = (
            treated_rate
            - control_rate
        )

        return {
            "treated_observations":
                n_treated,

            "control_observations":
                n_control,

            "treated_outcome_rate":
                treated_rate,

            "control_outcome_rate":
                control_rate,

            "estimated_uplift":
                estimated_uplift,
        }

    # ========================================================
    # CREATE RANKING
    # ========================================================

    def create_ranking(
        self,
        df,
        uplift_col,
    ):

        ranked = (
            df
            .sort_values(
                uplift_col,
                ascending=False,
            )
            .reset_index(
                drop=True
            )
            .copy()
        )

        ranked[
            "uplift_rank"
        ] = (
            np.arange(
                1,
                len(ranked) + 1,
            )
        )

        ranked[
            "population_fraction"
        ] = (
            ranked[
                "uplift_rank"
            ]
            / len(ranked)
        )

        return ranked

    # ========================================================
    # DECILE EVALUATION
    # ========================================================

    def evaluate_deciles(
        self,
        df,
        treatment_col,
        outcome_col,
        uplift_col,
        weight_col=None,
        n_bins=10,
    ):

        result = df.copy()

        # ----------------------------------------------------
        # Rank first so decile 1 = highest predicted uplift
        # ----------------------------------------------------

        result = (
            result
            .sort_values(
                uplift_col,
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Robust equal-frequency bins
        # ----------------------------------------------------

        ranks = (
            result[uplift_col]
            .rank(
                method="first",
                ascending=False,
            )
        )

        result[
            "uplift_decile"
        ] = pd.qcut(
            ranks,
            q=n_bins,
            labels=range(
                1,
                n_bins + 1,
            ),
        )

        result[
            "uplift_decile"
        ] = (
            result[
                "uplift_decile"
            ]
            .astype(int)
        )

        rows = []

        for decile in range(
            1,
            n_bins + 1,
        ):

            subset = result[
                result[
                    "uplift_decile"
                ]
                == decile
            ]

            if subset.empty:
                continue

            effect = (
                self.estimate_effect(
                    df=subset,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    weight_col=weight_col,
                )
            )

            rows.append({
                "uplift_decile":
                    decile,

                "observations":
                    len(subset),

                "mean_predicted_uplift":
                    subset[
                        uplift_col
                    ].mean(),

                "median_predicted_uplift":
                    subset[
                        uplift_col
                    ].median(),

                **effect,
            })

        decile_df = pd.DataFrame(
            rows
        )

        return decile_df

    # ========================================================
    # TOP-K TARGETING
    # ========================================================

    def evaluate_top_k(
        self,
        df,
        treatment_col,
        outcome_col,
        uplift_col,
        weight_col=None,
        fractions=None,
    ):

        if fractions is None:

            fractions = [
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.75,
                1.00,
            ]

        ranked = (
            self.create_ranking(
                df=df,
                uplift_col=uplift_col,
            )
        )

        rows = []

        n_total = len(
            ranked
        )

        for fraction in fractions:

            n_target = max(
                1,
                int(
                    np.ceil(
                        n_total
                        * fraction
                    )
                ),
            )

            subset = (
                ranked
                .iloc[
                    :n_target
                ]
            )

            effect = (
                self.estimate_effect(
                    df=subset,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    weight_col=weight_col,
                )
            )

            mean_predicted_uplift = (
                subset[
                    uplift_col
                ]
                .mean()
            )

            estimated_incremental_outcomes = (
                effect[
                    "estimated_uplift"
                ]
                * n_target
            )

            rows.append({
                "target_fraction":
                    fraction,

                "target_pct":
                    fraction * 100,

                "observations":
                    n_target,

                "mean_predicted_uplift":
                    mean_predicted_uplift,

                **effect,

                "estimated_incremental_outcomes":
                    estimated_incremental_outcomes,
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # CUMULATIVE GAIN
    # ========================================================

    def cumulative_gain(
        self,
        df,
        treatment_col,
        outcome_col,
        uplift_col,
        weight_col=None,
        n_points=100,
    ):

        """
        Calculate cumulative estimated treatment gain as
        progressively larger fractions of the population
        are targeted.

        Gain:

            estimated uplift in selected population
            × selected population size

        This makes the curve interpretable in terms of
        estimated incremental outcomes.
        """

        ranked = (
            self.create_ranking(
                df=df,
                uplift_col=uplift_col,
            )
        )

        n_total = len(
            ranked
        )

        fractions = np.linspace(
            0.01,
            1.00,
            n_points,
        )

        rows = []

        for fraction in fractions:

            n_target = max(
                1,
                int(
                    np.ceil(
                        n_total
                        * fraction
                    )
                ),
            )

            subset = (
                ranked
                .iloc[
                    :n_target
                ]
            )

            effect = (
                self.estimate_effect(
                    df=subset,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    weight_col=weight_col,
                )
            )

            estimated_uplift = (
                effect[
                    "estimated_uplift"
                ]
            )

            cumulative_gain = (
                estimated_uplift
                * n_target
                if pd.notna(
                    estimated_uplift
                )
                else np.nan
            )

            rows.append({
                "population_fraction":
                    fraction,

                "population_pct":
                    fraction * 100,

                "observations":
                    n_target,

                "mean_predicted_uplift":
                    subset[
                        uplift_col
                    ].mean(),

                "estimated_uplift":
                    estimated_uplift,

                "cumulative_gain":
                    cumulative_gain,

                "treated_observations":
                    effect[
                        "treated_observations"
                    ],

                "control_observations":
                    effect[
                        "control_observations"
                    ],
            })

        gain_df = pd.DataFrame(
            rows
        )

        return gain_df

    # ========================================================
    # AUUC
    # ========================================================

    def calculate_auuc(
        self,
        gain_df,
    ):

        valid = (
            gain_df[
                [
                    "population_fraction",
                    "cumulative_gain",
                ]
            ]
            .dropna()
        )

        if len(valid) < 2:
            return np.nan

        x = (
            valid[
                "population_fraction"
            ]
            .to_numpy()
        )

        y = (
            valid[
                "cumulative_gain"
            ]
            .to_numpy()
        )

        # Include origin
        x = np.insert(
            x,
            0,
            0.0,
        )

        y = np.insert(
            y,
            0,
            0.0,
        )

        auuc = np.trapezoid(
            y,
            x,
        )

        return float(
            auuc
        )

    # ========================================================
    # QINI CURVE
    # ========================================================

    def calculate_qini(
        self,
        df,
        treatment_col,
        outcome_col,
        uplift_col,
        n_points=100,
    ):

        """
        Calculate a standard ranking-based Qini-style curve.

        For each top-ranked subset:

            incremental responses
            =
            treated responders
            -
            control responders
              * (treated count / control count)

        This is primarily a ranking diagnostic.

        Note:
        For observational treatment assignment, the
        propensity-adjusted top-K / gain results should
        generally receive more causal weight than raw Qini.
        """

        ranked = (
            self.create_ranking(
                df=df,
                uplift_col=uplift_col,
            )
        )

        n_total = len(
            ranked
        )

        fractions = np.linspace(
            0.01,
            1.00,
            n_points,
        )

        rows = []

        for fraction in fractions:

            n_target = max(
                1,
                int(
                    np.ceil(
                        n_total
                        * fraction
                    )
                ),
            )

            subset = (
                ranked
                .iloc[
                    :n_target
                ]
            )

            treated = subset[
                subset[
                    treatment_col
                ] == 1
            ]

            control = subset[
                subset[
                    treatment_col
                ] == 0
            ]

            n_treated = len(
                treated
            )

            n_control = len(
                control
            )

            treated_outcomes = (
                treated[
                    outcome_col
                ]
                .sum()
            )

            control_outcomes = (
                control[
                    outcome_col
                ]
                .sum()
            )

            if (
                n_treated == 0
                or n_control == 0
            ):

                qini_gain = np.nan

            else:

                qini_gain = (
                    treated_outcomes
                    -
                    control_outcomes
                    * (
                        n_treated
                        / n_control
                    )
                )

            rows.append({
                "population_fraction":
                    fraction,

                "population_pct":
                    fraction * 100,

                "observations":
                    n_target,

                "treated_observations":
                    n_treated,

                "control_observations":
                    n_control,

                "treated_outcomes":
                    treated_outcomes,

                "control_outcomes":
                    control_outcomes,

                "qini_gain":
                    qini_gain,
            })

        qini_df = pd.DataFrame(
            rows
        )

        return qini_df

    # ========================================================
    # QINI COEFFICIENT
    # ========================================================

    def calculate_qini_coefficient(
        self,
        qini_df,
    ):

        valid = (
            qini_df[
                [
                    "population_fraction",
                    "qini_gain",
                ]
            ]
            .dropna()
        )

        if len(valid) < 2:
            return {
                "qini_area":
                    np.nan,

                "random_area":
                    np.nan,

                "qini_coefficient":
                    np.nan,
            }

        x = (
            valid[
                "population_fraction"
            ]
            .to_numpy()
        )

        y = (
            valid[
                "qini_gain"
            ]
            .to_numpy()
        )

        x = np.insert(
            x,
            0,
            0.0,
        )

        y = np.insert(
            y,
            0,
            0.0,
        )

        qini_area = (
            np.trapezoid(
                y,
                x,
            )
        )

        final_gain = (
            y[-1]
        )

        random_y = (
            x
            * final_gain
        )

        random_area = (
            np.trapezoid(
                random_y,
                x,
            )
        )

        qini_coefficient = (
            qini_area
            - random_area
        )

        return {
            "qini_area":
                float(
                    qini_area
                ),

            "random_area":
                float(
                    random_area
                ),

            "qini_coefficient":
                float(
                    qini_coefficient
                ),
        }

    # ========================================================
    # PREDICTED VS ESTIMATED UPLIFT
    # ========================================================

    def calibration_table(
        self,
        decile_df,
    ):

        if decile_df.empty:

            return pd.DataFrame()

        calibration = (
            decile_df[
                [
                    "uplift_decile",
                    "observations",
                    "mean_predicted_uplift",
                    "estimated_uplift",
                ]
            ]
            .copy()
        )

        calibration[
            "uplift_error"
        ] = (
            calibration[
                "mean_predicted_uplift"
            ]
            -
            calibration[
                "estimated_uplift"
            ]
        )

        calibration[
            "absolute_uplift_error"
        ] = (
            calibration[
                "uplift_error"
            ]
            .abs()
        )

        return calibration

    # ========================================================
    # PLOT DECILE UPLIFT
    # ========================================================

    def plot_deciles(
        self,
        decile_df,
    ):

        if decile_df.empty:
            return None

        plot_df = (
            decile_df
            .sort_values(
                "uplift_decile"
            )
        )

        x = np.arange(
            len(plot_df)
        )

        width = 0.38

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            x - width / 2,
            plot_df[
                "mean_predicted_uplift"
            ],
            width=width,
            label="Predicted uplift",
        )

        plt.bar(
            x + width / 2,
            plot_df[
                "estimated_uplift"
            ],
            width=width,
            label="Estimated uplift",
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        plt.xticks(
            x,
            plot_df[
                "uplift_decile"
            ],
        )

        plt.xlabel(
            "Uplift Decile "
            "(1 = Highest Predicted Uplift)"
        )

        plt.ylabel(
            "Treatment Effect"
        )

        plt.title(
            "Predicted vs Estimated Uplift by Decile"
        )

        plt.legend()

        plt.grid(
            axis="y",
            alpha=0.2,
        )

        path = os.path.join(
            self.output_dir,
            "uplift_decile_evaluation.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[UPLIFT EVALUATION] "
            "Saved decile plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # PLOT CUMULATIVE GAIN
    # ========================================================

    def plot_cumulative_gain(
        self,
        gain_df,
    ):

        if gain_df.empty:
            return None

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            gain_df[
                "population_pct"
            ],
            gain_df[
                "cumulative_gain"
            ],
            linewidth=2,
            label="Uplift model",
        )

        # ----------------------------------------------------
        # Random targeting baseline
        # ----------------------------------------------------

        final_gain = (
            gain_df[
                "cumulative_gain"
            ]
            .dropna()
        )

        if len(final_gain) > 0:

            final_gain = (
                final_gain.iloc[-1]
            )

            random_gain = (
                gain_df[
                    "population_fraction"
                ]
                * final_gain
            )

            plt.plot(
                gain_df[
                    "population_pct"
                ],
                random_gain,
                linestyle="--",
                linewidth=1.5,
                label="Random targeting",
            )

        plt.xlabel(
            "Population Targeted (%)"
        )

        plt.ylabel(
            "Estimated Incremental Outcomes"
        )

        plt.title(
            "Cumulative Uplift Gain"
        )

        plt.legend()

        plt.grid(
            alpha=0.2,
        )

        path = os.path.join(
            self.output_dir,
            "cumulative_uplift_gain.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[UPLIFT EVALUATION] "
            "Saved cumulative gain plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # PLOT QINI
    # ========================================================

    def plot_qini(
        self,
        qini_df,
    ):

        if qini_df.empty:
            return None

        valid = (
            qini_df
            .dropna(
                subset=[
                    "qini_gain"
                ]
            )
        )

        if valid.empty:
            return None

        final_gain = (
            valid[
                "qini_gain"
            ]
            .iloc[-1]
        )

        random_gain = (
            valid[
                "population_fraction"
            ]
            * final_gain
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            valid[
                "population_pct"
            ],
            valid[
                "qini_gain"
            ],
            linewidth=2,
            label="Uplift model",
        )

        plt.plot(
            valid[
                "population_pct"
            ],
            random_gain,
            linestyle="--",
            linewidth=1.5,
            label="Random targeting",
        )

        plt.xlabel(
            "Population Targeted (%)"
        )

        plt.ylabel(
            "Incremental Responses"
        )

        plt.title(
            "Qini Curve"
        )

        plt.legend()

        plt.grid(
            alpha=0.2,
        )

        path = os.path.join(
            self.output_dir,
            "qini_curve.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[UPLIFT EVALUATION] "
            "Saved Qini plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # PLOT TOP-K POLICY
    # ========================================================

    def plot_top_k(
        self,
        top_k_df,
    ):

        if top_k_df.empty:
            return None

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            top_k_df[
                "target_pct"
            ],
            top_k_df[
                "estimated_uplift"
            ],
            marker="o",
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        plt.xlabel(
            "Top Population Targeted (%)"
        )

        plt.ylabel(
            "Estimated Treatment Effect"
        )

        plt.title(
            "Treatment Effect by Targeting Depth"
        )

        plt.grid(
            alpha=0.2,
        )

        path = os.path.join(
            self.output_dir,
            "top_k_targeting_effect.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[UPLIFT EVALUATION] "
            "Saved Top-K plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # SUMMARY
    # ========================================================

    def build_summary(
        self,
        df,
        uplift_col,
        decile_df,
        top_k_df,
        auuc,
        qini_metrics,
    ):

        overall_mean = (
            df[
                uplift_col
            ]
            .mean()
        )

        positive_pct = (
            (
                df[
                    uplift_col
                ]
                > 0
            )
            .mean()
        )

        negative_pct = (
            (
                df[
                    uplift_col
                ]
                < 0
            )
            .mean()
        )

        top_decile_effect = np.nan
        top_decile_predicted = np.nan

        if not decile_df.empty:

            top_row = (
                decile_df
                .sort_values(
                    "uplift_decile"
                )
                .iloc[0]
            )

            top_decile_effect = (
                top_row[
                    "estimated_uplift"
                ]
            )

            top_decile_predicted = (
                top_row[
                    "mean_predicted_uplift"
                ]
            )

        top_10_effect = np.nan

        if not top_k_df.empty:

            nearest = (
                top_k_df.iloc[
                    (
                        top_k_df[
                            "target_fraction"
                        ]
                        - 0.10
                    )
                    .abs()
                    .argsort()[
                        :1
                    ]
                ]
            )

            if len(nearest) > 0:

                top_10_effect = (
                    nearest.iloc[0][
                        "estimated_uplift"
                    ]
                )

        summary = pd.DataFrame([
            {
                "observations":
                    len(df),

                "mean_predicted_uplift":
                    overall_mean,

                "positive_uplift_pct":
                    positive_pct,

                "negative_uplift_pct":
                    negative_pct,

                "top_decile_predicted_uplift":
                    top_decile_predicted,

                "top_decile_estimated_uplift":
                    top_decile_effect,

                "top_10_pct_estimated_uplift":
                    top_10_effect,

                "auuc":
                    auuc,

                "qini_area":
                    qini_metrics[
                        "qini_area"
                    ],

                "qini_random_area":
                    qini_metrics[
                        "random_area"
                    ],

                "qini_coefficient":
                    qini_metrics[
                        "qini_coefficient"
                    ],
            }
        ])

        return summary

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        df,
        treatment_col,
        outcome_col,
        uplift_col="predicted_uplift",
        propensity_col=None,
        weight_col=None,
        n_bins=10,
    ):

        print("\n" + "=" * 70)
        print("UPLIFT MODEL EVALUATION")
        print("=" * 70)

        print(
            f"\nRows: {len(df):,}"
        )

        print(
            f"Treatment: "
            f"{treatment_col}"
        )

        print(
            f"Outcome: "
            f"{outcome_col}"
        )

        print(
            f"Uplift prediction: "
            f"{uplift_col}"
        )

        if weight_col is not None:

            print(
                f"Evaluation weight: "
                f"{weight_col}"
            )

        else:

            print(
                "Evaluation weight: None"
            )

        # ====================================================
        # VALIDATE
        # ====================================================

        self._validate(
            df=df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
            propensity_col=propensity_col,
            weight_col=weight_col,
        )

        # ====================================================
        # DECILES
        # ====================================================

        print("\n" + "=" * 70)
        print("UPLIFT DECILE EVALUATION")
        print("=" * 70)

        decile_df = (
            self.evaluate_deciles(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                uplift_col=uplift_col,
                weight_col=weight_col,
                n_bins=n_bins,
            )
        )

        print(
            "\n"
            + decile_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # CALIBRATION
        # ====================================================

        calibration_df = (
            self.calibration_table(
                decile_df=decile_df,
            )
        )

        print("\n" + "=" * 70)
        print("UPLIFT CALIBRATION")
        print("=" * 70)

        print(
            "\n"
            + calibration_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # TOP-K
        # ====================================================

        print("\n" + "=" * 70)
        print("TOP-K TARGETING EVALUATION")
        print("=" * 70)

        top_k_df = (
            self.evaluate_top_k(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                uplift_col=uplift_col,
                weight_col=weight_col,
            )
        )

        print(
            "\n"
            + top_k_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # CUMULATIVE GAIN
        # ====================================================

        gain_df = (
            self.cumulative_gain(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                uplift_col=uplift_col,
                weight_col=weight_col,
            )
        )

        auuc = (
            self.calculate_auuc(
                gain_df=gain_df,
            )
        )

        # ====================================================
        # QINI
        # ====================================================

        qini_df = (
            self.calculate_qini(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                uplift_col=uplift_col,
            )
        )

        qini_metrics = (
            self.calculate_qini_coefficient(
                qini_df=qini_df,
            )
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary_df = (
            self.build_summary(
                df=df,
                uplift_col=uplift_col,
                decile_df=decile_df,
                top_k_df=top_k_df,
                auuc=auuc,
                qini_metrics=qini_metrics,
            )
        )

        print("\n" + "=" * 70)
        print("UPLIFT EVALUATION SUMMARY")
        print("=" * 70)

        print(
            "\n"
            + summary_df
            .round(6)
            .to_string(
                index=False
            )
        )

        print(
            f"\nAUUC: "
            f"{auuc:.4f}"
        )

        print(
            "Qini coefficient: "
            f"{qini_metrics['qini_coefficient']:.4f}"
        )

        # ====================================================
        # SAVE TABLES
        # ====================================================

        decile_path = os.path.join(
            self.output_dir,
            "uplift_evaluation_deciles.csv",
        )

        calibration_path = os.path.join(
            self.output_dir,
            "uplift_calibration.csv",
        )

        top_k_path = os.path.join(
            self.output_dir,
            "uplift_top_k.csv",
        )

        gain_path = os.path.join(
            self.output_dir,
            "uplift_cumulative_gain.csv",
        )

        qini_path = os.path.join(
            self.output_dir,
            "uplift_qini.csv",
        )

        summary_path = os.path.join(
            self.output_dir,
            "uplift_evaluation_summary.csv",
        )

        decile_df.to_csv(
            decile_path,
            index=False,
        )

        calibration_df.to_csv(
            calibration_path,
            index=False,
        )

        top_k_df.to_csv(
            top_k_path,
            index=False,
        )

        gain_df.to_csv(
            gain_path,
            index=False,
        )

        qini_df.to_csv(
            qini_path,
            index=False,
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        # ====================================================
        # PLOTS
        # ====================================================

        decile_plot = (
            self.plot_deciles(
                decile_df
            )
        )

        gain_plot = (
            self.plot_cumulative_gain(
                gain_df
            )
        )

        qini_plot = (
            self.plot_qini(
                qini_df
            )
        )

        top_k_plot = (
            self.plot_top_k(
                top_k_df
            )
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        print("\n" + "=" * 70)
        print("UPLIFT EVALUATION COMPLETE")
        print("=" * 70)

        print(
            "\nKey question:"
        )

        print(
            "Does targeting observations with higher "
            "predicted uplift produce larger estimated "
            "incremental effects?"
        )

        print(
            "\nUse the decile table, cumulative gain, "
            "Top-K evaluation, AUUC, and Qini curve "
            "to answer this."
        )

        return {
            "summary":
                summary_df,

            "deciles":
                decile_df,

            "calibration":
                calibration_df,

            "top_k":
                top_k_df,

            "cumulative_gain":
                gain_df,

            "qini":
                qini_df,

            "auuc":
                auuc,

            "qini_metrics":
                qini_metrics,

            "paths": {
                "deciles":
                    decile_path,

                "calibration":
                    calibration_path,

                "top_k":
                    top_k_path,

                "cumulative_gain":
                    gain_path,

                "qini":
                    qini_path,

                "summary":
                    summary_path,

                "decile_plot":
                    decile_plot,

                "gain_plot":
                    gain_plot,

                "qini_plot":
                    qini_plot,

                "top_k_plot":
                    top_k_plot,
            },
        }