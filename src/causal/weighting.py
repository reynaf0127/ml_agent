import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PropensityWeighting:
    """
    Propensity-score weighting for causal inference.

    Supports:
    - ATE IPTW weights
    - Stabilized ATE weights
    - ATT weights
    - Weight clipping
    - Effective sample size diagnostics
    - Weighted treatment effect estimation

    Expected input DataFrame must contain:
    - treatment column
    - outcome column
    - propensity score column

    Main output:
    - Original data plus propensity weights
    - Weight diagnostics
    - Weighted treatment-effect estimate
    - Name of the final weight column
    """

    def __init__(
        self,
        output_dir="artifacts/causal/weighting",
        propensity_col="propensity_score",
        weight_col="propensity_weight",
        raw_weight_col="propensity_weight_raw",
    ):

        self.output_dir = output_dir

        self.propensity_col = (
            propensity_col
        )

        self.weight_col = (
            weight_col
        )

        self.raw_weight_col = (
            raw_weight_col
        )

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
        outcome_col=None,
    ):

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                "PropensityWeighting expects df "
                "to be a pandas DataFrame. "
                f"Received: {type(df).__name__}"
            )

        required = [
            treatment_col,
            self.propensity_col,
        ]

        if outcome_col is not None:

            required.append(
                outcome_col
            )

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required columns for "
                "propensity weighting: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Treatment validation
        # ----------------------------------------------------

        treatment_values = set(
            pd.to_numeric(
                df[treatment_col],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
        )

        if not treatment_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "Treatment must be binary 0/1. "
                f"Found: {treatment_values}"
            )

        if (
            df[treatment_col]
            .isna()
            .any()
        ):

            raise ValueError(
                "Treatment column contains "
                "missing values."
            )

        # ----------------------------------------------------
        # Outcome validation
        # ----------------------------------------------------

        if outcome_col is not None:

            outcome = pd.to_numeric(
                df[outcome_col],
                errors="coerce",
            )

            if outcome.isna().any():

                raise ValueError(
                    "Outcome contains missing or "
                    "non-numeric values."
                )

        # ----------------------------------------------------
        # Propensity validation
        # ----------------------------------------------------

        propensity = pd.to_numeric(
            df[self.propensity_col],
            errors="coerce",
        )

        if propensity.isna().any():

            raise ValueError(
                "Propensity score contains "
                "missing/non-numeric values."
            )

        if not np.isfinite(
            propensity.to_numpy(
                dtype=float
            )
        ).all():

            raise ValueError(
                "Propensity score contains "
                "infinite values."
            )

        if (
            (propensity <= 0).any()
            or
            (propensity >= 1).any()
        ):

            print(
                "\n[WEIGHTING] Warning: "
                "propensity scores at or "
                "beyond 0/1 detected."
            )

            print(
                "[WEIGHTING] Scores will be "
                "safely clipped before weight "
                "calculation."
            )

    # ========================================================
    # PROPENSITY SAFETY CLIP
    # ========================================================

    def _safe_propensity(
        self,
        propensity,
        epsilon=1e-6,
    ):

        propensity = np.asarray(
            propensity,
            dtype=float,
        )

        return np.clip(
            propensity,
            epsilon,
            1.0 - epsilon,
        )

    # ========================================================
    # STANDARD ATE IPTW
    # ========================================================

    def calculate_ate_weights(
        self,
        df,
        treatment_col,
    ):
        """
        Standard inverse probability of treatment weights.

        Treated:
            1 / e(X)

        Control:
            1 / (1 - e(X))
        """

        treatment = (
            df[treatment_col]
            .astype(int)
            .to_numpy()
        )

        propensity = (
            self._safe_propensity(
                df[
                    self.propensity_col
                ]
            )
        )

        weights = np.where(
            treatment == 1,

            1.0
            / propensity,

            1.0
            / (
                1.0
                - propensity
            ),
        )

        return weights

    # ========================================================
    # STABILIZED ATE IPTW
    # ========================================================

    def calculate_stabilized_ate_weights(
        self,
        df,
        treatment_col,
    ):
        """
        Stabilized IPTW for ATE.

        Treated:
            P(T=1) / e(X)

        Control:
            P(T=0) / (1-e(X))

        Stabilization generally reduces variance compared
        with standard IPTW.
        """

        treatment = (
            df[treatment_col]
            .astype(int)
            .to_numpy()
        )

        propensity = (
            self._safe_propensity(
                df[
                    self.propensity_col
                ]
            )
        )

        treatment_rate = float(
            np.mean(
                treatment
            )
        )

        control_rate = (
            1.0
            - treatment_rate
        )

        weights = np.where(
            treatment == 1,

            treatment_rate
            / propensity,

            control_rate
            / (
                1.0
                - propensity
            ),
        )

        return weights

    # ========================================================
    # ATT WEIGHTS
    # ========================================================

    def calculate_att_weights(
        self,
        df,
        treatment_col,
    ):
        """
        ATT weighting.

        Treated:
            1

        Control:
            e(X) / (1-e(X))
        """

        treatment = (
            df[treatment_col]
            .astype(int)
            .to_numpy()
        )

        propensity = (
            self._safe_propensity(
                df[
                    self.propensity_col
                ]
            )
        )

        weights = np.where(
            treatment == 1,

            1.0,

            propensity
            / (
                1.0
                - propensity
            ),
        )

        return weights

    # ========================================================
    # WEIGHT CLIPPING
    # ========================================================

    def clip_weights(
        self,
        weights,
        lower_quantile=0.01,
        upper_quantile=0.99,
    ):
        """
        Winsorize extreme propensity weights.

        Default:
            1st to 99th percentile.
        """

        if not (
            0
            <= lower_quantile
            < upper_quantile
            <= 1
        ):

            raise ValueError(
                "Weight clipping quantiles must "
                "satisfy: "
                "0 <= lower < upper <= 1."
            )

        weights = np.asarray(
            weights,
            dtype=float,
        )

        if not np.isfinite(
            weights
        ).all():

            raise ValueError(
                "Raw propensity weights contain "
                "non-finite values."
            )

        lower = float(
            np.quantile(
                weights,
                lower_quantile,
            )
        )

        upper = float(
            np.quantile(
                weights,
                upper_quantile,
            )
        )

        clipped = np.clip(
            weights,
            lower,
            upper,
        )

        return (
            clipped,
            lower,
            upper,
        )

    # ========================================================
    # EFFECTIVE SAMPLE SIZE
    # ========================================================

    def effective_sample_size(
        self,
        weights,
    ):
        """
        Effective sample size:

            ESS = (sum w)^2 / sum(w^2)
        """

        weights = np.asarray(
            weights,
            dtype=float,
        )

        denominator = np.sum(
            weights ** 2
        )

        if (
            denominator == 0
            or not np.isfinite(
                denominator
            )
        ):

            return 0.0

        numerator = (
            np.sum(weights) ** 2
        )

        return float(
            numerator
            / denominator
        )

    # ========================================================
    # WEIGHT SUMMARY
    # ========================================================

    def summarize_weights(
        self,
        df,
        treatment_col,
        weight_col,
    ):

        rows = []

        for treatment_value in [
            0,
            1,
        ]:

            subset = df.loc[
                df[treatment_col]
                == treatment_value,
                weight_col,
            ]

            if len(subset) == 0:
                continue

            weights = (
                subset
                .astype(float)
                .to_numpy()
            )

            rows.append({
                "treatment":
                    treatment_value,

                "n":
                    len(weights),

                "mean":
                    float(
                        np.mean(
                            weights
                        )
                    ),

                "std":
                    float(
                        np.std(
                            weights
                        )
                    ),

                "min":
                    float(
                        np.min(
                            weights
                        )
                    ),

                "q01":
                    float(
                        np.quantile(
                            weights,
                            0.01,
                        )
                    ),

                "q05":
                    float(
                        np.quantile(
                            weights,
                            0.05,
                        )
                    ),

                "q25":
                    float(
                        np.quantile(
                            weights,
                            0.25,
                        )
                    ),

                "median":
                    float(
                        np.quantile(
                            weights,
                            0.50,
                        )
                    ),

                "q75":
                    float(
                        np.quantile(
                            weights,
                            0.75,
                        )
                    ),

                "q95":
                    float(
                        np.quantile(
                            weights,
                            0.95,
                        )
                    ),

                "q99":
                    float(
                        np.quantile(
                            weights,
                            0.99,
                        )
                    ),

                "max":
                    float(
                        np.max(
                            weights
                        )
                    ),

                "ess":
                    self.effective_sample_size(
                        weights
                    ),
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # WEIGHTED OUTCOME MEAN
    # ========================================================

    def _weighted_mean(
        self,
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

        if denominator == 0:

            return np.nan

        return float(
            np.sum(
                values
                * weights
            )
            / denominator
        )

    # ========================================================
    # WEIGHTED TREATMENT EFFECT
    # ========================================================

    def estimate_effect(
        self,
        df,
        treatment_col,
        outcome_col,
        weight_col,
    ):

        treated = (
            df[treatment_col]
            == 1
        )

        control = (
            df[treatment_col]
            == 0
        )

        if treated.sum() == 0:

            raise ValueError(
                "No treated observations are "
                "available for effect estimation."
            )

        if control.sum() == 0:

            raise ValueError(
                "No control observations are "
                "available for effect estimation."
            )

        treated_rate = (
            self._weighted_mean(
                df.loc[
                    treated,
                    outcome_col,
                ],

                df.loc[
                    treated,
                    weight_col,
                ],
            )
        )

        control_rate = (
            self._weighted_mean(
                df.loc[
                    control,
                    outcome_col,
                ],

                df.loc[
                    control,
                    weight_col,
                ],
            )
        )

        effect = (
            treated_rate
            - control_rate
        )

        if (
            pd.notna(
                control_rate
            )
            and control_rate != 0
        ):

            relative_lift = (
                effect
                / control_rate
            )

        else:

            relative_lift = np.nan

        return {
            "treated_outcome_rate":
                treated_rate,

            "control_outcome_rate":
                control_rate,

            "absolute_effect":
                effect,

            "relative_lift":
                relative_lift,
        }

    # ========================================================
    # PLOT WEIGHTS
    # ========================================================

    def plot_weights(
        self,
        df,
        treatment_col,
        weight_col,
        filename,
    ):

        plt.figure(
            figsize=(9, 6)
        )

        for treatment_value in [
            0,
            1,
        ]:

            values = (
                df.loc[
                    df[treatment_col]
                    == treatment_value,
                    weight_col,
                ]
                .astype(float)
                .to_numpy()
            )

            if len(values) == 0:
                continue

            plt.hist(
                values,
                bins=50,
                alpha=0.5,
                density=True,
                label=(
                    f"Treatment = "
                    f"{treatment_value}"
                ),
            )

        plt.xlabel(
            "Propensity Weight"
        )

        plt.ylabel(
            "Density"
        )

        plt.title(
            "Propensity Weight Distribution"
        )

        plt.legend()

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

        print(
            "[WEIGHTING] Saved weight plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        df,
        treatment_col,
        outcome_col,
        estimand="ATE",
        stabilize=True,
        clip=True,
        lower_quantile=0.01,
        upper_quantile=0.99,
    ):

        print("\n" + "=" * 70)
        print("PROPENSITY SCORE WEIGHTING")
        print("=" * 70)

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        self._validate(
            df=df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )

        result = df.copy()

        estimand = (
            str(
                estimand
            )
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # Centralize output column names
        # ----------------------------------------------------

        raw_weight_col = (
            self.raw_weight_col
        )

        weight_col = (
            self.weight_col
        )

        # ====================================================
        # CALCULATE WEIGHTS
        # ====================================================

        if estimand == "ATE":

            if stabilize:

                print(
                    "\nUsing stabilized "
                    "ATE IPTW."
                )

                raw_weights = (
                    self
                    .calculate_stabilized_ate_weights(
                        df=result,
                        treatment_col=(
                            treatment_col
                        ),
                    )
                )

            else:

                print(
                    "\nUsing standard "
                    "ATE IPTW."
                )

                raw_weights = (
                    self
                    .calculate_ate_weights(
                        df=result,
                        treatment_col=(
                            treatment_col
                        ),
                    )
                )

        elif estimand == "ATT":

            print(
                "\nUsing ATT weighting."
            )

            raw_weights = (
                self
                .calculate_att_weights(
                    df=result,
                    treatment_col=(
                        treatment_col
                    ),
                )
            )

        else:

            raise ValueError(
                "estimand must be "
                "'ATE' or 'ATT'."
            )

        result[
            raw_weight_col
        ] = raw_weights

        # ====================================================
        # CLIPPING
        # ====================================================

        if clip:

            (
                final_weights,
                lower,
                upper,
            ) = self.clip_weights(
                weights=raw_weights,
                lower_quantile=(
                    lower_quantile
                ),
                upper_quantile=(
                    upper_quantile
                ),
            )

            print(
                "\nWeight clipping:"
            )

            print(
                f"  Lower quantile: "
                f"{lower_quantile:.2%}"
            )

            print(
                f"  Upper quantile: "
                f"{upper_quantile:.2%}"
            )

            print(
                f"  Lower bound: "
                f"{lower:.4f}"
            )

            print(
                f"  Upper bound: "
                f"{upper:.4f}"
            )

        else:

            final_weights = (
                raw_weights
            )

            lower = None
            upper = None

            print(
                "\nWeight clipping disabled."
            )

        result[
            weight_col
        ] = final_weights

        # ====================================================
        # WEIGHT SUMMARY
        # ====================================================

        summary = (
            self.summarize_weights(
                df=result,
                treatment_col=(
                    treatment_col
                ),
                weight_col=(
                    weight_col
                ),
            )
        )

        print("\n" + "=" * 70)
        print("WEIGHT DIAGNOSTICS")
        print("=" * 70)

        print(
            "\n"
            + summary
            .round(4)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # EFFECTIVE SAMPLE SIZE
        # ====================================================

        total_ess = (
            self.effective_sample_size(
                result[
                    weight_col
                ]
            )
        )

        original_sample_size = (
            len(result)
        )

        if original_sample_size > 0:

            ess_retention = (
                total_ess
                / original_sample_size
            )

        else:

            ess_retention = np.nan

        print(
            f"\nOverall effective sample size: "
            f"{total_ess:,.0f}"
        )

        print(
            f"Original sample size: "
            f"{original_sample_size:,}"
        )

        print(
            "ESS retention: "
            f"{ess_retention:.2%}"
        )

        # ====================================================
        # EFFECT ESTIMATE
        # ====================================================

        effect = (
            self.estimate_effect(
                df=result,
                treatment_col=(
                    treatment_col
                ),
                outcome_col=(
                    outcome_col
                ),
                weight_col=(
                    weight_col
                ),
            )
        )

        print("\n" + "=" * 70)
        print("WEIGHTED TREATMENT EFFECT")
        print("=" * 70)

        print(
            "\nWeighted treated "
            "outcome rate: "
            f"{effect['treated_outcome_rate']:.6f}"
        )

        print(
            "Weighted control "
            "outcome rate: "
            f"{effect['control_outcome_rate']:.6f}"
        )

        print(
            "\nEstimated absolute effect: "
            f"{effect['absolute_effect']:.6f}"
        )

        if pd.notna(
            effect[
                "relative_lift"
            ]
        ):

            print(
                "Estimated relative lift: "
                f"{effect['relative_lift']:.2%}"
            )

        else:

            print(
                "Estimated relative lift: "
                "undefined"
            )

        print(
            "\n[CAUTION] This is a "
            "propensity-adjusted observational "
            "estimate. Its causal interpretation "
            "depends on the causal assumptions."
        )

        # ====================================================
        # SAVE WEIGHT DIAGNOSTICS
        # ====================================================

        summary_path = os.path.join(
            self.output_dir,
            "weight_diagnostics.csv",
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        # ====================================================
        # SAVE EFFECT
        # ====================================================

        effect_df = pd.DataFrame([
            {
                "estimand":
                    estimand,

                "stabilized":
                    (
                        stabilize
                        if estimand == "ATE"
                        else False
                    ),

                "clipped":
                    clip,

                "clip_lower_quantile":
                    (
                        lower_quantile
                        if clip
                        else None
                    ),

                "clip_upper_quantile":
                    (
                        upper_quantile
                        if clip
                        else None
                    ),

                "clip_lower_bound":
                    lower,

                "clip_upper_bound":
                    upper,

                "sample_size":
                    original_sample_size,

                "effective_sample_size":
                    total_ess,

                "ess_retention":
                    ess_retention,

                **effect,
            }
        ])

        effect_path = os.path.join(
            self.output_dir,
            "treatment_effect.csv",
        )

        effect_df.to_csv(
            effect_path,
            index=False,
        )

        # ====================================================
        # SAVE OBSERVATION-LEVEL WEIGHTS
        # ====================================================

        weights_path = os.path.join(
            self.output_dir,
            "propensity_weights.csv",
        )

        keep_columns = [
            treatment_col,
            outcome_col,
            self.propensity_col,
            raw_weight_col,
            weight_col,
        ]

        result[
            keep_columns
        ].to_csv(
            weights_path,
            index=False,
        )

        # ====================================================
        # PLOT
        # ====================================================

        plot_path = (
            self.plot_weights(
                df=result,
                treatment_col=(
                    treatment_col
                ),
                weight_col=(
                    weight_col
                ),
                filename=(
                    "weight_distribution.png"
                ),
            )
        )

        # ====================================================
        # OUTPUT SUMMARY
        # ====================================================

        print(
            "\n[WEIGHTING] Saved diagnostics:"
        )

        print(
            summary_path
        )

        print(
            effect_path
        )

        print(
            weights_path
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            # -----------------------------------------------
            # Full test/scoring DataFrame with propensity
            # scores and calculated weights.
            # -----------------------------------------------

            "data":
                result,

            # -----------------------------------------------
            # Weight distribution diagnostics.
            # -----------------------------------------------

            "summary":
                summary,

            # -----------------------------------------------
            # Estimated weighted treatment effect.
            # -----------------------------------------------

            "effect":
                effect_df,

            # -----------------------------------------------
            # Effective sample size.
            # -----------------------------------------------

            "effective_sample_size":
                total_ess,

            "ess_retention":
                ess_retention,

            # -----------------------------------------------
            # Important for downstream modules.
            # No downstream code needs to hardcode the
            # weight-column name.
            # -----------------------------------------------

            "weight_col":
                weight_col,

            "raw_weight_col":
                raw_weight_col,

            # -----------------------------------------------
            # Configuration
            # -----------------------------------------------

            "estimand":
                estimand,

            "stabilized":
                (
                    stabilize
                    if estimand == "ATE"
                    else False
                ),

            "clipped":
                clip,

            "clip_lower_bound":
                lower,

            "clip_upper_bound":
                upper,

            # -----------------------------------------------
            # Artifact paths
            # -----------------------------------------------

            "summary_path":
                summary_path,

            "effect_path":
                effect_path,

            "weights_path":
                weights_path,

            "plot_path":
                plot_path,
        }