import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CausalDiagnostics:
    """
    Covariate-balance diagnostics for causal inference.

    Main responsibilities:
    - Compare treatment and control groups
    - Calculate standardized mean differences (SMD)
    - Handle numerical and categorical features
    - Flag materially imbalanced covariates
    - Generate a Love plot
    - Save diagnostic results

    Important:
    These diagnostics should be run BEFORE propensity-score
    adjustment and again AFTER weighting / matching.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/diagnostics",
        smd_threshold=0.10,
        max_categories=100,
    ):

        self.output_dir = output_dir
        self.smd_threshold = smd_threshold
        self.max_categories = max_categories

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # ========================================================
    # NUMERIC SMD
    # ========================================================

    @staticmethod
    def _numeric_smd(
        treated,
        control,
    ):
        """
        Standardized mean difference:

        (mean_treated - mean_control)
        /
        sqrt((var_treated + var_control) / 2)
        """

        treated = pd.to_numeric(
            treated,
            errors="coerce",
        ).dropna()

        control = pd.to_numeric(
            control,
            errors="coerce",
        ).dropna()

        if (
            len(treated) == 0
            or len(control) == 0
        ):
            return np.nan

        treated_mean = (
            treated.mean()
        )

        control_mean = (
            control.mean()
        )

        treated_var = (
            treated.var(ddof=1)
        )

        control_var = (
            control.var(ddof=1)
        )

        pooled_sd = np.sqrt(
            (
                treated_var
                + control_var
            )
            / 2
        )

        if (
            not np.isfinite(pooled_sd)
            or pooled_sd == 0
        ):

            if treated_mean == control_mean:
                return 0.0

            return np.nan

        return (
            (treated_mean - control_mean)
            / pooled_sd
        )

    # ========================================================
    # BINARY / CATEGORY-LEVEL SMD
    # ========================================================

    @staticmethod
    def _binary_smd(
        treated_indicator,
        control_indicator,
    ):
        """
        SMD for a binary variable using proportions.
        """

        treated_indicator = (
            pd.Series(
                treated_indicator
            )
            .astype(float)
            .dropna()
        )

        control_indicator = (
            pd.Series(
                control_indicator
            )
            .astype(float)
            .dropna()
        )

        if (
            len(treated_indicator) == 0
            or len(control_indicator) == 0
        ):
            return np.nan

        p_t = (
            treated_indicator.mean()
        )

        p_c = (
            control_indicator.mean()
        )

        pooled = np.sqrt(
            (
                p_t * (1 - p_t)
                + p_c * (1 - p_c)
            )
            / 2
        )

        if pooled == 0:

            if p_t == p_c:
                return 0.0

            return np.nan

        return (
            (p_t - p_c)
            / pooled
        )

    # ========================================================
    # FEATURE TYPE
    # ========================================================

    def _infer_feature_type(
        self,
        series,
    ):
        """
        Infer whether a feature should be treated as numeric
        or categorical for balance diagnostics.

        Numeric ID-like columns with relatively few unique
        values are treated as categorical.
        """

        non_null = (
            series.dropna()
        )

        unique_count = (
            non_null.nunique()
        )

        if unique_count <= 2:
            return "categorical"

        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
            or pd.api.types.is_bool_dtype(series)
        ):
            return "categorical"

        if (
            pd.api.types.is_numeric_dtype(series)
        ):

            # ------------------------------------------------
            # Detect integer-coded categories
            # ------------------------------------------------

            if unique_count <= self.max_categories:

                return "categorical"

            return "numeric"

        return "categorical"

    # ========================================================
    # NUMERIC FEATURE DIAGNOSTICS
    # ========================================================

    def _numeric_feature_balance(
        self,
        df,
        treatment_col,
        feature,
    ):

        treated = df.loc[
            df[treatment_col] == 1,
            feature,
        ]

        control = df.loc[
            df[treatment_col] == 0,
            feature,
        ]

        treated_numeric = (
            pd.to_numeric(
                treated,
                errors="coerce",
            )
        )

        control_numeric = (
            pd.to_numeric(
                control,
                errors="coerce",
            )
        )

        smd = self._numeric_smd(
            treated_numeric,
            control_numeric,
        )

        return {
            "feature": feature,
            "feature_type": "numeric",
            "level": None,

            "treated_mean":
                treated_numeric.mean(),

            "control_mean":
                control_numeric.mean(),

            "treated_std":
                treated_numeric.std(),

            "control_std":
                control_numeric.std(),

            "treated_n":
                treated_numeric.notna().sum(),

            "control_n":
                control_numeric.notna().sum(),

            "smd": smd,

            "abs_smd":
                abs(smd)
                if pd.notna(smd)
                else np.nan,
        }

    # ========================================================
    # CATEGORICAL FEATURE DIAGNOSTICS
    # ========================================================

    def _categorical_feature_balance(
        self,
        df,
        treatment_col,
        feature,
    ):
        """
        Calculate balance for each category level.

        Each category is treated as a binary indicator:

            feature == category

        This avoids treating arbitrary numeric category IDs
        as continuous quantities.
        """

        rows = []

        series = (
            df[feature]
            .astype("string")
            .fillna("__MISSING__")
        )

        treatment = (
            df[treatment_col]
        )

        levels = (
            series
            .value_counts(
                dropna=False
            )
            .index
            .tolist()
        )

        # ----------------------------------------------------
        # Very high cardinality protection
        # ----------------------------------------------------

        if len(levels) > self.max_categories:

            levels = levels[
                :self.max_categories
            ]

            print(
                f"\n[DIAGNOSTICS] "
                f"{feature} has high cardinality. "
                f"Using top "
                f"{self.max_categories} levels "
                f"for balance diagnostics."
            )

        for level in levels:

            indicator = (
                series == level
            ).astype(int)

            treated_indicator = (
                indicator[
                    treatment == 1
                ]
            )

            control_indicator = (
                indicator[
                    treatment == 0
                ]
            )

            treated_pct = (
                treated_indicator.mean()
            )

            control_pct = (
                control_indicator.mean()
            )

            smd = self._binary_smd(
                treated_indicator,
                control_indicator,
            )

            rows.append({
                "feature": feature,
                "feature_type": "categorical",
                "level": str(level),

                "treated_mean":
                    treated_pct,

                "control_mean":
                    control_pct,

                "treated_std":
                    np.nan,

                "control_std":
                    np.nan,

                "treated_n":
                    len(
                        treated_indicator
                    ),

                "control_n":
                    len(
                        control_indicator
                    ),

                "smd": smd,

                "abs_smd":
                    abs(smd)
                    if pd.notna(smd)
                    else np.nan,
            })

        return rows

    # ========================================================
    # CALCULATE BALANCE
    # ========================================================

    def calculate_balance(
        self,
        df,
        treatment_col,
        feature_cols,
    ):

        rows = []

        for feature in feature_cols:

            if feature not in df.columns:

                print(
                    f"\n[DIAGNOSTICS] "
                    f"Skipping missing feature: "
                    f"{feature}"
                )

                continue

            feature_type = (
                self._infer_feature_type(
                    df[feature]
                )
            )

            if feature_type == "numeric":

                rows.append(
                    self._numeric_feature_balance(
                        df=df,
                        treatment_col=treatment_col,
                        feature=feature,
                    )
                )

            else:

                category_rows = (
                    self._categorical_feature_balance(
                        df=df,
                        treatment_col=treatment_col,
                        feature=feature,
                    )
                )

                rows.extend(
                    category_rows
                )

        balance_df = pd.DataFrame(
            rows
        )

        if balance_df.empty:
            return balance_df

        balance_df = (
            balance_df
            .sort_values(
                "abs_smd",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        balance_df[
            "imbalanced"
        ] = (
            balance_df[
                "abs_smd"
            ]
            >= self.smd_threshold
        )

        return balance_df

    # ========================================================
    # WEIGHTED HELPERS
    # ========================================================

    @staticmethod
    def _weighted_mean(
        values,
        weights,
    ):
        """
        Calculate weighted mean while safely handling
        missing / non-finite values.
        """

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
            & (weights >= 0)
        )

        values = values[valid]
        weights = weights[valid]

        if (
            len(values) == 0
            or weights.sum() <= 0
        ):
            return np.nan

        return np.average(
            values,
            weights=weights,
        )


    @staticmethod
    def _weighted_variance(
        values,
        weights,
    ):
        """
        Population-style weighted variance.

        This is appropriate for standardized balance
        diagnostics after propensity weighting.
        """

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
            & (weights >= 0)
        )

        values = values[valid]
        weights = weights[valid]

        if (
            len(values) == 0
            or weights.sum() <= 0
        ):
            return np.nan

        mean = np.average(
            values,
            weights=weights,
        )

        variance = np.average(
            (values - mean) ** 2,
            weights=weights,
        )

        return variance


    # ========================================================
    # WEIGHTED NUMERIC SMD
    # ========================================================

    def _weighted_numeric_smd(
        self,
        treated,
        control,
        treated_weights,
        control_weights,
    ):

        treated = pd.to_numeric(
            treated,
            errors="coerce",
        )

        control = pd.to_numeric(
            control,
            errors="coerce",
        )

        treated_mean = (
            self._weighted_mean(
                treated,
                treated_weights,
            )
        )

        control_mean = (
            self._weighted_mean(
                control,
                control_weights,
            )
        )

        treated_var = (
            self._weighted_variance(
                treated,
                treated_weights,
            )
        )

        control_var = (
            self._weighted_variance(
                control,
                control_weights,
            )
        )

        if (
            not np.isfinite(treated_mean)
            or not np.isfinite(control_mean)
            or not np.isfinite(treated_var)
            or not np.isfinite(control_var)
        ):
            return np.nan

        pooled_sd = np.sqrt(
            (
                treated_var
                + control_var
            )
            / 2
        )

        if (
            not np.isfinite(pooled_sd)
            or pooled_sd == 0
        ):

            if treated_mean == control_mean:
                return 0.0

            return np.nan

        return (
            (treated_mean - control_mean)
            / pooled_sd
        )


    # ========================================================
    # WEIGHTED BINARY SMD
    # ========================================================

    def _weighted_binary_smd(
        self,
        treated_indicator,
        control_indicator,
        treated_weights,
        control_weights,
    ):

        p_t = (
            self._weighted_mean(
                treated_indicator,
                treated_weights,
            )
        )

        p_c = (
            self._weighted_mean(
                control_indicator,
                control_weights,
            )
        )

        if (
            not np.isfinite(p_t)
            or not np.isfinite(p_c)
        ):
            return np.nan

        pooled = np.sqrt(
            (
                p_t * (1 - p_t)
                + p_c * (1 - p_c)
            )
            / 2
        )

        if (
            not np.isfinite(pooled)
            or pooled == 0
        ):

            if p_t == p_c:
                return 0.0

            return np.nan

        return (
            (p_t - p_c)
            / pooled
        )


    # ========================================================
    # WEIGHTED NUMERIC FEATURE BALANCE
    # ========================================================

    def _weighted_numeric_feature_balance(
        self,
        df,
        treatment_col,
        feature,
        weight_col,
    ):

        treated_mask = (
            df[treatment_col] == 1
        )

        control_mask = (
            df[treatment_col] == 0
        )

        treated = pd.to_numeric(
            df.loc[
                treated_mask,
                feature,
            ],
            errors="coerce",
        )

        control = pd.to_numeric(
            df.loc[
                control_mask,
                feature,
            ],
            errors="coerce",
        )

        treated_weights = pd.to_numeric(
            df.loc[
                treated_mask,
                weight_col,
            ],
            errors="coerce",
        )

        control_weights = pd.to_numeric(
            df.loc[
                control_mask,
                weight_col,
            ],
            errors="coerce",
        )

        smd = (
            self._weighted_numeric_smd(
                treated=treated,
                control=control,
                treated_weights=treated_weights,
                control_weights=control_weights,
            )
        )

        treated_mean = (
            self._weighted_mean(
                treated,
                treated_weights,
            )
        )

        control_mean = (
            self._weighted_mean(
                control,
                control_weights,
            )
        )

        treated_var = (
            self._weighted_variance(
                treated,
                treated_weights,
            )
        )

        control_var = (
            self._weighted_variance(
                control,
                control_weights,
            )
        )

        return {
            "feature":
                feature,

            "feature_type":
                "numeric",

            "level":
                None,

            "treated_mean":
                treated_mean,

            "control_mean":
                control_mean,

            "treated_std":
                np.sqrt(
                    treated_var
                )
                if np.isfinite(
                    treated_var
                )
                else np.nan,

            "control_std":
                np.sqrt(
                    control_var
                )
                if np.isfinite(
                    control_var
                )
                else np.nan,

            "treated_n":
                int(
                    treated.notna().sum()
                ),

            "control_n":
                int(
                    control.notna().sum()
                ),

            "smd":
                smd,

            "abs_smd":
                abs(smd)
                if pd.notna(smd)
                else np.nan,
        }


    # ========================================================
    # WEIGHTED CATEGORICAL FEATURE BALANCE
    # ========================================================

    def _weighted_categorical_feature_balance(
        self,
        df,
        treatment_col,
        feature,
        weight_col,
    ):

        rows = []

        series = (
            df[feature]
            .astype("string")
            .fillna("__MISSING__")
        )

        treatment = (
            df[treatment_col]
        )

        weights = pd.to_numeric(
            df[weight_col],
            errors="coerce",
        )

        levels = (
            series
            .value_counts(
                dropna=False
            )
            .index
            .tolist()
        )

        if len(levels) > self.max_categories:

            levels = (
                levels[
                    :self.max_categories
                ]
            )

            print(
                f"\n[DIAGNOSTICS] "
                f"{feature} has high cardinality. "
                f"Using top "
                f"{self.max_categories} levels "
                f"for weighted balance diagnostics."
            )

        treated_mask = (
            treatment == 1
        )

        control_mask = (
            treatment == 0
        )

        for level in levels:

            indicator = (
                series == level
            ).astype(float)

            treated_indicator = (
                indicator[
                    treated_mask
                ]
            )

            control_indicator = (
                indicator[
                    control_mask
                ]
            )

            treated_weights = (
                weights[
                    treated_mask
                ]
            )

            control_weights = (
                weights[
                    control_mask
                ]
            )

            treated_pct = (
                self._weighted_mean(
                    treated_indicator,
                    treated_weights,
                )
            )

            control_pct = (
                self._weighted_mean(
                    control_indicator,
                    control_weights,
                )
            )

            smd = (
                self._weighted_binary_smd(
                    treated_indicator=(
                        treated_indicator
                    ),
                    control_indicator=(
                        control_indicator
                    ),
                    treated_weights=(
                        treated_weights
                    ),
                    control_weights=(
                        control_weights
                    ),
                )
            )

            rows.append({
                "feature":
                    feature,

                "feature_type":
                    "categorical",

                "level":
                    str(level),

                "treated_mean":
                    treated_pct,

                "control_mean":
                    control_pct,

                "treated_std":
                    np.nan,

                "control_std":
                    np.nan,

                "treated_n":
                    int(
                        treated_mask.sum()
                    ),

                "control_n":
                    int(
                        control_mask.sum()
                    ),

                "smd":
                    smd,

                "abs_smd":
                    abs(smd)
                    if pd.notna(smd)
                    else np.nan,
            })

        return rows


    # ========================================================
    # CALCULATE WEIGHTED BALANCE
    # ========================================================

    def calculate_weighted_balance(
        self,
        df,
        treatment_col,
        feature_cols,
        weight_col,
    ):

        if weight_col not in df.columns:

            raise ValueError(
                f"Weight column not found: "
                f"{weight_col}"
            )

        weights = pd.to_numeric(
            df[weight_col],
            errors="coerce",
        )

        if weights.isna().any():

            raise ValueError(
                f"Weight column '{weight_col}' "
                "contains missing or "
                "non-numeric values."
            )

        if (
            ~np.isfinite(
                weights.to_numpy(
                    dtype=float
                )
            )
        ).any():

            raise ValueError(
                f"Weight column '{weight_col}' "
                "contains infinite values."
            )

        if (
            weights <= 0
        ).any():

            raise ValueError(
                f"Weight column '{weight_col}' "
                "must contain positive weights."
            )

        rows = []

        for feature in feature_cols:

            if feature not in df.columns:

                print(
                    f"\n[DIAGNOSTICS] "
                    f"Skipping missing feature: "
                    f"{feature}"
                )

                continue

            feature_type = (
                self._infer_feature_type(
                    df[feature]
                )
            )

            if feature_type == "numeric":

                rows.append(
                    self._weighted_numeric_feature_balance(
                        df=df,
                        treatment_col=treatment_col,
                        feature=feature,
                        weight_col=weight_col,
                    )
                )

            else:

                category_rows = (
                    self._weighted_categorical_feature_balance(
                        df=df,
                        treatment_col=treatment_col,
                        feature=feature,
                        weight_col=weight_col,
                    )
                )

                rows.extend(
                    category_rows
                )

        balance_df = pd.DataFrame(
            rows
        )

        if balance_df.empty:
            return balance_df

        balance_df = (
            balance_df
            .sort_values(
                "abs_smd",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        balance_df[
            "imbalanced"
        ] = (
            balance_df[
                "abs_smd"
            ]
            >= self.smd_threshold
        )

        return balance_df

    # ========================================================
    # FEATURE-LEVEL SUMMARY
    # ========================================================

    def summarize_features(
        self,
        balance_df,
    ):
        """
        Collapse category-level results into a feature-level
        table using maximum absolute SMD.

        This makes the Love plot easier to interpret.
        """

        if balance_df.empty:

            return pd.DataFrame()

        summary = (
            balance_df
            .groupby(
                "feature",
                as_index=False,
            )
            .agg(
                feature_type=(
                    "feature_type",
                    "first",
                ),

                max_abs_smd=(
                    "abs_smd",
                    "max",
                ),

                mean_abs_smd=(
                    "abs_smd",
                    "mean",
                ),

                levels_tested=(
                    "abs_smd",
                    "count",
                ),
            )
        )

        summary[
            "imbalanced"
        ] = (
            summary[
                "max_abs_smd"
            ]
            >= self.smd_threshold
        )

        summary = (
            summary
            .sort_values(
                "max_abs_smd",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        return summary

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    def print_summary(
        self,
        summary_df,
        stage,
    ):

        print("\n" + "=" * 70)
        print(
            f"COVARIATE BALANCE: "
            f"{stage.upper()}"
        )
        print("=" * 70)

        if summary_df.empty:

            print(
                "\nNo balance results available."
            )

            return

        display = (
            summary_df.copy()
        )

        print(
            "\nFeature-level balance:\n"
        )

        print(
            display
            .round(4)
            .to_string(
                index=False
            )
        )

        n_features = (
            len(summary_df)
        )

        n_imbalanced = int(
            summary_df[
                "imbalanced"
            ].sum()
        )

        print(
            f"\nFeatures evaluated: "
            f"{n_features}"
        )

        print(
            f"Imbalanced features "
            f"(|SMD| >= "
            f"{self.smd_threshold:.2f}): "
            f"{n_imbalanced}"
        )

        if n_features > 0:

            pct = (
                n_imbalanced
                / n_features
            )

            print(
                f"Percent imbalanced: "
                f"{pct:.1%}"
            )

        if n_imbalanced > 0:

            print(
                "\nMost imbalanced features:"
            )

            top = (
                summary_df[
                    summary_df[
                        "imbalanced"
                    ]
                ]
                .head(10)
            )

            for _, row in top.iterrows():

                print(
                    f"  - "
                    f"{row['feature']}: "
                    f"{row['max_abs_smd']:.4f}"
                )

    # ========================================================
    # LOVE PLOT
    # ========================================================

    def plot_love(
        self,
        summary_df,
        stage,
        top_n=30,
    ):

        if summary_df.empty:
            return None

        plot_df = (
            summary_df
            .head(top_n)
            .sort_values(
                "max_abs_smd",
                ascending=True,
            )
        )

        plt.figure(
            figsize=(
                10,
                max(
                    6,
                    len(plot_df) * 0.35,
                ),
            )
        )

        plt.scatter(
            plot_df[
                "max_abs_smd"
            ],
            plot_df[
                "feature"
            ],
            s=45,
        )

        plt.axvline(
            self.smd_threshold,
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "Absolute Standardized Mean Difference"
        )

        plt.ylabel(
            "Covariate"
        )

        plt.title(
            "Covariate Balance "
            f"({stage})"
        )

        plt.grid(
            axis="x",
            alpha=0.2,
        )

        path = os.path.join(
            self.output_dir,
            f"love_plot_{stage}.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\n[DIAGNOSTICS] "
            f"Saved Love plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # BEFORE / AFTER BALANCE COMPARISON
    # ========================================================

    def compare_balance(
        self,
        before_summary,
        after_summary,
    ):

        before = (
            before_summary[
                [
                    "feature",
                    "max_abs_smd",
                ]
            ]
            .rename(
                columns={
                    "max_abs_smd":
                        "smd_before",
                }
            )
        )

        after = (
            after_summary[
                [
                    "feature",
                    "max_abs_smd",
                ]
            ]
            .rename(
                columns={
                    "max_abs_smd":
                        "smd_after",
                }
            )
        )

        comparison = (
            before
            .merge(
                after,
                on="feature",
                how="outer",
            )
        )

        comparison[
            "improvement"
        ] = (
            comparison[
                "smd_before"
            ]
            - comparison[
                "smd_after"
            ]
        )

        comparison[
            "pct_reduction"
        ] = np.where(
            comparison[
                "smd_before"
            ] > 0,

            comparison[
                "improvement"
            ]
            / comparison[
                "smd_before"
            ],

            np.nan,
        )

        comparison[
            "balanced_after"
        ] = (
            comparison[
                "smd_after"
            ]
            < self.smd_threshold
        )

        comparison = (
            comparison
            .sort_values(
                "smd_before",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        return comparison


    # ========================================================
    # BEFORE / AFTER LOVE PLOT
    # ========================================================

    def plot_balance_comparison(
        self,
        comparison_df,
        top_n=30,
    ):

        if comparison_df.empty:
            return None

        plot_df = (
            comparison_df
            .head(top_n)
            .sort_values(
                "smd_before",
                ascending=True,
            )
        )

        plt.figure(
            figsize=(
                10,
                max(
                    6,
                    len(plot_df) * 0.4,
                ),
            )
        )

        y_positions = np.arange(
            len(plot_df)
        )

        # --------------------------------------------------------
        # Connecting lines
        # --------------------------------------------------------

        for i, (_, row) in enumerate(
            plot_df.iterrows()
        ):

            plt.plot(
                [
                    row["smd_before"],
                    row["smd_after"],
                ],
                [
                    i,
                    i,
                ],
                linewidth=1,
                alpha=0.5,
            )

        # --------------------------------------------------------
        # Before
        # --------------------------------------------------------

        plt.scatter(
            plot_df[
                "smd_before"
            ],
            y_positions,
            s=55,
            label="Before weighting",
        )

        # --------------------------------------------------------
        # After
        # --------------------------------------------------------

        plt.scatter(
            plot_df[
                "smd_after"
            ],
            y_positions,
            s=55,
            marker="x",
            label="After weighting",
        )

        plt.axvline(
            self.smd_threshold,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"SMD threshold "
                f"({self.smd_threshold:.2f})"
            ),
        )

        plt.yticks(
            y_positions,
            plot_df[
                "feature"
            ],
        )

        plt.xlabel(
            "Absolute Standardized Mean Difference"
        )

        plt.ylabel(
            "Covariate"
        )

        plt.title(
            "Covariate Balance: "
            "Before vs After Weighting"
        )

        plt.legend()

        plt.grid(
            axis="x",
            alpha=0.2,
        )

        path = os.path.join(
            self.output_dir,
            "love_plot_before_vs_after_weighting.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "\n[DIAGNOSTICS] "
            "Saved before/after Love plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    def save_results(
        self,
        balance_df,
        summary_df,
        stage,
    ):

        detail_path = os.path.join(
            self.output_dir,
            f"covariate_balance_"
            f"{stage}_detail.csv",
        )

        summary_path = os.path.join(
            self.output_dir,
            f"covariate_balance_"
            f"{stage}_summary.csv",
        )

        balance_df.to_csv(
            detail_path,
            index=False,
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        print(
            "\n[DIAGNOSTICS] "
            "Saved balance detail:"
            f"\n{detail_path}"
        )

        print(
            "\n[DIAGNOSTICS] "
            "Saved balance summary:"
            f"\n{summary_path}"
        )

        return (
            detail_path,
            summary_path,
        )

    # ========================================================
    # MAIN ENTRY POINT
    # ========================================================

    def run(
        self,
        df,
        treatment_col,
        feature_cols,
        stage="before_adjustment",
    ):

        print("\n" + "=" * 70)
        print("CAUSAL BALANCE DIAGNOSTICS")
        print("=" * 70)

        print(
            f"\nStage: {stage}"
        )

        print(
            f"Treatment: "
            f"{treatment_col}"
        )

        print(
            f"Features: "
            f"{len(feature_cols)}"
        )

        # ----------------------------------------------------
        # Calculate detailed balance
        # ----------------------------------------------------

        balance_df = (
            self.calculate_balance(
                df=df,
                treatment_col=treatment_col,
                feature_cols=feature_cols,
            )
        )

        # ----------------------------------------------------
        # Feature-level summary
        # ----------------------------------------------------

        summary_df = (
            self.summarize_features(
                balance_df
            )
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        self.print_summary(
            summary_df=summary_df,
            stage=stage,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        self.save_results(
            balance_df=balance_df,
            summary_df=summary_df,
            stage=stage,
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        love_plot_path = (
            self.plot_love(
                summary_df=summary_df,
                stage=stage,
            )
        )

        return {
            "detail": balance_df,
            "summary": summary_df,
            "love_plot": love_plot_path,
        }

    # ========================================================
    # POST-WEIGHTING DIAGNOSTICS
    # ========================================================

    def run_weighted(
        self,
        df,
        treatment_col,
        feature_cols,
        weight_col,
        before_summary=None,
        stage="after_weighting",
    ):

        print("\n" + "=" * 70)
        print("POST-WEIGHTING BALANCE DIAGNOSTICS")
        print("=" * 70)

        print(
            f"\nTreatment: "
            f"{treatment_col}"
        )

        print(
            f"Weight: "
            f"{weight_col}"
        )

        print(
            f"Features: "
            f"{len(feature_cols)}"
        )

        # ====================================================
        # WEIGHTED BALANCE
        # ====================================================

        balance_df = (
            self.calculate_weighted_balance(
                df=df,
                treatment_col=treatment_col,
                feature_cols=feature_cols,
                weight_col=weight_col,
            )
        )

        summary_df = (
            self.summarize_features(
                balance_df
            )
        )

        self.print_summary(
            summary_df=summary_df,
            stage=stage,
        )

        # ====================================================
        # SAVE WEIGHTED RESULTS
        # ====================================================

        (
            detail_path,
            summary_path,
        ) = self.save_results(
            balance_df=balance_df,
            summary_df=summary_df,
            stage=stage,
        )

        love_plot_path = (
            self.plot_love(
                summary_df=summary_df,
                stage=stage,
            )
        )

        # ====================================================
        # BALANCE GATE
        # ====================================================

        if summary_df.empty:

            n_imbalanced = 0
            max_smd = np.nan

        else:

            n_imbalanced = int(
                summary_df[
                    "imbalanced"
                ].sum()
            )

            max_smd = (
                summary_df[
                    "max_abs_smd"
                ].max()
            )

        n_features = len(
            summary_df
        )

        balance_passed = (
            n_features > 0
            and n_imbalanced == 0
        )

        print("\n" + "=" * 70)
        print("POST-WEIGHTING BALANCE GATE")
        print("=" * 70)

        print(
            f"\nFeatures evaluated: "
            f"{n_features}"
        )

        print(
            f"Imbalanced after weighting: "
            f"{n_imbalanced}"
        )

        if np.isfinite(
            max_smd
        ):

            print(
                f"Maximum |SMD|: "
                f"{max_smd:.4f}"
            )

        if balance_passed:

            print(
                "\nBalance status: PASS"
            )

            print(
                "All evaluated covariates are "
                f"below |SMD| < "
                f"{self.smd_threshold:.2f}."
            )

        else:

            print(
                "\nBalance status: WARNING"
            )

            print(
                "Residual covariate imbalance "
                "remains after weighting."
            )

        # ====================================================
        # BEFORE / AFTER COMPARISON
        # ====================================================

        comparison_df = None
        comparison_path = None
        comparison_plot = None

        if before_summary is not None:

            comparison_df = (
                self.compare_balance(
                    before_summary=(
                        before_summary
                    ),
                    after_summary=(
                        summary_df
                    ),
                )
            )

            print("\n" + "=" * 70)
            print("BALANCE IMPROVEMENT")
            print("=" * 70)

            print(
                "\n"
                + comparison_df
                .round(4)
                .to_string(
                    index=False
                )
            )

            comparison_path = os.path.join(
                self.output_dir,
                "covariate_balance_"
                "before_vs_after_weighting.csv",
            )

            comparison_df.to_csv(
                comparison_path,
                index=False,
            )

            print(
                "\n[DIAGNOSTICS] "
                "Saved balance comparison:"
                f"\n{comparison_path}"
            )

            comparison_plot = (
                self.plot_balance_comparison(
                    comparison_df
                )
            )

        return {
            "detail":
                balance_df,

            "summary":
                summary_df,

            "detail_path":
                detail_path,

            "summary_path":
                summary_path,

            "love_plot":
                love_plot_path,

            "comparison":
                comparison_df,

            "comparison_path":
                comparison_path,

            "comparison_plot":
                comparison_plot,

            "balance_passed":
                balance_passed,

            "n_imbalanced":
                n_imbalanced,

            "max_abs_smd":
                max_smd,
        }