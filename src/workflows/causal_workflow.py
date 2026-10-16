import numpy as np
import pandas as pd
from src.causal.features import CausalFeatureSelector
from src.causal.diagnostics import (
    CausalDiagnostics,
)
from src.causal.split import (
    causal_group_split,
)
from src.causal.propensity import (
    PropensityModel,
)
from src.preprocessing.pipeline import (
    build_preprocessor,
)
from src.causal.preprocessing import (
    CausalPreprocessor,
)
from src.causal.propensity import (
    PropensityModel,
)
from src.causal.weighting import (
    PropensityWeighting,
)
from src.causal.uplift import (
    UpliftModel,
)
from src.causal.uplift_evaluate import (
    UpliftEvaluator,
)
from src.causal.policy import CausalPolicy
from src.causal.policy_optimizer import (
    CausalPolicyOptimizer,
)
from src.causal.roi import CausalROIAnalyzer
# ============================================================
# HELPERS
# ============================================================

def _section(
    title,
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _yes_no(
    prompt,
    default=True,
):

    suffix = "(Y/n)" if default else "(y/N)"

    answer = input(
        f"\n{prompt} {suffix}:\n> "
    ).strip().lower()

    if answer == "":
        return default

    return answer in [
        "y",
        "yes",
    ]


# ============================================================
# CAUSAL DATA SUMMARY
# ============================================================

def summarize_causal_data(
    df,
    config,
):

    _section(
        "CAUSAL DATA SUMMARY"
    )

    treatment_col = (
        config.treatment_col
    )

    outcome_col = (
        config.outcome_col
    )

    group_col = (
        config.group_col
    )

    time_col = (
        config.time_col
    )

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"\nTreatment: "
        f"{treatment_col}"
    )

    print(
        f"Outcome: "
        f"{outcome_col}"
    )

    print(
        f"Group: "
        f"{group_col}"
    )

    print(
        f"Time: "
        f"{time_col}"
    )

    # --------------------------------------------------------
    # Treatment distribution
    # --------------------------------------------------------

    treatment_counts = (
        df[treatment_col]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    treatment_pct = (
        df[treatment_col]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .sort_index()
    )

    treatment_summary = (
        pd.DataFrame({
            "observations":
                treatment_counts,

            "pct":
                treatment_pct,
        })
        .reset_index()
        .rename(
            columns={
                treatment_col:
                    "treatment"
            }
        )
    )

    print(
        "\nTreatment distribution:"
    )

    print(
        treatment_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Outcome distribution
    # --------------------------------------------------------

    outcome_counts = (
        df[outcome_col]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    outcome_pct = (
        df[outcome_col]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .sort_index()
    )

    outcome_summary = (
        pd.DataFrame({
            "observations":
                outcome_counts,

            "pct":
                outcome_pct,
        })
        .reset_index()
        .rename(
            columns={
                outcome_col:
                    "outcome"
            }
        )
    )

    print(
        "\nOutcome distribution:"
    )

    print(
        outcome_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Outcome by treatment
    # --------------------------------------------------------

    treatment_outcome = (
        df.groupby(
            treatment_col,
            dropna=False,
        )
        .agg(
            observations=(
                outcome_col,
                "size",
            ),

            outcome_rate=(
                outcome_col,
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nObserved outcome by treatment:"
    )

    print(
        treatment_outcome
        .round(6)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Naive difference
    # --------------------------------------------------------

    treatment_values = (
        df[treatment_col]
        .dropna()
        .unique()
    )

    naive_effect = None

    if (
        0 in treatment_values
        and 1 in treatment_values
    ):

        treated_rate = (
            df.loc[
                df[treatment_col] == 1,
                outcome_col,
            ]
            .mean()
        )

        control_rate = (
            df.loc[
                df[treatment_col] == 0,
                outcome_col,
            ]
            .mean()
        )

        naive_effect = (
            treated_rate
            - control_rate
        )

        print(
            "\nNaive observed difference:"
        )

        print(
            f"  Treated outcome rate: "
            f"{treated_rate:.6f}"
        )

        print(
            f"  Control outcome rate: "
            f"{control_rate:.6f}"
        )

        print(
            f"  Difference: "
            f"{naive_effect:.6f}"
        )

        print(
            "\n[CAUTION] This is an observed "
            "difference, NOT a causal treatment "
            "effect."
        )

    # --------------------------------------------------------
    # Unique groups
    # --------------------------------------------------------

    if (
        group_col
        and group_col in df.columns
    ):

        n_groups = (
            df[group_col]
            .nunique(
                dropna=True
            )
        )

        print(
            f"\nUnique groups/entities: "
            f"{n_groups:,}"
        )

        rows_per_group = (
            df.groupby(
                group_col
            )
            .size()
        )

        print(
            "Rows per group:"
        )

        print(
            f"  Mean: "
            f"{rows_per_group.mean():.2f}"
        )

        print(
            f"  Median: "
            f"{rows_per_group.median():.2f}"
        )

        print(
            f"  Max: "
            f"{rows_per_group.max():,}"
        )

    return {
        "treatment_summary":
            treatment_summary,

        "outcome_summary":
            outcome_summary,

        "treatment_outcome":
            treatment_outcome,

        "naive_effect":
            naive_effect,
    }


# ============================================================
# BASIC CAUSAL VALIDATION
# ============================================================

def validate_causal_data(
    df,
    config,
):

    _section(
        "CAUSAL DATA VALIDATION"
    )

    treatment_col = (
        config.treatment_col
    )

    outcome_col = (
        config.outcome_col
    )

    group_col = (
        config.group_col
    )

    time_col = (
        config.time_col
    )

    issues = []
    warnings = []

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        treatment_col,
        outcome_col,
    ]

    if group_col:

        required.append(
            group_col
        )

    if time_col:

        required.append(
            time_col
        )

    missing_columns = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing_columns:

        issues.append(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if issues:

        for issue in issues:

            print(
                f"[ERROR] {issue}"
            )

        raise ValueError(
            "Causal dataset validation failed."
        )

    # --------------------------------------------------------
    # Missing treatment / outcome
    # --------------------------------------------------------

    treatment_missing = int(
        df[treatment_col]
        .isna()
        .sum()
    )

    outcome_missing = int(
        df[outcome_col]
        .isna()
        .sum()
    )

    print(
        f"\nMissing treatment values: "
        f"{treatment_missing:,}"
    )

    print(
        f"Missing outcome values: "
        f"{outcome_missing:,}"
    )

    if treatment_missing > 0:

        issues.append(
            f"{treatment_missing:,} rows "
            "have missing treatment."
        )

    if outcome_missing > 0:

        issues.append(
            f"{outcome_missing:,} rows "
            "have missing outcome."
        )

    # --------------------------------------------------------
    # Binary treatment check
    # --------------------------------------------------------

    treatment_values = set(
        df[treatment_col]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "\nTreatment values:"
    )

    print(
        treatment_values
    )

    if not treatment_values.issubset(
        {0, 1, False, True}
    ):

        issues.append(
            "Current causal workflow requires "
            "a binary treatment encoded as 0/1."
        )

    if len(treatment_values) < 2:

        issues.append(
            "Treatment column contains fewer "
            "than two treatment groups."
        )

    # --------------------------------------------------------
    # Treatment balance
    # --------------------------------------------------------

    treatment_rate = (
        pd.to_numeric(
            df[treatment_col],
            errors="coerce",
        )
        .mean()
    )

    print(
        f"\nTreatment rate: "
        f"{treatment_rate:.4%}"
    )

    if (
        treatment_rate < 0.05
        or treatment_rate > 0.95
    ):

        warnings.append(
            "Treatment assignment is highly "
            "imbalanced. Overlap may be weak."
        )

    # --------------------------------------------------------
    # Outcome check
    # --------------------------------------------------------

    outcome_values = set(
        df[outcome_col]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "\nOutcome values:"
    )

    print(
        outcome_values
    )

    # Binary outcome is ideal for the first uplift version,
    # but we don't make this fatal because later causal
    # workflows can support continuous outcomes.

    binary_outcome = (
        outcome_values.issubset(
            {0, 1, False, True}
        )
        and len(outcome_values) <= 2
    )

    print(
        "\nBinary outcome:",
        binary_outcome,
    )

    # --------------------------------------------------------
    # Group leakage risk
    # --------------------------------------------------------

    if group_col:

        duplicate_groups = (
            df[group_col]
            .duplicated()
            .any()
        )

        if duplicate_groups:

            print(
                "\n[CAUSAL] Repeated entities detected."
            )

            print(
                "[CAUSAL] Group-aware splitting "
                "will be required."
            )

    # --------------------------------------------------------
    # Time ordering
    # --------------------------------------------------------

    if time_col:

        if df[
            time_col
        ].isna().any():

            warnings.append(
                "Decision-time column contains "
                "missing values."
            )

    # --------------------------------------------------------
    # Print warnings / errors
    # --------------------------------------------------------

    if warnings:

        print(
            "\nWarnings:"
        )

        for warning in warnings:

            print(
                f"  - {warning}"
            )

    if issues:

        print(
            "\nValidation issues:"
        )

        for issue in issues:

            print(
                f"  - {issue}"
            )

        raise ValueError(
            "Causal dataset validation failed."
        )

    print(
        "\n[CAUSAL] Basic validation passed."
    )

    return {
        "binary_outcome":
            binary_outcome,

        "treatment_rate":
            treatment_rate,

        "warnings":
            warnings,
    }


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_causal_features(
    df,
    config,
    causal_features,
):

    _section(
        "CAUSAL FEATURE VALIDATION"
    )

    if not causal_features:

        raise ValueError(
            "No causal features were selected."
        )

    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    causal_features = list(
        dict.fromkeys(
            causal_features
        )
    )

    # --------------------------------------------------------
    # Make sure features exist
    # --------------------------------------------------------

    missing = [
        col
        for col in causal_features
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Causal features missing from "
            f"dataset: {missing}"
        )

    # --------------------------------------------------------
    # Leakage columns
    # --------------------------------------------------------

    forbidden = {
        config.treatment_col,
        config.outcome_col,
    }

    if config.group_col:

        forbidden.add(
            config.group_col
        )

    if config.time_col:

        forbidden.add(
            config.time_col
        )

    leakage_features = [
        col
        for col in causal_features
        if col in forbidden
    ]

    if leakage_features:

        print(
            "\n[CAUSAL] Removing reserved columns "
            "from causal features:"
        )

        for col in leakage_features:

            print(
                f"  - {col}"
            )

        causal_features = [
            col
            for col in causal_features
            if col not in forbidden
        ]

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    constant_features = []

    for col in causal_features:

        if (
            df[col]
            .nunique(
                dropna=False
            )
            <= 1
        ):

            constant_features.append(
                col
            )

    if constant_features:

        print(
            "\n[CAUSAL] Removing constant features:"
        )

        for col in constant_features:

            print(
                f"  - {col}"
            )

        causal_features = [
            col
            for col in causal_features
            if col not in constant_features
        ]

    print(
        f"\nFinal causal features: "
        f"{len(causal_features)}"
    )

    for col in causal_features:

        print(
            f"  - {col}"
        )

    return causal_features


# ============================================================
# CREATE CAUSAL MODEL DATA
# ============================================================

def prepare_causal_data(
    df,
    config,
    causal_features,
):

    _section(
        "PREPARING CAUSAL MODEL DATA"
    )

    columns = list(
        causal_features
    )

    required_columns = [
        config.treatment_col,
        config.outcome_col,
    ]

    if config.group_col:

        required_columns.append(
            config.group_col
        )

    if config.time_col:

        required_columns.append(
            config.time_col
        )

    for col in required_columns:

        if col not in columns:

            columns.append(
                col
            )

    causal_df = (
        df[columns]
        .copy()
    )

    # --------------------------------------------------------
    # Drop rows missing treatment/outcome
    # --------------------------------------------------------

    before = len(
        causal_df
    )

    causal_df = (
        causal_df
        .dropna(
            subset=[
                config.treatment_col,
                config.outcome_col,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    removed = (
        before
        - len(causal_df)
    )

    if removed:

        print(
            f"\nRemoved {removed:,} rows "
            "with missing treatment/outcome."
        )

    # --------------------------------------------------------
    # Standardize treatment
    # --------------------------------------------------------

    causal_df[
        config.treatment_col
    ] = (
        causal_df[
            config.treatment_col
        ]
        .astype(int)
    )

    print(
        f"\nCausal modeling rows: "
        f"{len(causal_df):,}"
    )

    print(
        f"Causal features: "
        f"{len(causal_features):,}"
    )

    return causal_df


# ============================================================
# CAUSAL PIPELINE ROADMAP
# ============================================================

def print_causal_pipeline():

    _section(
        "CAUSAL MODELING PIPELINE"
    )

    stages = [
        (
            "1",
            "Treatment / outcome validation",
            "complete",
        ),
        (
            "2",
            "Covariate balance diagnostics",
            "next",
        ),
        (
            "3",
            "Group-aware train/test split",
            "next",
        ),
        (
            "4",
            "Propensity score model",
            "planned",
        ),
        (
            "5",
            "Overlap / positivity diagnostics",
            "planned",
        ),
        (
            "6",
            "PSM / weighting adjustment",
            "planned",
        ),
        (
            "7",
            "Uplift / CATE model",
            "planned",
        ),
        (
            "8",
            "Uplift evaluation",
            "planned",
        ),
        (
            "9",
            "Marginal ROI analysis",
            "planned",
        ),
        (
            "10",
            "Causal scoring output",
            "planned",
        ),
    ]

    stage_df = pd.DataFrame(
        stages,
        columns=[
            "stage",
            "component",
            "status",
        ],
    )

    print(
        "\n"
        + stage_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN CAUSAL WORKFLOW
# ============================================================

def run_causal_workflow(
    df,
    causal_config,
):

    print("\n" + "=" * 70)
    print("CAUSAL / UPLIFT WORKFLOW")
    print("=" * 70)

    print(
        "\nObjective:"
        "\nEstimate incremental treatment impact "
        "rather than simple outcome probability."
    )

    # ========================================================
    # 1. VALIDATE CONFIGURATION
    # ========================================================

    print(
        "\n[1/5] Validating causal configuration..."
    )

    causal_config.validate(
        df
    )

    # ========================================================
    # 2. VALIDATE DATASET
    # ========================================================

    print(
        "\n[2/5] Validating causal dataset..."
    )

    # your existing validation code here


    # ========================================================
    # 3. SUMMARIZE TREATMENT / OUTCOME
    # ========================================================

    print(
        "\n[3/5] Summarizing treatment and outcome..."
    )

    # your existing summary code here


    # ========================================================
    # 4. SELECT CAUSAL FEATURES
    # ========================================================

    print(
        "\n[4/5] Selecting causal features..."
    )

    selector = CausalFeatureSelector(
        df=df,
        config=causal_config,
    )

    (
        causal_features,
        causal_excluded_cols,
    ) = selector.run()

    # ============================================================
    # CAUSAL PREPROCESSING CONFIGURATION
    # ============================================================

    causal_preprocessing = (
        CausalPreprocessor(
            df=df,
            feature_cols=causal_features,
        )
    )

    (
        causal_preprocessor,
        causal_preprocessing_config,
    ) = causal_preprocessing.run()

    # ============================================================
    # COVARIATE BALANCE DIAGNOSTICS
    # ============================================================

    diagnostics = CausalDiagnostics()

    balance_results = diagnostics.run(
        df=df,
        treatment_col=causal_config.treatment_col,
        feature_cols=causal_features,
        stage="before_adjustment",
    )
    # ============================================================
    # GROUP-AWARE CAUSAL SPLIT
    # ============================================================

    split_results = causal_group_split(
        df=df,
        group_col=causal_config.group_col,
        treatment_col=causal_config.treatment_col,
        outcome_col=causal_config.outcome_col,
        feature_cols=causal_features,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    # ============================================================
    # PROPENSITY SCORE MODEL
    # ============================================================

    propensity_model = (
        PropensityModel()
    )

    propensity_results = (
        propensity_model.run(
            preprocessor=(
                causal_preprocessor
            ),

            split_results=(
                split_results
            ),

            treatment_col=(
                causal_config.treatment_col
            ),

            outcome_col=(
                causal_config.outcome_col
            ),

            group_col=(
                causal_config.group_col
            ),
        )
    )

    # ============================================================
    # PROPENSITY SCORE WEIGHTING
    # ============================================================

    print("\n" + "=" * 70)
    print("PROPENSITY SCORE ADJUSTMENT")
    print("=" * 70)

    run_weighting = input(
        "\nRun propensity-score weighting? "
        "(Y/n):\n> "
    ).strip().lower()

    weighting_results = None

    if run_weighting in [
        "",
        "y",
        "yes",
    ]:

        # --------------------------------------------------------
        # Estimand
        # --------------------------------------------------------

        print(
            "\nEstimand:"
            "\n  1. ATE - effect across the target population"
            "\n  2. ATT - effect among treated observations"
        )

        estimand_choice = input(
            "\nSelect estimand "
            "(Enter for ATE):\n> "
        ).strip()

        if estimand_choice == "2":
            estimand = "ATT"
        else:
            estimand = "ATE"

        # --------------------------------------------------------
        # Use TEST data for final causal estimate
        #
        # Propensity model was learned using train data.
        # Test remains our clean held-out evaluation population.
        # --------------------------------------------------------

        causal_test = (
            propensity_results[
                "scores"
            ][
                "test"
            ][
                "scores"
            ]
            .copy()
        )

        weighting = (
            PropensityWeighting()
        )

        weighting_results = (
            weighting.run(
                df=causal_test,
                treatment_col=(
                    causal_config.treatment_col
                ),
                outcome_col=(
                    causal_config.outcome_col
                ),
                estimand=estimand,
                stabilize=True,
                clip=True,
                lower_quantile=0.01,
                upper_quantile=0.99,
            )
        )

        weighted_test = (
            weighting_results[
                "data"
            ]
        )

    # ============================================================
    # POST-WEIGHTING BALANCE DIAGNOSTICS
    # ============================================================

    print("\n" + "=" * 70)
    print("POST-WEIGHTING COVARIATE BALANCE")
    print("=" * 70)

    # ------------------------------------------------------------
    # Get weighted dataset
    # ------------------------------------------------------------

    weighted_df = (
        weighting_results[
            "data"
        ]
    )

    # ------------------------------------------------------------
    # Get weight column dynamically
    #
    # IMPORTANT:
    # Do not hardcode "propensity_weight" here.
    # PropensityWeighting tells downstream modules which
    # final weight column it created.
    # ------------------------------------------------------------

    weight_col = (
        weighting_results[
            "weight_col"
        ]
    )

    print(
        f"\n[CAUSAL] Using weight column: "
        f"{weight_col}"
    )

    # ------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------

    if weight_col not in weighted_df.columns:

        raise ValueError(
            "Weighting output is inconsistent. "
            f"Expected weight column "
            f"'{weight_col}' was not found "
            "in the weighted dataset."
        )

    # ------------------------------------------------------------
    # Run weighted balance diagnostics
    # ------------------------------------------------------------

    weighted_balance_results = (
        diagnostics.run_weighted(
            df=weighted_df,

            treatment_col=(
                causal_config
                .treatment_col
            ),

            feature_cols=(
                causal_features
            ),

            weight_col=(
                weight_col
            ),

            stage=(
                "after_weighting"
            ),
        )
    )

    # ============================================================
    # UPLIFT / CATE MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("UPLIFT / CATE ESTIMATION")
    print("=" * 70)

    run_uplift = input(
        "\nRun uplift / CATE modeling? "
        "(Y/n):\n> "
    ).strip().lower()

    uplift_results = None

    if run_uplift in [
        "",
        "y",
        "yes",
    ]:

        uplift_model = UpliftModel()

        uplift_results = (
            uplift_model.run(
                preprocessor=(
                    causal_preprocessor
                ),

                split_results=(
                    split_results
                ),

                propensity_results=(
                    propensity_results
                ),

                original_df=(
                    df
                ),

                # Start WITHOUT using IPTW during T-learner
                # training.
                #
                # We have already used propensity weighting
                # for balance/effect diagnostics.
                use_training_weights=False,
            )
        )
    # ========================================================
    # UPLIFT EVALUATION
    # ========================================================

    uplift_evaluation_results = None

    if uplift_results is not None:

        print("\n" + "=" * 70)
        print("UPLIFT MODEL EVALUATION")
        print("=" * 70)

        run_uplift_evaluation = input(
            "\nEvaluate uplift model? "
            "(Y/n):\n> "
        ).strip().lower()

        if run_uplift_evaluation in [
            "",
            "y",
            "yes",
        ]:

            # ----------------------------------------------------
            # Retrieve test scoring table from uplift model
            # ----------------------------------------------------

            uplift_test = (
                uplift_results[
                    "scores"
                ]
                .copy()
            )

            # ----------------------------------------------------
            # Add propensity weights if available
            # ----------------------------------------------------

            evaluation_weight_col = None

            if weighting_results is not None:

                weighted_test = (
                    weighting_results[
                        "data"
                    ]
                )

                weight_col = (
                    weighting_results[
                        "weight_col"
                    ]
                )

                # The uplift test scores and weighted test data
                # must represent the SAME test rows in the SAME
                # order.
                if (
                    len(weighted_test)
                    == len(uplift_test)
                ):

                    uplift_test[
                        weight_col
                    ] = (
                        weighted_test[
                            weight_col
                        ]
                        .to_numpy()
                    )

                    evaluation_weight_col = (
                        weight_col
                    )

                    print(
                        "\n[UPLIFT EVALUATION] "
                        "Using propensity weights."
                    )

                else:

                    print(
                        "\n[UPLIFT EVALUATION] Warning: "
                        "weighted test rows do not match "
                        "uplift test rows."
                    )

                    print(
                        "Running unweighted evaluation."
                    )

            # ----------------------------------------------------
            # Evaluate
            # ----------------------------------------------------

            evaluator = (
                UpliftEvaluator()
            )

            uplift_evaluation_results = (
                evaluator.run(
                    df=uplift_test,
                    treatment_col=(
                        causal_config
                        .treatment_col
                    ),
                    outcome_col=(
                        causal_config
                        .outcome_col
                    ),
                    uplift_col=(
                        "uplift_score"
                    ),
                    weight_col=(
                        evaluation_weight_col
                    ),
                )
            )

    # ========================================================
    # CAUSAL TARGETING POLICY
    # ========================================================

    run_policy = input(
        "\nBuild causal targeting policy? (Y/n):\n> "
    ).strip().lower()

    policy_results = None

    if run_policy not in [
        "n",
        "no",
    ]:

        print("\n" + "=" * 70)
        print("CAUSAL TARGETING POLICY")
        print("=" * 70)

        print(
            "\nAvailable policies:"
        )

        print(
            "  1. Positive uplift"
        )

        print(
            "  2. Minimum uplift threshold"
        )

        print(
            "  3. Top-K targeting"
        )

        print(
            "  4. Positive expected net value"
        )

        policy_choice = input(
            "\nSelect policy "
            "(Enter for positive uplift):\n> "
        ).strip()

        policy_map = {
            "":
                "positive_uplift",

            "1":
                "positive_uplift",

            "2":
                "uplift_threshold",

            "3":
                "top_k",

            "4":
                "net_value",
        }

        policy_name = (
            policy_map.get(
                policy_choice
            )
        )

        if policy_name is None:

            raise ValueError(
                "Invalid policy selection."
            )

        # ====================================================
        # POLICY-SPECIFIC SETTINGS
        # ====================================================

        uplift_threshold = 0.0
        target_fraction = 0.10

        if (
            policy_name
            == "uplift_threshold"
        ):

            threshold_input = input(
                "\nMinimum predicted uplift "
                "(example 0.05 for 5%):\n> "
            ).strip()

            uplift_threshold = float(
                threshold_input
            )

        elif (
            policy_name
            == "top_k"
        ):

            fraction_input = input(
                "\nTarget what fraction of "
                "highest-uplift observations? "
                "(example 0.20 for top 20%):\n> "
            ).strip()

            target_fraction = float(
                fraction_input
            )

        # ====================================================
        # BUSINESS VALUE
        # ====================================================

        outcome_value_constant = None
        outcome_value_col = None

        treatment_cost_constant = None
        treatment_cost_col = None

        add_business_value = input(
            "\nAdd business value / cost "
            "analysis? (y/N):\n> "
        ).strip().lower()

        if add_business_value in [
            "y",
            "yes",
        ]:

            print(
                "\nOutcome value can be:"
            )

            print(
                "  1. Constant value per outcome"
            )

            print(
                "  2. Existing DataFrame column"
            )

            value_type = input(
                "\nSelect outcome value type:\n> "
            ).strip()

            if value_type == "1":

                value_input = input(
                    "\nValue of one incremental "
                    "outcome:\n> "
                ).strip()

                outcome_value_constant = float(
                    value_input
                )

            elif value_type == "2":

                outcome_value_col = input(
                    "\nEnter outcome-value column:\n> "
                ).strip()

            else:

                raise ValueError(
                    "Invalid outcome value type."
                )

            print(
                "\nTreatment cost can be:"
            )

            print(
                "  1. Constant cost per treatment"
            )

            print(
                "  2. Existing DataFrame column"
            )

            cost_type = input(
                "\nSelect treatment cost type:\n> "
            ).strip()

            if cost_type == "1":

                cost_input = input(
                    "\nCost of one treatment:\n> "
                ).strip()

                treatment_cost_constant = float(
                    cost_input
                )

            elif cost_type == "2":

                treatment_cost_col = input(
                    "\nEnter treatment-cost column:\n> "
                ).strip()

            else:

                raise ValueError(
                    "Invalid treatment cost type."
                )

        # ====================================================
        # NET VALUE REQUIRES VALUE + COST
        # ====================================================

        if (
            policy_name
            == "net_value"
            and add_business_value
            not in [
                "y",
                "yes",
            ]
        ):

            raise ValueError(
                "Net-value policy requires "
                "outcome value and treatment cost."
            )

        # ====================================================
        # RUN POLICY
        # ====================================================

        policy_builder = (
            CausalPolicy()
        )

        policy_results = (
            policy_builder.run(
                df=uplift_results[
                    "scores"
                ],

                uplift_col=(
                    "uplift_score"
                ),

                policy=(
                    policy_name
                ),

                uplift_threshold=(
                    uplift_threshold
                ),

                target_fraction=(
                    target_fraction
                ),

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

    # ========================================================
    # POLICY OPTIMIZATION
    # ========================================================
    policy_optimization_results = None
    if (
        policy_results
        is not None
    ):

        run_policy_optimization = input(
            "\nOptimize causal targeting policy? (Y/n):\n> "
        ).strip().lower()

        if run_policy_optimization not in [
            "n",
            "no",
        ]:

            print("\n" + "=" * 70)
            print("POLICY OPTIMIZATION CONFIGURATION")
            print("=" * 70)

            print(
                "\nOptimization objective:"
            )

            print(
                "  1. Maximize incremental outcomes"
            )

            print(
                "  2. Maximize incremental outcomes "
                "per targeted observation"
            )

            has_business_value = (
                "expected_net_value"
                in policy_results[
                    "data"
                ].columns
            )

            has_roi = (
                "expected_incremental_value"
                in policy_results[
                    "data"
                ].columns
                and
                "treatment_cost"
                in policy_results[
                    "data"
                ].columns
            )

            if has_business_value:

                print(
                    "  3. Maximize expected net value"
                )

            if has_roi:

                print(
                    "  4. Maximize expected ROI"
                )

            objective_choice = input(
                "\nSelect objective "
                "(Enter for 1):\n> "
            ).strip()

            objective_map = {
                "":
                    "incremental_outcomes",

                "1":
                    "incremental_outcomes",

                "2":
                    "incremental_outcomes_per_target",

                "3":
                    "net_value",

                "4":
                    "roi",
            }

            objective = (
                objective_map.get(
                    objective_choice
                )
            )

            if objective is None:

                raise ValueError(
                    "Invalid policy optimization "
                    "objective."
                )

            if (
                objective == "net_value"
                and not has_business_value
            ):

                raise ValueError(
                    "Net-value optimization requires "
                    "business value and treatment cost."
                )

            if (
                objective == "roi"
                and not has_roi
            ):

                raise ValueError(
                    "ROI optimization requires "
                    "business value and treatment cost."
                )

            # ====================================================
            # TARGETING GRID
            # ====================================================

            step_input = input(
                "\nTargeting grid step "
                "(Enter for 0.05 = 5%):\n> "
            ).strip()

            target_step = (
                float(
                    step_input
                )
                if step_input
                else 0.05
            )

            # ====================================================
            # OPTIONAL CAPACITY CONSTRAINT
            # ====================================================

            capacity_input = input(
                "\nMaximum fraction that can be "
                "targeted "
                "(Enter for no limit, "
                "example 0.30):\n> "
            ).strip()

            maximum_target_fraction = (
                float(
                    capacity_input
                )
                if capacity_input
                else None
            )

            # ====================================================
            # OPTIONAL MINIMUM UPLIFT
            # ====================================================

            minimum_uplift_input = input(
                "\nMinimum acceptable mean uplift "
                "(Enter for none, example 0.05):\n> "
            ).strip()

            minimum_mean_uplift = (
                float(
                    minimum_uplift_input
                )
                if minimum_uplift_input
                else None
            )

            # ====================================================
            # RUN OPTIMIZER
            # ====================================================

            optimizer = (
                CausalPolicyOptimizer()
            )

            policy_optimization_results = (
                optimizer.run(
                    df=policy_results[
                        "data"
                    ],

                    uplift_col=(
                        "uplift_score"
                    ),

                    objective=(
                        objective
                    ),

                    start=0.05,

                    stop=1.00,

                    step=(
                        target_step
                    ),

                    minimum_mean_uplift=(
                        minimum_mean_uplift
                    ),

                    maximum_target_fraction=(
                        maximum_target_fraction
                    ),
                )
            )

    # ========================================================
    # MARGINAL ROI ANALYSIS
    # ========================================================

    roi_results = None

    if uplift_results is not None:

        print("\n" + "=" * 70)
        print("MARGINAL ROI ANALYSIS")
        print("=" * 70)

        run_roi = input(
            "\nRun marginal ROI analysis? (Y/n):\n> "
        ).strip().lower()

        if run_roi not in {
            "n",
            "no",
        }:

            uplift_test = (
                uplift_results[
                    "scores"
                ].copy()
            )

            # ----------------------------------------------------
            # OUTCOME VALUE
            # ----------------------------------------------------

            print(
                "\nValue of one positive outcome."
            )

            print(
                "Example: if one conversion is worth "
                "$20, enter 20."
            )

            while True:

                value_input = input(
                    "\nOutcome value:\n> "
                ).strip()

                try:

                    outcome_value = float(
                        value_input
                    )

                    if outcome_value <= 0:

                        print(
                            "Outcome value must be "
                            "greater than 0."
                        )

                        continue

                    break

                except ValueError:

                    print(
                        "Please enter a numeric value."
                    )

            # ----------------------------------------------------
            # TREATMENT COST
            # ----------------------------------------------------

            print(
                "\nCost of treating one observation."
            )

            print(
                "Example: ad impression, incentive, "
                "message, or intervention cost."
            )

            while True:

                cost_input = input(
                    "\nTreatment cost per observation:\n> "
                ).strip()

                try:

                    treatment_cost = float(
                        cost_input
                    )

                    if treatment_cost < 0:

                        print(
                            "Treatment cost cannot "
                            "be negative."
                        )

                        continue

                    break

                except ValueError:

                    print(
                        "Please enter a numeric value."
                    )

            # ----------------------------------------------------
            # OPTIONAL BUDGET
            # ----------------------------------------------------

            print(
                "\nOptional total treatment budget."
            )

            budget_input = input(
                "Enter for no budget constraint:\n> "
            ).strip()

            if budget_input == "":

                budget = None

            else:

                try:

                    budget = float(
                        budget_input
                    )

                    if budget < 0:

                        raise ValueError

                except ValueError:

                    raise ValueError(
                        "Budget must be a "
                        "non-negative number."
                    )

            # ----------------------------------------------------
            # RUN
            # ----------------------------------------------------

            roi_analyzer = (
                CausalROIAnalyzer()
            )

            roi_results = (
                roi_analyzer.run(
                    df=uplift_test,
                    uplift_col="uplift_score",
                    outcome_value=outcome_value,
                    treatment_cost=treatment_cost,
                    budget=budget,
                    targeting_step=0.05,
                )
            )

    # ========================================================
    # 5. PREPARE CAUSAL MODELING DATA
    # ========================================================

    print(
        "\n[5/5] Preparing causal modeling dataset..."
    )

    X = df[
        causal_features
    ].copy()

    treatment = df[
        causal_config.treatment_col
    ].copy()

    outcome = df[
        causal_config.outcome_col
    ].copy()

    print(
        f"\nCausal modeling rows: "
        f"{len(X):,}"
    )

    print(
        f"Causal features: "
        f"{len(causal_features):,}"
    )

    return {
        "X": X,
        "treatment": treatment,
        "outcome": outcome,
        "features": causal_features,
        "excluded_features": causal_excluded_cols,
        "config": causal_config,

        "balance_before":
            balance_results,

        "split":
            split_results,

        "propensity":
            propensity_results,

        "weighting":
            weighting_results,

        "balance_after":
            weighted_balance_results,

        "uplift":
            uplift_results,

        "uplift_evaluation":
            uplift_evaluation_results,

        "policy":
            policy_results,

        "policy_optimization":
            policy_optimization_results,

        "roi":
            roi_results,    
    }