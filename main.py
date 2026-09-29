import os
import joblib
import pandas as pd
import numpy as np
import shutil
import json
from src.data.bigquery import BigQueryDataSource
from src.profiling.profiler import DatasetProfiler
from src.features.columns import ColumnManager
from src.profiling.categorical_eda import CategoricalEDA
from src.profiling.numerical_eda import NumericalEDA
from src.preprocessing.categorical import CategoricalPreprocessor
from src.preprocessing.numerical import NumericalPreprocessor
from src.profiling.correlation import CorrelationAnalysis
from src.modeling.problem import confirm_problem_type
from src.modeling.models import get_available_models
from src.modeling.split import split_data
from src.preprocessing.pipeline import build_preprocessor
from src.modeling.train import train_models
from src.modeling.evaluate import ModelEvaluator
from src.modeling.tune import tune_model
from src.modeling.finalize import (
    finalize_model,
    load_final_model,
)
from src.modeling.interpret import ModelInterpreter
from src.utils.run_logger import RunLogger
from sklearn import set_config

set_config(
    transform_output="pandas"
)

PROJECT_ID = "project-business-prd"
DATASET_ID = "prism_studio"

CACHE_PATH = "data/raw/dataset.parquet"
DATASET_METADATA_PATH = "data/raw/dataset_metadata.json"


# ============================================================
# DATA PROJECT
# ============================================================

print("\n" + "=" * 70)
print("DATA PROJECT")
print("=" * 70)


# ------------------------------------------------------------
# Check whether a cached dataset exists
# ------------------------------------------------------------

cache_exists = os.path.exists(
    CACHE_PATH
)

metadata_exists = os.path.exists(
    DATASET_METADATA_PATH
)


# ------------------------------------------------------------
# Read existing dataset metadata
# ------------------------------------------------------------

dataset_metadata = None

if metadata_exists:

    with open(
        DATASET_METADATA_PATH,
        "r",
    ) as f:

        dataset_metadata = json.load(f)


# ------------------------------------------------------------
# Existing project
# ------------------------------------------------------------

if cache_exists:

    if dataset_metadata:

        current_table = (
            dataset_metadata.get(
                "full_table_name",
                "Unknown",
            )
        )

        print(
            f"\nCurrent cached dataset:"
            f"\n{current_table}"
        )

    else:

        print(
            "\nCurrent cached dataset found."
        )

        print(
            "BigQuery table metadata is unavailable."
        )


    new_project = input(
        "\nStart a new data project with "
        "a different BigQuery table? (y/N):\n> "
    ).strip().lower()


else:

    print(
        "\nNo cached dataset found."
    )

    new_project = "y"


# ============================================================
# NEW DATA PROJECT
# ============================================================

if new_project in [
    "y",
    "yes",
]:

    print("\n" + "=" * 70)
    print("NEW DATA PROJECT")
    print("=" * 70)

    # --------------------------------------------------------
    # Connect to BigQuery
    # --------------------------------------------------------

    bq = BigQueryDataSource()

    # --------------------------------------------------------
    # List tables
    # --------------------------------------------------------

    tables = bq.list_tables(
        DATASET_ID
    )

    print(
        f"\nAvailable tables in "
        f"{PROJECT_ID}.{DATASET_ID}:"
    )

    for table in tables:

        print(
            f"  - {table}"
        )


    # --------------------------------------------------------
    # Ask user for table
    # --------------------------------------------------------

    table_name = input(
        "\nEnter table name:\n> "
    ).strip()


    if not table_name:

        raise ValueError(
            "Table name cannot be empty."
        )


    if table_name not in tables:

        raise ValueError(
            f"Unknown table: {table_name}"
        )


    # --------------------------------------------------------
    # Full BigQuery table name
    # --------------------------------------------------------

    full_table_name = (
        f"{PROJECT_ID}."
        f"{DATASET_ID}."
        f"{table_name}"
    )


    print(
        f"\n[BIGQUERY] Loading:"
        f"\n{full_table_name}"
    )


    # ========================================================
    # FULL DATA OR SAMPLE
    # ========================================================

    use_sample = input(
        "\nLoad a random sample instead of "
        "the full table? (y/N):\n> "
    ).strip().lower()
    if use_sample in [
        "y",
        "yes",
    ]:
        # ----------------------------------------------------
        # Ask sample size
        # ----------------------------------------------------
        sample_input = input(
            "\nHow many rows would you like "
            "to sample? "
            "(default 50000):\n> "
        ).strip()
        if sample_input == "":
            sample_size = 50_000
        else:
            try:
                sample_size = int(
                    sample_input
                )
            except ValueError:
                print(
                    "\n[DATA] Invalid sample size. "
                    "Using 50,000."
                )
                sample_size = 50_000
        if sample_size <= 0:
            raise ValueError(
                "Sample size must be greater than 0."
            )
        # ----------------------------------------------------
        # Load sample directly from BigQuery
        # ----------------------------------------------------
        df = bq.load_table_sample(
            table_name=full_table_name,
            sample_size=sample_size,
        )
        sampling_method = "random_sample"
    # ========================================================
    # FULL DATA
    # ========================================================
    else:
        print(
            "\n[BIGQUERY] Loading full table..."
        )
        df = bq.load_table(
            full_table_name
        )
        sample_size = None
        sampling_method = "full_table"
    # --------------------------------------------------------
    # Save local cache
    # --------------------------------------------------------

    bq.save_local(
        df,
        CACHE_PATH,
    )
    # --------------------------------------------------------
    # Save dataset metadata
    # --------------------------------------------------------

    dataset_metadata = {
        "project_id": PROJECT_ID,
        "dataset_id": DATASET_ID,
        "table_name": table_name,
        "full_table_name": (
            full_table_name
        ),
        "rows": int(
            df.shape[0]
        ),
        "columns": int(
            df.shape[1]
        ),
    }


    with open(
        DATASET_METADATA_PATH,
        "w",
    ) as f:

        json.dump(
            dataset_metadata,
            f,
            indent=4,
        )


    print(
        "\n[DATA] New dataset cached."
    )

    print(
        f"[DATA] Shape: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )


# ============================================================
# EXISTING DATA PROJECT
# ============================================================

else:

    print(
        "\n[CACHE] Loading existing dataset..."
    )

    df = pd.read_parquet(
        CACHE_PATH
    )


    if dataset_metadata:

        print(
            f"[CACHE] Table: "
            f"{dataset_metadata.get('full_table_name')}"
        )


    print(
        f"[CACHE] Shape: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )


# --------------------------------------------------
# Profile
# --------------------------------------------------

print(
    f"\nDataset loaded: "
    f"{df.shape[0]:,} rows × "
    f"{df.shape[1]:,} columns"
)

profiler = DatasetProfiler(df)

profiler.run()

# --------------------------------------------------
# Column configuration
# --------------------------------------------------

column_manager = ColumnManager(df)
use_existing = column_manager.load_existing()

if use_existing:
    excl_cols = column_manager.excl_cols
    cat_cols = column_manager.cat_cols
    num_cols = column_manager.num_cols
    bool_cols = column_manager.bool_cols
else:
    excl_cols, cat_cols, num_cols = (
        column_manager.prepare_for_eda()
    )
# --------------------------------------------------
# EDA
# --------------------------------------------------
categorical_eda = CategoricalEDA(
    df=df,
    cat_cols=cat_cols,
    num_cols=num_cols,
)
categorical_eda.run()
numerical_eda = NumericalEDA(
    df=df,
    num_cols=num_cols,
)
numerical_eda.run()
# --------------------------------------------------
# Finalize column configuration only if updating
# --------------------------------------------------
if not use_existing:
    (
        excl_cols,
        cat_cols,
        num_cols,
        bool_cols,
    ) = column_manager.finalize()

# --------------------------------------------------
# Data cleaning
# --------------------------------------------------

categorical_preprocessor = CategoricalPreprocessor(
    df=df,
    cat_cols=cat_cols,
    excl_cols=excl_cols
)
ordinal_cols, map_cols, onehot_cols = categorical_preprocessor.run()


numerical_preprocessor = NumericalPreprocessor(
    df=df,
    num_cols=num_cols,
)
log_cols, clip_cols = numerical_preprocessor.run()

# ============================================================
# CORRELATION / VIF
# ============================================================
correlation_analysis = CorrelationAnalysis(
    df=df,
    num_cols=num_cols,
    bool_cols=bool_cols,
    excl_cols=excl_cols,
)

target_col, corr, vif = correlation_analysis.run()

# ============================================================
# MODEL SETUP
# ============================================================
# ------------------------------------------------------------
# 1. Create target y
# ------------------------------------------------------------
y = df[target_col].copy()
# ------------------------------------------------------------
# 2. Determine problem type
# ------------------------------------------------------------
problem_type = confirm_problem_type(y)
# ------------------------------------------------------------
# 3. List available models
# ------------------------------------------------------------
models = get_available_models(
    problem_type
)
print("\n" + "=" * 60)
print("AVAILABLE MODELS")
print("=" * 60)
for name in models:
    print(f"  - {name}")
# ------------------------------------------------------------
# 4. Remove target from feature groups
# ------------------------------------------------------------
cat_cols = [
    col for col in cat_cols
    if col != target_col
]
num_cols = [
    col for col in num_cols
    if col != target_col
]
bool_cols = [
    col for col in bool_cols
    if col != target_col
]
# ------------------------------------------------------------
# 5. Create X
# ------------------------------------------------------------
feature_cols = (
    cat_cols
    + num_cols
    + bool_cols
)
X = df[feature_cols].copy()
print("\n" + "=" * 60)
print("MODEL DATA")
print("=" * 60)
print(f"\nTarget: {target_col}")
print(f"Problem type: {problem_type}")
print(
    f"X shape: {X.shape}"
)
print(
    f"y shape: {y.shape}"
)
print(
    f"\nNumber of features: {len(feature_cols)}"
)
# ============================================================
# SAFETY: PREPROCESSING COLUMNS MUST EXIST IN X
# ============================================================

valid_features = set(
    X.columns
)

cat_cols = [
    col for col in cat_cols
    if col in valid_features
]

num_cols = [
    col for col in num_cols
    if col in valid_features
]

bool_cols = [
    col for col in bool_cols
    if col in valid_features
]

ordinal_cols = [
    col for col in ordinal_cols
    if col in valid_features
]

map_cols = [
    col for col in map_cols
    if col in valid_features
]

onehot_cols = [
    col for col in onehot_cols
    if col in valid_features
]

log_cols = [
    col for col in log_cols
    if col in valid_features
]

clip_cols = [
    col for col in clip_cols
    if col in valid_features
]
# ------------------------------------------------------------
# 6. Train / Validation / Test split
# ------------------------------------------------------------
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) = split_data(
    X=X,
    y=y,
    problem_type=problem_type,
)

# ============================================================
# BUILD PREPROCESSOR
# ============================================================

preprocessor = build_preprocessor(
    cat_cols=cat_cols,
    num_cols=num_cols,
    bool_cols=bool_cols,
    ordinal_cols=ordinal_cols,
    map_cols=map_cols,
    onehot_cols=onehot_cols,
    log_cols=log_cols,
    clip_cols=clip_cols,
)


# ============================================================
# TEST PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("TESTING PREPROCESSING PIPELINE")
print("=" * 60)

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_val_processed = (
    preprocessor.transform(
        X_val
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


print(
    "\nRaw train shape:",
    X_train.shape,
)

print(
    "Processed train shape:",
    X_train_processed.shape,
)

print(
    "Processed validation shape:",
    X_val_processed.shape,
)

print(
    "Processed test shape:",
    X_test_processed.shape,
)

print("\nProcessed features:")
for col in X_train_processed.columns:
    print(f"  - {col}")
print("\nMissing values after preprocessing:")
print(X_train_processed.isnull().sum().sum())
print("\nInfinite values:")
print(
    np.isinf(
        X_train_processed.select_dtypes(
            include="number"
        )
    ).sum().sum()
)
# ============================================================

# BASELINE MODEL TRAINING

# ============================================================

results, fitted_models = (

    train_models(

        models=models,

        preprocessor=preprocessor,

        problem_type=problem_type,

        X_train=X_train,

        y_train=y_train,

        X_val=X_val,

        y_val=y_val,

    )

)

# ============================================================

# OPTIONAL DETAILED BASELINE EVALUATION

# ============================================================

print("\n" + "=" * 70)

print("DETAILED MODEL EVALUATION")

print("=" * 70)

print(

    "\nAvailable fitted models:"

)

for name in fitted_models:

    print(

        f"  - {name}"

    )

model_to_evaluate = input(

    "\nEnter model to evaluate in detail "

    "(Enter to skip):\n> "

).strip()

if model_to_evaluate:

    if (

        model_to_evaluate

        not in fitted_models

    ):

        print(

            "\n[EVALUATION] "

            f"Unknown model: "

            f"{model_to_evaluate}"

        )

    else:

        evaluator = (

            ModelEvaluator(

                problem_type=problem_type

            )

        )

        evaluator.evaluate(

            model_name=model_to_evaluate,

            pipeline=fitted_models[

                model_to_evaluate

            ],

            X_train=X_train,

            y_train=y_train,

            X_val=X_val,

            y_val=y_val,

        )

# ============================================================

# FINAL MODEL STATE

# ============================================================

final_model = None

final_model_name = None

final_metrics = None

final_metadata = None

tuned_model = None

model_to_tune = None

# ============================================================

# HYPERPARAMETER TUNING

# ============================================================

print("\n" + "=" * 70)

print("HYPERPARAMETER TUNING")

print("=" * 70)

print(

    "\nAvailable models:"

)

for name in models:

    print(

        f"  - {name}"

    )

model_to_tune = input(

    "\nEnter model to tune "

    "(Enter to skip):\n> "

).strip()

if model_to_tune:

    if (

        model_to_tune

        not in models

    ):

        print(

            "\n[TUNING] "

            f"Unknown model: "

            f"{model_to_tune}"

        )

        model_to_tune = None

    else:

        (

            tuned_model,

            tuning_results,

            tuning_search,

        ) = tune_model(

            model_name=model_to_tune,

            model=models[

                model_to_tune

            ],

            preprocessor=preprocessor,

            problem_type=problem_type,

            X_train=X_train,

            y_train=y_train,

        )

# ============================================================

# FINALIZE TUNED MODEL

# ============================================================

if (

    model_to_tune

    and tuned_model is not None

):

    finalize = input(

        f"\nFinalize {model_to_tune} "

        "using the selected "

        "hyperparameters? (y/N):\n> "

    ).strip().lower()

    if finalize in [

        "y",

        "yes",

    ]:

        (

            final_model,

            final_metrics,

            final_metadata,

        ) = finalize_model(

            model=tuned_model,

            model_name=model_to_tune,

            problem_type=problem_type,

            target_col=target_col,

            X_train=X_train,

            y_train=y_train,

            X_val=X_val,

            y_val=y_val,

            X_test=X_test,

            y_test=y_test,

        )

        final_model_name = (

            model_to_tune

        )

# ============================================================

# LOAD EXISTING FINAL MODEL

#

# If the user skipped tuning/finalization, or did not finalize

# during this run, attempt to load the previously finalized

# artifact.

# ============================================================

if final_model is None:

    (

        loaded_model,

        loaded_model_name,

        loaded_metadata,

    ) = load_final_model()

    if loaded_model is not None:

        # ----------------------------------------------------

        # Safety check:

        # Make sure saved target matches current project target

        # ----------------------------------------------------

        saved_target = (

            loaded_metadata.get(

                "target"

            )

        )

        if (

            saved_target

            and saved_target

            != target_col

        ):

            print(

                "\n[MODEL] Existing finalized "

                "model target does not match "

                "the current target."

            )

            print(

                f"Saved target:   "

                f"{saved_target}"

            )

            print(

                f"Current target: "

                f"{target_col}"

            )

            print(

                "\nExisting final model will "

                "not be used."

            )

        else:

            final_model = (

                loaded_model

            )

            final_model_name = (

                loaded_model_name

            )

            final_metadata = (

                loaded_metadata

            )
# ============================================================
# CREATE RUN ARTIFACT DIRECTORY
# ============================================================
run_logger = None
if final_model is not None:

    run_logger = RunLogger(
        model_name=final_model_name,
        target_col=target_col,
    )

    run_logger.start()

    run_dir = (
        run_logger.run_dir
    )

    model_output_dir = (
        run_logger.model_dir
    )

    evaluation_output_dir = (
        run_logger.evaluation_dir
    )

    interpretation_output_dir = (
        run_logger.interpretation_dir
    )

# ============================================================
# COPY FINAL MODEL ARTIFACTS INTO RUN
# ============================================================

model_files = [
    f"{final_model_name}_final.joblib",
    "model_metadata.json",
    f"{final_model_name}_test_predictions.csv",
]

for filename in model_files:

    source = os.path.join(
        "artifacts/models",
        filename,
    )

    destination = os.path.join(
        model_output_dir,
        filename,
    )

    if os.path.exists(source):

        shutil.copy2(
            source,
            destination,
        )

# ============================================================
# MODEL INTERPRETATION
# ============================================================
if final_model is not None:
    print("\n" + "=" * 70)
    print("MODEL INTERPRETATION")
    print("=" * 70)
    print(
        f"\nFinal model: "
        f"{final_model_name}"
    )
    run_interpretation = input(
        "\nRun model interpretation? "
        "(Y/n):\n> "
    ).strip().lower()
    if run_interpretation in [
        "",
        "y",
        "yes",
    ]:
        interpreter = ModelInterpreter(
            output_dir=interpretation_output_dir
        )
        # ====================================================
        # LINEAR / RIDGE COEFFICIENTS
        # ====================================================
        if final_model_name in [
            "linear_regression",
            "ridge",
        ]:
            coef_df = (
                interpreter
                .linear_coefficients(
                    pipeline=final_model,
                    model_name=(
                        final_model_name
                    ),
                )
            )
        # ====================================================
        # PERMUTATION IMPORTANCE
        # ====================================================
        permutation_df = (
            interpreter
            .permutation_importance_plot(
                pipeline=final_model,
                model_name=(
                    final_model_name
                ),
                X=X_val,
                y=y_val,
                problem_type=(
                    problem_type
                ),
            )
        )

        # ============================================================
        # SHAP
        # ============================================================

        run_shap = input(
            "\nRun SHAP interpretation? "
            "(Y/n):\n> "
        ).strip().lower()

        if run_shap in [
            "",
            "y",
            "yes",
        ]:

            print("\n" + "=" * 70)
            print("STARTING SHAP ANALYSIS")
            print("=" * 70)

            print(
                "\n[SHAP] Calculating SHAP values "
                "and global importance..."
            )

            (
                shap_values,
                shap_importance,
                X_shap,
            ) = interpreter.shap_analysis(
                pipeline=final_model,
                model_name=final_model_name,
                X=X_val,

                # Start smaller while debugging.
                # Increase to 1000/2000 later if needed.
                max_samples=500,

                top_n=15,
            )

            print(
                "\n[SHAP] Global SHAP analysis complete."
            )

            # ========================================================
            # SHAP RELATIONSHIP PLOTS
            # ========================================================

            print("\n" + "=" * 70)
            print("SHAP FEATURE RELATIONSHIPS")
            print("=" * 70)

            print(
                "\n[SHAP] Creating relationship plots "
                "for top 5 features..."
            )

            interpreter.shap_dependence_plots(
                shap_values=shap_values,
                shap_importance=shap_importance,
                model_name=final_model_name,
                top_n=None,
                ncols=3,
            )

            print(
                "\n[SHAP] Relationship plots complete."
            )
            # ========================================================
            # INDIVIDUAL SHAP EXPLANATION
            # ========================================================
            explain_row = input(
                "\nEnter SHAP row number "
                "for individual explanation "
                "(Enter for row 0):\n> "
            ).strip()
            if explain_row == "":
                explain_row = 0
            else:
                try:
                    explain_row = int(
                        explain_row
                    )
                except ValueError:
                    print(
                        "\n[SHAP] Invalid row. "
                        "Using row 0."
                    )
                    explain_row = 0

            # --------------------------------------------------------
            # Validate requested row
            # --------------------------------------------------------

            if (
                explain_row < 0
                or explain_row >= len(shap_values)
            ):

                print(
                    "\n[SHAP] Row outside available "
                    "SHAP sample. Using row 0."
                )

                explain_row = 0

            print(
                f"\n[SHAP] Creating individual "
                f"explanation for row "
                f"{explain_row}..."
            )

            interpreter.shap_individual_explanation(
                shap_values=shap_values,
                row=explain_row,
            )

            print(
                "\n[SHAP] Individual explanation complete."
            )

            print("\n" + "=" * 70)
            print("SHAP ANALYSIS COMPLETE")
            print("=" * 70)

# ============================================================
# PIPELINE COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ML WORKFLOW COMPLETE")
print("=" * 70)

if final_model is not None:

    print(
        f"\nFinal model: "
        f"{final_model_name}"
    )

    print(
        "Artifact: "
        f"artifacts/models/"
        f"{final_model_name}_final.joblib"
    )

    print(
        "Metadata: "
        "artifacts/models/"
        "model_metadata.json"
    )

    if run_logger is not None:
        run_logger.stop()