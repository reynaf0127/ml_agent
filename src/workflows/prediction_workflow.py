import os
import shutil

import numpy as np
from scipy import sparse

from src.features.columns import ColumnManager
from src.profiling.categorical_eda import CategoricalEDA
from src.profiling.numerical_eda import NumericalEDA
from src.profiling.correlation import CorrelationAnalysis

from src.preprocessing.categorical import CategoricalPreprocessor
from src.preprocessing.numerical import NumericalPreprocessor
from src.preprocessing.pipeline import build_preprocessor

from src.modeling.problem import confirm_problem_type
from src.modeling.models import get_available_models
from src.modeling.split import split_data
from src.modeling.train import train_models
from src.modeling.evaluate import ModelEvaluator
from src.modeling.tune import tune_model
from src.modeling.finalize import (
    finalize_model,
    load_final_model,
)
from src.modeling.interpret import ModelInterpreter
from src.modeling.predict import ModelPredictor

from src.data.bigquery import BigQueryDataSource
from src.utils.run_logger import RunLogger


# ============================================================
# PREDICTION WORKFLOW
# ============================================================

def run_prediction_workflow(
    df,
    project_id,
    dataset_id,
):
    """
    Run the complete supervised prediction workflow.

    Workflow:
        Column configuration
        -> EDA
        -> preprocessing configuration
        -> correlation / VIF
        -> target selection
        -> problem detection
        -> train/validation/test split
        -> preprocessing pipeline
        -> baseline models
        -> optional detailed evaluation
        -> hyperparameter tuning
        -> final model
        -> interpretation
        -> full prediction table
        -> optional BigQuery upload
    """

    print("\n" + "=" * 70)
    print("PREDICTION WORKFLOW")
    print("=" * 70)

    # ========================================================
    # COLUMN CONFIGURATION
    # ========================================================

    column_manager = ColumnManager(df)

    use_existing = (
        column_manager.load_existing()
    )

    if use_existing:

        excl_cols = (
            column_manager.excl_cols
        )

        cat_cols = (
            column_manager.cat_cols
        )

        num_cols = (
            column_manager.num_cols
        )

        bool_cols = (
            column_manager.bool_cols
        )

    else:

        (
            excl_cols,
            cat_cols,
            num_cols,
        ) = column_manager.prepare_for_eda()

        # bool_cols is finalized later
        bool_cols = []

    # ========================================================
    # EDA
    # ========================================================

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

    # ========================================================
    # FINALIZE COLUMN CONFIGURATION
    # ========================================================

    if not use_existing:

        (
            excl_cols,
            cat_cols,
            num_cols,
            bool_cols,
        ) = column_manager.finalize()

    # ========================================================
    # CATEGORICAL PREPROCESSING CONFIGURATION
    # ========================================================

    categorical_preprocessor = (
        CategoricalPreprocessor(
            df=df,
            cat_cols=cat_cols,
            excl_cols=excl_cols,
        )
    )

    (
        ordinal_cols,
        map_cols,
        onehot_cols,
        hash_cols,
    ) = categorical_preprocessor.run()

    # ========================================================
    # NUMERICAL PREPROCESSING CONFIGURATION
    # ========================================================

    numerical_preprocessor = (
        NumericalPreprocessor(
            df=df,
            num_cols=num_cols,
        )
    )

    (
        log_cols,
        clip_cols,
    ) = numerical_preprocessor.run()

    # ========================================================
    # CORRELATION / VIF
    # ========================================================

    correlation_analysis = (
        CorrelationAnalysis(
            df=df,
            num_cols=num_cols,
            bool_cols=bool_cols,
            excl_cols=excl_cols,
        )
    )

    (
        target_col,
        corr,
        vif,
    ) = correlation_analysis.run()

    # ========================================================
    # TARGET
    # ========================================================

    y = df[target_col].copy()

    # ========================================================
    # PROBLEM TYPE
    # ========================================================

    problem_type = (
        confirm_problem_type(y)
    )

    # ========================================================
    # AVAILABLE MODELS
    # ========================================================

    models = get_available_models(
        problem_type
    )

    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)

    for name in models:
        print(
            f"  - {name}"
        )

    # ========================================================
    # REMOVE TARGET FROM FEATURE GROUPS
    # ========================================================

    cat_cols = [
        col
        for col in cat_cols
        if col != target_col
    ]

    num_cols = [
        col
        for col in num_cols
        if col != target_col
    ]

    bool_cols = [
        col
        for col in bool_cols
        if col != target_col
    ]

    # ========================================================
    # CREATE X
    # ========================================================

    feature_cols = (
        cat_cols
        + num_cols
        + bool_cols
    )

    # Remove accidental duplicates while preserving order
    feature_cols = list(
        dict.fromkeys(
            feature_cols
        )
    )

    X = (
        df[feature_cols]
        .copy()
    )

    print("\n" + "=" * 70)
    print("MODEL DATA")
    print("=" * 70)

    print(
        f"\nTarget: {target_col}"
    )

    print(
        f"Problem type: {problem_type}"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"\nNumber of features: "
        f"{len(feature_cols)}"
    )

    # ========================================================
    # SAFETY:
    # PREPROCESSING COLUMNS MUST EXIST IN X
    # ========================================================

    valid_features = set(
        X.columns
    )

    cat_cols = [
        col
        for col in cat_cols
        if col in valid_features
    ]

    num_cols = [
        col
        for col in num_cols
        if col in valid_features
    ]

    bool_cols = [
        col
        for col in bool_cols
        if col in valid_features
    ]

    ordinal_cols = [
        col
        for col in ordinal_cols
        if col in valid_features
    ]

    map_cols = [
        col
        for col in map_cols
        if col in valid_features
    ]

    onehot_cols = [
        col
        for col in onehot_cols
        if col in valid_features
    ]

    hash_cols = [
        col
        for col in hash_cols
        if col in valid_features
    ]

    log_cols = [
        col
        for col in log_cols
        if col in valid_features
    ]

    clip_cols = [
        col
        for col in clip_cols
        if col in valid_features
    ]

    # ========================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

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

    # ========================================================
    # BUILD PREPROCESSOR
    # ========================================================

    preprocessor = build_preprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        bool_cols=bool_cols,
        ordinal_cols=ordinal_cols,
        map_cols=map_cols,
        onehot_cols=onehot_cols,
        hash_cols=hash_cols,
        log_cols=log_cols,
        clip_cols=clip_cols,
        hash_n_features=256,
    )

    # ========================================================
    # TEST PREPROCESSING
    # ========================================================

    print("\n" + "=" * 70)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 70)

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

    # ========================================================
    # SHAPES
    # ========================================================

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

    # ========================================================
    # SPARSE / DENSE
    # ========================================================

    is_sparse = sparse.issparse(
        X_train_processed
    )

    print(
        "\nSparse matrix:",
        is_sparse,
    )

    # ========================================================
    # MISSING / INFINITE CHECK
    # ========================================================

    if is_sparse:

        values = (
            X_train_processed.data
        )

    else:

        values = np.asarray(
            X_train_processed
        )

    missing_count = int(
        np.isnan(values).sum()
    )

    infinite_count = int(
        np.isinf(values).sum()
    )

    print(
        "\nMissing values after preprocessing:",
        missing_count,
    )

    print(
        "Infinite values after preprocessing:",
        infinite_count,
    )

    if missing_count > 0:

        raise ValueError(
            "Preprocessing produced missing values. "
            "Resolve missing-value handling before "
            "model training."
        )

    if infinite_count > 0:

        raise ValueError(
            "Preprocessing produced infinite values."
        )

    # ========================================================
    # SPARSITY
    # ========================================================

    if is_sparse:

        total_cells = (
            X_train_processed.shape[0]
            * X_train_processed.shape[1]
        )

        nonzero_cells = (
            X_train_processed.nnz
        )

        density = (
            nonzero_cells
            / total_cells
        )

        print(
            "\nNon-zero values:",
            f"{nonzero_cells:,}",
        )

        print(
            "Matrix density:",
            f"{density:.4%}",
        )

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        print(
            "\nNumber of processed features:",
            len(feature_names),
        )

        print(
            "\nFirst 30 processed features:"
        )

        for feature in feature_names[:30]:

            print(
                f"  - {feature}"
            )

    except Exception as e:

        print(
            "\n[PREPROCESSING] "
            "Could not retrieve feature names:"
        )

        print(
            f"{e}"
        )

    # ========================================================
    # BASELINE MODELS
    # ========================================================

    (
        results,
        fitted_models,
    ) = train_models(
        models=models,
        preprocessor=preprocessor,
        problem_type=problem_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # ========================================================
    # OPTIONAL DETAILED EVALUATION
    # ========================================================

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

            evaluator = ModelEvaluator(
                problem_type=problem_type
            )

            evaluator.evaluate(
                model_name=model_to_evaluate,
                pipeline=(
                    fitted_models[
                        model_to_evaluate
                    ]
                ),
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

    # ========================================================
    # FINAL MODEL STATE
    # ========================================================

    final_model = None
    final_model_name = None
    final_metrics = None
    final_metadata = None

    tuned_model = None
    model_to_tune = None

    # ========================================================
    # HYPERPARAMETER TUNING
    # ========================================================

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

        if model_to_tune not in models:

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
                model=(
                    models[
                        model_to_tune
                    ]
                ),
                preprocessor=preprocessor,
                problem_type=problem_type,
                X_train=X_train,
                y_train=y_train,
            )

    # ========================================================
    # FINALIZE TUNED MODEL
    # ========================================================

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

    # ========================================================
    # LOAD EXISTING FINAL MODEL
    # ========================================================

    if final_model is None:

        (
            loaded_model,
            loaded_model_name,
            loaded_metadata,
        ) = load_final_model()

        if loaded_model is not None:

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
                    "\nExisting final model "
                    "will not be used."
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

    # ========================================================
    # NO FINAL MODEL
    # ========================================================

    if final_model is None:

        print(
            "\n[MODEL] No final model available."
        )

        print(
            "[MODEL] Prediction output and "
            "interpretation will be skipped."
        )

        return {
            "results": results,
            "fitted_models": fitted_models,
            "final_model": None,
            "final_model_name": None,
            "target_col": target_col,
            "problem_type": problem_type,
            "feature_cols": feature_cols,
        }

    # ========================================================
    # RUN LOGGER
    # ========================================================

    run_logger = RunLogger(
        model_name=final_model_name,
        target_col=target_col,
    )

    run_logger.start()

    model_output_dir = (
        run_logger.model_dir
    )

    interpretation_output_dir = (
        run_logger.interpretation_dir
    )

    # ========================================================
    # COPY FINAL MODEL ARTIFACTS
    # ========================================================

    model_files = [
        f"{final_model_name}_final.joblib",
        "model_metadata.json",
        (
            f"{final_model_name}_"
            "test_predictions.csv"
        ),
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

    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

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
            output_dir=(
                interpretation_output_dir
            )
        )

        # ----------------------------------------------------
        # COEFFICIENTS
        # ----------------------------------------------------

        if final_model_name in [
            "linear_regression",
            "ridge",
            "logistic_regression",
        ]:

            try:

                interpreter.linear_coefficients(
                    pipeline=final_model,
                    model_name=(
                        final_model_name
                    ),
                )

            except Exception as e:

                print(
                    "\n[INTERPRETATION] "
                    "Coefficient analysis failed:"
                )

                print(
                    e
                )

        # ----------------------------------------------------
        # PERMUTATION IMPORTANCE
        # ----------------------------------------------------

        try:

            interpreter.permutation_importance_plot(
                pipeline=final_model,
                model_name=final_model_name,
                X=X_val,
                y=y_val,
                problem_type=problem_type,
            )

        except Exception as e:

            print(
                "\n[INTERPRETATION] "
                "Permutation importance failed:"
            )

            print(
                e
            )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

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

            try:

                (
                    shap_values,
                    shap_importance,
                    X_shap,
                ) = interpreter.shap_analysis(
                    pipeline=final_model,
                    model_name=(
                        final_model_name
                    ),
                    X=X_val,
                    max_samples=500,
                    top_n=15,
                )

                print(
                    "\n[SHAP] Global SHAP "
                    "analysis complete."
                )

                # --------------------------------------------
                # RELATIONSHIP PLOTS
                # --------------------------------------------

                interpreter.shap_dependence_plots(
                    shap_values=shap_values,
                    shap_importance=(
                        shap_importance
                    ),
                    model_name=(
                        final_model_name
                    ),
                    top_n=None,
                    ncols=3,
                )

                # --------------------------------------------
                # INDIVIDUAL EXPLANATION
                # --------------------------------------------

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

                if (
                    explain_row < 0
                    or explain_row
                    >= len(shap_values)
                ):

                    print(
                        "\n[SHAP] Row outside "
                        "available SHAP sample. "
                        "Using row 0."
                    )

                    explain_row = 0

                interpreter.shap_individual_explanation(
                    shap_values=shap_values,
                    row=explain_row,
                )

                print("\n" + "=" * 70)
                print("SHAP ANALYSIS COMPLETE")
                print("=" * 70)

            except Exception as e:

                print(
                    "\n[SHAP] Analysis failed:"
                )

                print(
                    e
                )

    # ========================================================
    # FINAL PREDICTION OUTPUT
    # ========================================================

    prediction_df = generate_predictions(
        df=df,
        final_model=final_model,
        final_model_name=final_model_name,
        problem_type=problem_type,
        target_col=target_col,
        feature_cols=feature_cols,
        project_id=project_id,
        dataset_id=dataset_id,
    )

    # ========================================================
    # STOP LOGGER
    # ========================================================

    run_logger.stop()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDICTION WORKFLOW COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal model: "
        f"{final_model_name}"
    )

    print(
        "\nArtifact:"
    )

    print(
        f"artifacts/models/"
        f"{final_model_name}_final.joblib"
    )

    # ========================================================
    # RETURN WORKFLOW STATE
    # ========================================================

    return {
        "results": results,
        "fitted_models": fitted_models,
        "final_model": final_model,
        "final_model_name": final_model_name,
        "final_metrics": final_metrics,
        "final_metadata": final_metadata,
        "target_col": target_col,
        "problem_type": problem_type,
        "feature_cols": feature_cols,
        "prediction_df": prediction_df,
    }


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    df,
    final_model,
    final_model_name,
    problem_type,
    target_col,
    feature_cols,
    project_id,
    dataset_id,
):
    """
    Score the full loaded dataset using the finalized model.

    Supports:
        - classification
        - regression
        - local prediction output
        - optional BigQuery upload
    """

    if final_model is None:

        print(
            "\n[PREDICTION] "
            "No final model available."
        )

        return None

    print("\n" + "=" * 70)
    print("FINAL PREDICTION OUTPUT")
    print("=" * 70)

    run_prediction = input(
        "\nGenerate prediction table "
        "for the full modeling dataset? "
        "(Y/n):\n> "
    ).strip().lower()

    if run_prediction not in [
        "",
        "y",
        "yes",
    ]:

        print(
            "\n[PREDICTION] "
            "Prediction generation skipped."
        )

        return None

    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    missing_features = [
        col
        for col in feature_cols
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Prediction dataset is missing "
            "required model features:\n"
            f"{missing_features}"
        )

    # ========================================================
    # FULL MODEL MATRIX
    # ========================================================

    X_full = (
        df[feature_cols]
        .copy()
    )

    print(
        f"\n[PREDICTION] "
        f"Scoring {len(X_full):,} rows..."
    )

    # ========================================================
    # PREDICT
    # ========================================================

    predictor = ModelPredictor()

    prediction_df = (
        predictor.predict(
            pipeline=final_model,
            X=X_full,
            original_df=df,
            problem_type=problem_type,
            model_name=final_model_name,
            target_col=target_col,
        )
    )

    print(
        f"\n[PREDICTION] Generated "
        f"{len(prediction_df):,} predictions."
    )

    # ========================================================
    # SAVE LOCALLY
    # ========================================================

    prediction_path = (
        predictor.save(
            prediction_df=prediction_df,
            model_name=final_model_name,
            target_col=target_col,
        )
    )

    print(
        "\n[PREDICTION] Saved locally:"
    )

    print(
        prediction_path
    )

    # ========================================================
    # OPTIONAL BIGQUERY UPLOAD
    # ========================================================

    upload_bq = input(
        "\nUpload prediction table "
        "to BigQuery? (y/N):\n> "
    ).strip().lower()

    if upload_bq not in [
        "y",
        "yes",
    ]:

        return prediction_df

    # ========================================================
    # OUTPUT DATASET
    # ========================================================

    dataset_input = input(
        "\nBigQuery dataset "
        f"(Enter for '{dataset_id}'):\n> "
    ).strip()

    output_dataset = (
        dataset_input
        if dataset_input
        else dataset_id
    )

    # ========================================================
    # OUTPUT TABLE
    # ========================================================

    default_table = (
        f"predictions_"
        f"{final_model_name}_"
        f"{target_col}"
    )

    table_input = input(
        "\nBigQuery table name "
        f"(Enter for '{default_table}'):\n> "
    ).strip()

    output_table = (
        table_input
        if table_input
        else default_table
    )

    # ========================================================
    # BIGQUERY UPLOAD
    # ========================================================

    bq_output = (
        BigQueryDataSource()
    )

    table_id = (
        bq_output.upload_dataframe(
            df=prediction_df,
            project_id=project_id,
            dataset_id=output_dataset,
            table_name=output_table,
            write_disposition=(
                "WRITE_TRUNCATE"
            ),
        )
    )

    print(
        "\n[PREDICTION] "
        "BigQuery upload complete."
    )

    print(
        f"[PREDICTION] Table: "
        f"{table_id}"
    )

    return prediction_df