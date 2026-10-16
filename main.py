import os
import shutil
import io
import sys

from src.reporting.causal_report import (
    TeeOutput,
    CausalMarkdownReport,
)

from src.modeling.models import (
    get_available_models,
)

from src.utils.run_logger import (
    RunLogger,
)

from src.workflows.data_workflow import (
    load_data_project,
)

from src.workflows.analysis_workflow import (
    choose_analysis_type,
    configure_causal_analysis,
)

from src.workflows.eda_workflow import (
    run_profile,
    run_eda,
    run_correlation,
)

from src.workflows.preprocessing_workflow import (
    configure_preprocessing,
    build_model_preprocessor,
    test_preprocessor,
)

from src.workflows.modeling_workflow import (
    prepare_model_data,
    create_model_split,
    run_model_training,
    evaluate_baseline_models,
    tune_and_finalize,
)

from src.workflows.interpretation_workflow import (
    run_interpretation,
)

from src.workflows.prediction_workflow import (
    generate_predictions,
)

from src.workflows.causal_workflow import (
    run_causal_workflow,
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_ID = "project-business-prd"
DATASET_ID = "prism_studio"


# ============================================================
# DATA
# ============================================================

df, dataset_metadata = (
    load_data_project(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
    )
)


# ============================================================
# PROFILE
# ============================================================

run_profile(
    df
)


# ============================================================
# ANALYSIS OBJECTIVE
# ============================================================

analysis_type = (
    choose_analysis_type()
)

causal_config = None
causal_features = None

if analysis_type in [
    "causal",
    "both",
]:

    (
        causal_config,
        causal_features,
    ) = configure_causal_analysis(
        df=df,
        analysis_type=analysis_type,
    )


# ============================================================
# PREDICTION WORKFLOW
# ============================================================

if analysis_type in [
    "prediction",
    "both",
]:

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    column_config = (
        run_eda(
            df
        )
    )

    # --------------------------------------------------------
    # PREPROCESSING CONFIGURATION
    # --------------------------------------------------------

    preprocessing_config = (
        configure_preprocessing(
            df=df,
            column_config=column_config,
        )
    )

    # --------------------------------------------------------
    # CORRELATION / VIF / TARGET
    # --------------------------------------------------------

    (
        target_col,
        corr,
        vif,
    ) = run_correlation(
        df=df,
        column_config=column_config,
    )

    # --------------------------------------------------------
    # MODEL DATA
    # --------------------------------------------------------

    (
        X,
        y,
        problem_type,
        feature_cols,
        column_config,
    ) = prepare_model_data(

        df=df,

        target_col=target_col,

        column_config=column_config,
    )

    models = get_available_models(
        problem_type
    )

    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)

    for name in models:

        print(
            f"  - {name}"
        )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_model_split(

        X=X,

        y=y,

        problem_type=problem_type,
    )

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = (
        build_model_preprocessor(

            X=X,

            column_config=column_config,

            preprocessing_config=(
                preprocessing_config
            ),
        )
    )

    # --------------------------------------------------------
    # TEST PREPROCESSOR
    # --------------------------------------------------------

    test_preprocessor(

        preprocessor=preprocessor,

        X_train=X_train,

        X_val=X_val,

        X_test=X_test,
    )

    # --------------------------------------------------------
    # BASELINE MODELS
    # --------------------------------------------------------

    (
        results,
        fitted_models,
    ) = run_model_training(

        models=models,

        preprocessor=preprocessor,

        problem_type=problem_type,

        X_train=X_train,

        y_train=y_train,

        X_val=X_val,

        y_val=y_val,
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluate_baseline_models(

        fitted_models=fitted_models,

        problem_type=problem_type,

        X_train=X_train,

        y_train=y_train,

        X_val=X_val,

        y_val=y_val,
    )

    # --------------------------------------------------------
    # TUNE + FINALIZE
    # --------------------------------------------------------

    (
        final_model,
        final_model_name,
        final_metrics,
        final_metadata,
    ) = tune_and_finalize(

        models=models,

        preprocessor=preprocessor,

        problem_type=problem_type,

        target_col=target_col,

        X_train=X_train,

        y_train=y_train,

        X_val=X_val,

        y_val=y_val,

        X_test=X_test,

        y_test=y_test,
    )

    # ========================================================
    # RUN LOGGER
    # ========================================================

    run_logger = None

    if final_model is not None:

        run_logger = RunLogger(

            model_name=final_model_name,

            target_col=target_col,
        )

        run_logger.start()

        interpretation_output_dir = (
            run_logger.interpretation_dir
        )

        model_output_dir = (
            run_logger.model_dir
        )

        # ----------------------------------------------------
        # Copy model artifacts
        # ----------------------------------------------------

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

            if os.path.exists(
                source
            ):

                shutil.copy2(
                    source,
                    destination,
                )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        run_interpretation(

            final_model=final_model,

            final_model_name=(
                final_model_name
            ),

            problem_type=problem_type,

            X_val=X_val,

            y_val=y_val,

            output_dir=(
                interpretation_output_dir
            ),
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        generate_predictions(

            df=df,

            feature_cols=feature_cols,

            final_model=final_model,

            final_model_name=(
                final_model_name
            ),

            problem_type=problem_type,

            target_col=target_col,

            project_id=PROJECT_ID,

            dataset_id=DATASET_ID,
        )


# ============================================================
# CAUSAL WORKFLOW
# ============================================================

causal_results = None

if analysis_type in [
    "causal",
    "both",
]:

    buffer = io.StringIO()

    original_stdout = sys.stdout

    sys.stdout = TeeOutput(
        original_stdout,
        buffer,
    )

    try:

        causal_results = run_causal_workflow(
            df=df,
            causal_config=causal_config,
        )

    finally:

        sys.stdout = original_stdout


    terminal_output = buffer.getvalue()


    reporter = CausalMarkdownReport()

    report_path = reporter.save(
        terminal_output=terminal_output,
    )

    causal_results["report_path"] = report_path

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ML WORKFLOW COMPLETE")
print("=" * 70)


if (
    analysis_type in [
        "prediction",
        "both",
    ]
    and final_model is not None
):

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