import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# FINALIZE MODEL
# ============================================================

def finalize_model(
    model,
    model_name,
    problem_type,
    target_col,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    output_dir="artifacts/models",
):
    """
    Finalize a selected model.

    Workflow:
        1. Combine train + validation
        2. Clone selected tuned pipeline
        3. Refit preprocessing + model
        4. Evaluate once on untouched test set
        5. Save complete sklearn pipeline
        6. Save model metadata

    Returns
    -------
    final_model
        Fitted sklearn Pipeline.

    metrics
        Dictionary containing final test metrics.

    metadata
        Dictionary describing the finalized model.
    """

    print("\n" + "=" * 70)
    print(f"FINAL MODEL: {model_name}")
    print("=" * 70)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ========================================================
    # COMBINE TRAIN + VALIDATION
    # ========================================================

    X_final_train = pd.concat(
        [
            X_train,
            X_val,
        ],
        axis=0,
    )

    y_final_train = pd.concat(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )

    print(
        f"\nFinal training rows: "
        f"{len(X_final_train):,}"
    )

    print(
        f"Final test rows: "
        f"{len(X_test):,}"
    )

    # ========================================================
    # CLONE SELECTED PIPELINE
    # ========================================================

    final_model = clone(
        model
    )

    # ========================================================
    # REFIT
    # ========================================================

    print(
        "\nRefitting final pipeline "
        "using train + validation..."
    )

    final_model.fit(
        X_final_train,
        y_final_train,
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    y_pred = final_model.predict(
        X_test
    )

    # ========================================================
    # REGRESSION METRICS
    # ========================================================

    if problem_type == "regression":

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred,
            )
        )

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        r2 = r2_score(
            y_test,
            y_pred,
        )

        metrics = {
            "Test_RMSE": float(rmse),
            "Test_MAE": float(mae),
            "Test_R2": float(r2),
        }

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    elif problem_type == "classification":

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        metrics = {
            "Test_Accuracy": float(
                accuracy
            ),
            "Test_Precision": float(
                precision
            ),
            "Test_Recall": float(
                recall
            ),
            "Test_F1": float(
                f1
            ),
        }

        # ----------------------------------------------------
        # Probability-based metrics
        # ----------------------------------------------------

        if hasattr(
            final_model,
            "predict_proba",
        ):

            y_prob = (
                final_model.predict_proba(
                    X_test
                )
            )

            classes = (
                final_model
                .named_steps["model"]
                .classes_
            )

            # -----------------------------------------------
            # Binary classification
            # -----------------------------------------------

            if len(classes) == 2:

                positive_class = (
                    classes[1]
                )

                y_binary = (
                    np.asarray(y_test)
                    == positive_class
                ).astype(int)

                positive_prob = (
                    y_prob[:, 1]
                )

                metrics[
                    "Test_ROC_AUC"
                ] = float(
                    roc_auc_score(
                        y_binary,
                        positive_prob,
                    )
                )

                metrics[
                    "Test_PR_AUC"
                ] = float(
                    average_precision_score(
                        y_binary,
                        positive_prob,
                    )
                )

                metrics[
                    "Test_LogLoss"
                ] = float(
                    log_loss(
                        y_test,
                        y_prob,
                        labels=classes,
                    )
                )

            # -----------------------------------------------
            # Multiclass classification
            # -----------------------------------------------

            elif len(classes) > 2:

                metrics[
                    "Test_ROC_AUC"
                ] = float(
                    roc_auc_score(
                        y_test,
                        y_prob,
                        labels=classes,
                        multi_class="ovr",
                        average="weighted",
                    )
                )

                metrics[
                    "Test_LogLoss"
                ] = float(
                    log_loss(
                        y_test,
                        y_prob,
                        labels=classes,
                    )
                )

    else:

        raise ValueError(
            "problem_type must be "
            "'regression' or "
            "'classification'."
        )

    # ========================================================
    # PRINT TEST RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():

        print(
            f"{metric:<22}: "
            f"{value:.4f}"
        )

    # ========================================================
    # SAVE COMPLETE PIPELINE
    # ========================================================

    model_filename = (
        f"{model_name}_final.joblib"
    )

    model_path = os.path.join(
        output_dir,
        model_filename,
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print(
        "\n[MODEL] Saved final model to:"
    )

    print(
        model_path
    )

    # ========================================================
    # GET FINAL MODEL PARAMETERS
    # ========================================================

    estimator = (
        final_model
        .named_steps["model"]
    )

    if hasattr(
        estimator,
        "get_params",
    ):

        model_params = (
            estimator.get_params()
        )

    else:

        model_params = {}

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    feature_columns = (
        X_final_train
        .columns
        .tolist()
    )

    # ========================================================
    # BUILD METADATA
    # ========================================================

    metadata = {
        "model_name": model_name,
        "problem_type": problem_type,
        "target": target_col,

        "model_file": model_filename,

        "training_rows": int(
            len(X_final_train)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "raw_feature_count": int(
            X_final_train.shape[1]
        ),

        "raw_features": (
            feature_columns
        ),

        "metrics": metrics,

        "model_params": (
            model_params
        ),
    }

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = os.path.join(
        output_dir,
        "model_metadata.json",
    )

    with open(
        metadata_path,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=str,
        )

    print(
        "\n[MODEL] Saved metadata to:"
    )

    print(
        metadata_path
    )

    # ========================================================
    # SAVE FINAL TEST PREDICTIONS
    # ========================================================

    predictions = pd.DataFrame(
        {
            "actual": np.asarray(
                y_test
            ),
            "prediction": np.asarray(
                y_pred
            ),
        },
        index=X_test.index,
    )

    prediction_path = os.path.join(
        output_dir,
        f"{model_name}_test_predictions.csv",
    )

    predictions.to_csv(
        prediction_path,
        index=True,
    )

    print(
        "\n[MODEL] Saved test predictions to:"
    )

    print(
        prediction_path
    )

    return (
        final_model,
        metrics,
        metadata,
    )


# ============================================================
# LOAD FINALIZED MODEL
# ============================================================

def load_final_model(
    output_dir="artifacts/models",
):
    """
    Load the currently finalized model using model_metadata.json.

    Returns
    -------
    final_model
    final_model_name
    metadata

    If no finalized model exists:
        returns (None, None, None)
    """

    metadata_path = os.path.join(
        output_dir,
        "model_metadata.json",
    )

    if not os.path.exists(
        metadata_path
    ):

        print(
            "\n[MODEL] No finalized model metadata found."
        )

        return (
            None,
            None,
            None,
        )

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    with open(
        metadata_path,
        "r",
    ) as f:

        metadata = json.load(
            f
        )

    final_model_name = (
        metadata["model_name"]
    )

    model_filename = (
        metadata.get(
            "model_file",
            f"{final_model_name}_final.joblib",
        )
    )

    model_path = os.path.join(
        output_dir,
        model_filename,
    )

    # --------------------------------------------------------
    # Check model artifact
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        print(
            "\n[MODEL] Metadata found, "
            "but model artifact is missing:"
        )

        print(
            model_path
        )

        return (
            None,
            final_model_name,
            metadata,
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    final_model = joblib.load(
        model_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADED FINAL MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nModel:  "
        f"{final_model_name}"
    )

    print(
        f"Target: "
        f"{metadata.get('target')}"
    )

    print(
        f"Type:   "
        f"{metadata.get('problem_type')}"
    )

    print(
        f"Path:   "
        f"{model_path}"
    )

    return (
        final_model,
        final_model_name,
        metadata,
    )