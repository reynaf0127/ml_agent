import time

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression metrics.
    """

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
    y_prob,
    classes,
):
    """
    Calculate classification metrics.

    Supports:
        binary classification
        multiclass classification
    """

    n_classes = len(classes)

    # --------------------------------------------------------
    # Binary classification
    # --------------------------------------------------------

    if n_classes == 2:

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            pos_label=classes[1],
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            pos_label=classes[1],
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            pos_label=classes[1],
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_true,
            y_prob[:, 1],
        )

        # Convert target to 0/1 relative to
        # the model's positive class.
        y_binary = (
            np.asarray(y_true)
            == classes[1]
        ).astype(int)

        pr_auc = average_precision_score(
            y_binary,
            y_prob[:, 1],
        )

        loss = log_loss(
            y_true,
            y_prob,
            labels=classes,
        )

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "LogLoss": loss,
        }

    # --------------------------------------------------------
    # Multiclass classification
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        labels=classes,
        multi_class="ovr",
        average="weighted",
    )

    # Binarize multiclass target for PR-AUC
    y_binary = label_binarize(
        y_true,
        classes=classes,
    )

    pr_auc = average_precision_score(
        y_binary,
        y_prob,
        average="weighted",
    )

    loss = log_loss(
        y_true,
        y_prob,
        labels=classes,
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "LogLoss": loss,
    }


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    models,
    preprocessor,
    problem_type,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Train all candidate models.

    Each model receives its own cloned preprocessing pipeline.

    Returns
    -------
    results_df
        Train/validation metrics for every model.

    fitted_models
        Dictionary containing fitted sklearn pipelines.
    """

    results = []
    fitted_models = {}

    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODELS")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nTraining: {name}")

        # ----------------------------------------------------
        # Complete ML pipeline
        # ----------------------------------------------------

        pipeline = Pipeline([
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "model",
                clone(model),
            ),
        ])

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        start_time = time.time()

        pipeline.fit(
            X_train,
            y_train,
        )

        train_seconds = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        train_pred = pipeline.predict(
            X_train
        )

        val_pred = pipeline.predict(
            X_val
        )

        # ====================================================
        # REGRESSION
        # ====================================================

        if problem_type == "regression":

            train_metrics = regression_metrics(
                y_train,
                train_pred,
            )

            val_metrics = regression_metrics(
                y_val,
                val_pred,
            )

            result = {
                "model": name,

                "Train_RMSE":
                    train_metrics["RMSE"],

                "Val_RMSE":
                    val_metrics["RMSE"],

                "RMSE_Gap":
                    val_metrics["RMSE"]
                    - train_metrics["RMSE"],

                "Train_MAE":
                    train_metrics["MAE"],

                "Val_MAE":
                    val_metrics["MAE"],

                "Train_R2":
                    train_metrics["R2"],

                "Val_R2":
                    val_metrics["R2"],

                "R2_Gap":
                    train_metrics["R2"]
                    - val_metrics["R2"],

                "Train_Seconds":
                    train_seconds,
            }

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        elif problem_type == "classification":

            if not hasattr(
                pipeline,
                "predict_proba",
            ):
                raise ValueError(
                    f"Model '{name}' does not support "
                    "predict_proba(), which is required "
                    "for ROC-AUC, PR-AUC and log loss."
                )

            train_prob = (
                pipeline.predict_proba(
                    X_train
                )
            )

            val_prob = (
                pipeline.predict_proba(
                    X_val
                )
            )

            classes = (
                pipeline
                .named_steps["model"]
                .classes_
            )

            train_metrics = (
                classification_metrics(
                    y_train,
                    train_pred,
                    train_prob,
                    classes,
                )
            )

            val_metrics = (
                classification_metrics(
                    y_val,
                    val_pred,
                    val_prob,
                    classes,
                )
            )

            result = {
                "model": name,

                "Train_Accuracy":
                    train_metrics["Accuracy"],

                "Val_Accuracy":
                    val_metrics["Accuracy"],

                "Train_Precision":
                    train_metrics["Precision"],

                "Val_Precision":
                    val_metrics["Precision"],

                "Train_Recall":
                    train_metrics["Recall"],

                "Val_Recall":
                    val_metrics["Recall"],

                "Train_F1":
                    train_metrics["F1"],

                "Val_F1":
                    val_metrics["F1"],

                "F1_Gap":
                    train_metrics["F1"]
                    - val_metrics["F1"],

                "Train_ROC_AUC":
                    train_metrics["ROC_AUC"],

                "Val_ROC_AUC":
                    val_metrics["ROC_AUC"],

                "ROC_AUC_Gap":
                    train_metrics["ROC_AUC"]
                    - val_metrics["ROC_AUC"],

                "Train_PR_AUC":
                    train_metrics["PR_AUC"],

                "Val_PR_AUC":
                    val_metrics["PR_AUC"],

                "Train_LogLoss":
                    train_metrics["LogLoss"],

                "Val_LogLoss":
                    val_metrics["LogLoss"],

                "Train_Seconds":
                    train_seconds,
            }

        else:

            raise ValueError(
                "problem_type must be "
                "'regression' or 'classification'."
            )

        results.append(result)

        fitted_models[name] = pipeline

        print(
            f"Finished in "
            f"{train_seconds:.2f} seconds."
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if problem_type == "regression":

        results_df = (
            results_df
            .sort_values(
                "Val_RMSE",
                ascending=True,
            )
            .reset_index(drop=True)
        )

    else:

        results_df = (
            results_df
            .sort_values(
                "Val_F1",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    return (
        results_df,
        fitted_models,
    )