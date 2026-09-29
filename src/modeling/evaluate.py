import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats

from sklearn.base import clone

from sklearn.calibration import calibration_curve

from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from sklearn.model_selection import (
    learning_curve,
)


# ============================================================
# MODEL EVALUATOR
# ============================================================

class ModelEvaluator:

    def __init__(
        self,
        problem_type,
        output_dir="artifacts/model_evaluation",
    ):

        self.problem_type = problem_type
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # ========================================================
    # SAVE / SHOW
    # ========================================================

    def _finish_plot(
        self,
        filename,
    ):

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

        print(
            f"[EVALUATION] Saved: {path}"
        )

        plt.close()

    # ========================================================
    # REGRESSION
    # ========================================================

    def evaluate_regression(
        self,
        model_name,
        pipeline,
        X_train,
        y_train,
        X_val,
        y_val,
    ):

        print("\n" + "=" * 70)
        print(
            f"REGRESSION DIAGNOSTICS: "
            f"{model_name}"
        )
        print("=" * 70)

        train_pred = pipeline.predict(
            X_train
        )

        val_pred = pipeline.predict(
            X_val
        )

        residuals = (
            np.asarray(y_val)
            - np.asarray(val_pred)
        )

        # ----------------------------------------------------
        # 1. Train vs validation RMSE
        # ----------------------------------------------------

        train_rmse = np.sqrt(
            np.mean(
                (
                    np.asarray(y_train)
                    - np.asarray(train_pred)
                ) ** 2
            )
        )

        val_rmse = np.sqrt(
            np.mean(
                (
                    np.asarray(y_val)
                    - np.asarray(val_pred)
                ) ** 2
            )
        )

        plt.figure(
            figsize=(7, 5)
        )

        sns.barplot(
            x=["Train", "Validation"],
            y=[train_rmse, val_rmse],
        )

        plt.ylabel("RMSE")

        plt.title(
            f"{model_name}: Train vs Validation RMSE"
        )

        self._finish_plot(
            f"{model_name}_train_val_rmse.png"
        )

        # ----------------------------------------------------
        # 2. Actual vs predicted
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 6)
        )

        sns.scatterplot(
            x=y_val,
            y=val_pred,
            alpha=0.5,
        )

        minimum = min(
            np.min(y_val),
            np.min(val_pred),
        )

        maximum = max(
            np.max(y_val),
            np.max(val_pred),
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        plt.title(
            f"{model_name}: Actual vs Predicted"
        )

        self._finish_plot(
            f"{model_name}_actual_vs_predicted.png"
        )

        # ----------------------------------------------------
        # 3. Residual vs predicted
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 6)
        )

        sns.scatterplot(
            x=val_pred,
            y=residuals,
            alpha=0.5,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted")
        plt.ylabel("Residual")

        plt.title(
            f"{model_name}: Residuals vs Predicted"
        )

        self._finish_plot(
            f"{model_name}_residuals_vs_predicted.png"
        )

        # ----------------------------------------------------
        # 4. Residual distribution
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        sns.histplot(
            residuals,
            kde=True,
        )

        plt.xlabel("Residual")

        plt.title(
            f"{model_name}: Residual Distribution"
        )

        self._finish_plot(
            f"{model_name}_residual_distribution.png"
        )

        # ----------------------------------------------------
        # 5. Q-Q plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 6)
        )

        stats.probplot(
            residuals,
            dist="norm",
            plot=plt,
        )

        plt.title(
            f"{model_name}: Residual Q-Q Plot"
        )

        self._finish_plot(
            f"{model_name}_qq_plot.png"
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def evaluate_classification(
        self,
        model_name,
        pipeline,
        X_train,
        y_train,
        X_val,
        y_val,
    ):

        print("\n" + "=" * 70)
        print(
            f"CLASSIFICATION DIAGNOSTICS: "
            f"{model_name}"
        )
        print("=" * 70)

        y_pred = pipeline.predict(
            X_val
        )

        y_prob = pipeline.predict_proba(
            X_val
        )

        classes = (
            pipeline
            .named_steps["model"]
            .classes_
        )

        n_classes = len(classes)

        # ----------------------------------------------------
        # 1. Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_val,
            y_pred,
            labels=classes,
        )

        plt.figure(
            figsize=(7, 6)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.title(
            f"{model_name}: Confusion Matrix"
        )

        self._finish_plot(
            f"{model_name}_confusion_matrix.png"
        )

        # ----------------------------------------------------
        # Binary classification plots
        # ----------------------------------------------------

        if n_classes == 2:

            positive_class = classes[1]

            y_binary = (
                np.asarray(y_val)
                == positive_class
            ).astype(int)

            positive_prob = y_prob[:, 1]

            # ------------------------------------------------
            # 2. ROC curve
            # ------------------------------------------------

            fpr, tpr, _ = roc_curve(
                y_binary,
                positive_prob,
            )

            roc_auc = auc(
                fpr,
                tpr,
            )

            plt.figure(
                figsize=(7, 6)
            )

            plt.plot(
                fpr,
                tpr,
                label=f"AUC = {roc_auc:.3f}",
            )

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
            )

            plt.xlabel(
                "False Positive Rate"
            )

            plt.ylabel(
                "True Positive Rate"
            )

            plt.title(
                f"{model_name}: ROC Curve"
            )

            plt.legend()

            self._finish_plot(
                f"{model_name}_roc_curve.png"
            )

            # ------------------------------------------------
            # 3. Precision-recall curve
            # ------------------------------------------------

            precision, recall, _ = (
                precision_recall_curve(
                    y_binary,
                    positive_prob,
                )
            )

            pr_auc = auc(
                recall,
                precision,
            )

            plt.figure(
                figsize=(7, 6)
            )

            plt.plot(
                recall,
                precision,
                label=f"PR AUC = {pr_auc:.3f}",
            )

            plt.xlabel("Recall")
            plt.ylabel("Precision")

            plt.title(
                f"{model_name}: Precision-Recall Curve"
            )

            plt.legend()

            self._finish_plot(
                f"{model_name}_precision_recall.png"
            )

            # ------------------------------------------------
            # 4. Calibration curve
            # ------------------------------------------------

            prob_true, prob_pred = (
                calibration_curve(
                    y_binary,
                    positive_prob,
                    n_bins=10,
                    strategy="quantile",
                )
            )

            plt.figure(
                figsize=(7, 6)
            )

            plt.plot(
                prob_pred,
                prob_true,
                marker="o",
            )

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
            )

            plt.xlabel(
                "Mean predicted probability"
            )

            plt.ylabel(
                "Observed positive rate"
            )

            plt.title(
                f"{model_name}: Calibration Curve"
            )

            self._finish_plot(
                f"{model_name}_calibration.png"
            )

        else:

            print(
                "\n[EVALUATION] Multiclass target detected."
            )

            print(
                "Confusion matrix generated. "
                "ROC/PR curves are skipped in V1; "
                "we can add one-vs-rest curves next."
            )

    # ========================================================
    # LEARNING CURVE
    # ========================================================

    def plot_learning_curve(
        self,
        model_name,
        pipeline,
        X_train,
        y_train,
    ):

        print(
            f"\nGenerating learning curve "
            f"for {model_name}..."
        )

        if self.problem_type == "regression":

            scoring = (
                "neg_root_mean_squared_error"
            )

        else:

            scoring = "f1_weighted"

        train_sizes = np.linspace(
            0.2,
            1.0,
            5,
        )

        (
            sizes,
            train_scores,
            val_scores,
        ) = learning_curve(
            estimator=clone(pipeline),
            X=X_train,
            y=y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
        )

        # ----------------------------------------------------
        # Regression:
        # sklearn returns negative RMSE
        # ----------------------------------------------------

        if self.problem_type == "regression":

            train_mean = (
                -train_scores.mean(axis=1)
            )

            val_mean = (
                -val_scores.mean(axis=1)
            )

            ylabel = "RMSE"

        else:

            train_mean = (
                train_scores.mean(axis=1)
            )

            val_mean = (
                val_scores.mean(axis=1)
            )

            ylabel = "F1 weighted"

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            sizes,
            train_mean,
            marker="o",
            label="Training",
        )

        plt.plot(
            sizes,
            val_mean,
            marker="o",
            label="Cross-validation",
        )

        plt.xlabel(
            "Training examples"
        )

        plt.ylabel(
            ylabel
        )

        plt.title(
            f"{model_name}: Learning Curve"
        )

        plt.legend()

        self._finish_plot(
            f"{model_name}_learning_curve.png"
        )

    # ========================================================
    # MAIN ROUTER
    # ========================================================

    def evaluate(
        self,
        model_name,
        pipeline,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_curve_plot=True,
    ):

        if self.problem_type == "regression":

            self.evaluate_regression(
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

        elif self.problem_type == "classification":

            self.evaluate_classification(
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

        else:

            raise ValueError(
                "problem_type must be "
                "'regression' or 'classification'."
            )

        if learning_curve_plot:

            self.plot_learning_curve(
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
            )