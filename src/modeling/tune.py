import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    StratifiedKFold,
    KFold,
)
from sklearn.pipeline import Pipeline


# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================

REGRESSION_PARAM_GRIDS = {

    # LinearRegression has very little meaningful tuning.
    "linear_regression": {},

    "ridge": {
        "model__alpha": [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
        ],
    },

    "decision_tree": {
        "model__max_depth": [
            3,
            5,
            10,
            20,
            None,
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
            20,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            5,
            10,
            20,
        ],
    },

    "random_forest": {
        "model__n_estimators": [
            100,
            300,
            500,
        ],

        "model__max_depth": [
            5,
            10,
            20,
            None,
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            5,
            10,
        ],

        "model__max_features": [
            "sqrt",
            0.5,
            1.0,
        ],
    },

    "gradient_boosting": {
        "model__n_estimators": [
            100,
            200,
            300,
        ],

        "model__learning_rate": [
            0.01,
            0.05,
            0.1,
        ],

        "model__max_depth": [
            2,
            3,
            5,
        ],

        "model__min_samples_leaf": [
            1,
            5,
            10,
        ],
    },
}


CLASSIFICATION_PARAM_GRIDS = {

    "logistic_regression": {

        "model__C": [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],

        "model__class_weight": [
            None,
            "balanced",
        ],
    },

    "decision_tree": {

        "model__max_depth": [
            3,
            5,
            10,
            20,
            None,
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
            20,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            5,
            10,
            20,
        ],

        "model__class_weight": [
            None,
            "balanced",
        ],
    },

    "random_forest": {

        "model__n_estimators": [
            100,
            300,
            500,
        ],

        "model__max_depth": [
            5,
            10,
            20,
            None,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            5,
            10,
        ],

        "model__max_features": [
            "sqrt",
            0.5,
            1.0,
        ],

        "model__class_weight": [
            None,
            "balanced",
        ],
    },

    "gradient_boosting": {

        "model__n_estimators": [
            100,
            200,
            300,
        ],

        "model__learning_rate": [
            0.01,
            0.05,
            0.1,
        ],

        "model__max_depth": [
            2,
            3,
            5,
        ],
    },
}


# ============================================================
# CV SPLITTER
# ============================================================

def get_cv_splitter(
    problem_type,
):

    if problem_type == "classification":

        return StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


# ============================================================
# SCORING
# ============================================================

def get_scoring(
    problem_type,
    y=None,
):

    if problem_type == "regression":

        return {
            "rmse":
                "neg_root_mean_squared_error",

            "mae":
                "neg_mean_absolute_error",

            "r2":
                "r2",
        }

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    n_classes = None

    if y is not None:

        n_classes = (
            pd.Series(y)
            .nunique()
        )

    # --------------------------------------------------------
    # Binary classification
    # --------------------------------------------------------

    if (
        n_classes is None
        or n_classes == 2
    ):

        return {
            "accuracy":
                "accuracy",

            "precision":
                "precision",

            "recall":
                "recall",

            "f1":
                "f1",

            "roc_auc":
                "roc_auc",

            # PR-AUC / Average Precision
            "pr_auc":
                "average_precision",

            "neg_log_loss":
                "neg_log_loss",
        }

    # --------------------------------------------------------
    # Multiclass classification
    #
    # Average precision is less straightforward for generic
    # multiclass problems, so use weighted metrics here.
    # --------------------------------------------------------

    return {
        "accuracy":
            "accuracy",

        "precision":
            "precision_weighted",

        "recall":
            "recall_weighted",

        "f1":
            "f1_weighted",

        "roc_auc":
            "roc_auc_ovr_weighted",

        "neg_log_loss":
            "neg_log_loss",
    }


# ============================================================
# TUNING OBJECTIVE
# ============================================================

def get_tuning_objective(
    problem_type,
    y_train,
):

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    if problem_type == "regression":

        return (
            "neg_root_mean_squared_error",
            "RMSE",
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    n_classes = (
        pd.Series(y_train)
        .nunique()
    )

    if n_classes == 2:

        # Especially appropriate for imbalanced binary
        # classification such as the Criteo conversion model.
        return (
            "average_precision",
            "PR-AUC",
        )

    # Generic multiclass fallback
    return (
        "roc_auc_ovr_weighted",
        "ROC-AUC",
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    pipeline,
    problem_type,
    X_train,
    y_train,
):

    print("\n" + "=" * 70)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 70)

    scoring = get_scoring(
        problem_type=problem_type,
        y=y_train,
    )

    cv = get_cv_splitter(
        problem_type
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    rows = []

    for metric in scoring:

        train_values = (
            scores[
                f"train_{metric}"
            ]
        )

        val_values = (
            scores[
                f"test_{metric}"
            ]
        )

        # ----------------------------------------------------
        # sklearn returns negative values for loss/error
        # metrics because higher score must always be better.
        # Convert them back for human-readable output.
        # ----------------------------------------------------

        if metric in [
            "rmse",
            "mae",
            "neg_log_loss",
        ]:

            train_values = (
                -train_values
            )

            val_values = (
                -val_values
            )

        display_metric = metric

        if metric == "pr_auc":
            display_metric = "PR_AUC"

        elif metric == "roc_auc":
            display_metric = "ROC_AUC"

        elif metric == "neg_log_loss":
            display_metric = "LogLoss"

        elif metric == "rmse":
            display_metric = "RMSE"

        elif metric == "mae":
            display_metric = "MAE"

        elif metric == "r2":
            display_metric = "R2"

        rows.append({
            "metric":
                display_metric,

            "train_mean":
                np.mean(
                    train_values
                ),

            "train_std":
                np.std(
                    train_values
                ),

            "cv_mean":
                np.mean(
                    val_values
                ),

            "cv_std":
                np.std(
                    val_values
                ),
        })

    result = pd.DataFrame(
        rows
    )

    print(
        result
        .round(4)
        .to_string(
            index=False
        )
    )

    return result


# ============================================================
# MODEL TUNING
# ============================================================

def tune_model(
    model_name,
    model,
    preprocessor,
    problem_type,
    X_train,
    y_train,
):

    print("\n" + "=" * 70)
    print(
        f"MODEL TUNING: "
        f"{model_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Build fresh pipeline
    # --------------------------------------------------------

    pipeline = Pipeline([
        (
            "preprocessor",
            clone(
                preprocessor
            ),
        ),

        (
            "model",
            clone(
                model
            ),
        ),
    ])

    # --------------------------------------------------------
    # Parameter grid
    # --------------------------------------------------------

    if problem_type == "regression":

        param_grid = (
            REGRESSION_PARAM_GRIDS
            .get(
                model_name,
                {},
            )
        )

    elif problem_type == "classification":

        param_grid = (
            CLASSIFICATION_PARAM_GRIDS
            .get(
                model_name,
                {},
            )
        )

    else:

        raise ValueError(
            "Unknown problem type: "
            f"{problem_type}"
        )

    # --------------------------------------------------------
    # Determine tuning objective
    # --------------------------------------------------------

    (
        scoring,
        metric_name,
    ) = get_tuning_objective(
        problem_type=problem_type,
        y_train=y_train,
    )

    print(
        "\nTuning objective:"
    )

    print(
        f"  {metric_name}"
    )

    # --------------------------------------------------------
    # CV strategy
    # --------------------------------------------------------

    cv = get_cv_splitter(
        problem_type
    )

    # --------------------------------------------------------
    # No meaningful hyperparameters
    # --------------------------------------------------------

    if not param_grid:

        print(
            f"\n{model_name} has no meaningful "
            "hyperparameter grid configured."
        )

        print(
            "\nRunning cross-validation "
            "instead..."
        )

        cv_results = (
            run_cross_validation(
                pipeline=pipeline,
                problem_type=problem_type,
                X_train=X_train,
                y_train=y_train,
            )
        )

        print(
            "\nFitting model on full "
            "training set..."
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        return (
            pipeline,
            cv_results,
            None,
        )

    # --------------------------------------------------------
    # Print grid
    # --------------------------------------------------------

    print(
        "\nParameter grid:"
    )

    total_candidates = 1

    for (
        param,
        values,
    ) in param_grid.items():

        print(
            f"  {param}: "
            f"{values}"
        )

        total_candidates *= (
            len(values)
        )

    print(
        "\nNumber of parameter "
        f"combinations: "
        f"{total_candidates:,}"
    )

    print(
        "\nRunning GridSearchCV..."
    )

    # --------------------------------------------------------
    # Grid search
    # --------------------------------------------------------

    start = time.time()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        refit=True,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
    )

    elapsed = (
        time.time()
        - start
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    print(
        f"\nSearch time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        "\nBest parameters:"
    )

    for (
        key,
        value,
    ) in search.best_params_.items():

        print(
            f"  {key}: "
            f"{value}"
        )

    # --------------------------------------------------------
    # Human-readable best score
    # --------------------------------------------------------

    if problem_type == "regression":

        best_score = (
            -search.best_score_
        )

    else:

        best_score = (
            search.best_score_
        )

    print(
        f"\nBest CV "
        f"{metric_name}: "
        f"{best_score:.4f}"
    )

    # ========================================================
    # SEARCH RESULT TABLE
    # ========================================================

    search_results = (
        pd.DataFrame(
            search.cv_results_
        )
    )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    if problem_type == "regression":

        search_results[
            "CV_RMSE"
        ] = (
            -search_results[
                "mean_test_score"
            ]
        )

        search_results[
            "Train_RMSE"
        ] = (
            -search_results[
                "mean_train_score"
            ]
        )

        search_results[
            "RMSE_Gap"
        ] = (
            search_results[
                "CV_RMSE"
            ]
            -
            search_results[
                "Train_RMSE"
            ]
        )

        display_columns = [
            "params",
            "CV_RMSE",
            "Train_RMSE",
            "RMSE_Gap",
            "std_test_score",
        ]

        search_results = (
            search_results
            .sort_values(
                "CV_RMSE",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Binary classification: PR-AUC
    # --------------------------------------------------------

    elif metric_name == "PR-AUC":

        search_results[
            "CV_PR_AUC"
        ] = (
            search_results[
                "mean_test_score"
            ]
        )

        search_results[
            "Train_PR_AUC"
        ] = (
            search_results[
                "mean_train_score"
            ]
        )

        search_results[
            "PR_AUC_Gap"
        ] = (
            search_results[
                "Train_PR_AUC"
            ]
            -
            search_results[
                "CV_PR_AUC"
            ]
        )

        display_columns = [
            "params",
            "CV_PR_AUC",
            "Train_PR_AUC",
            "PR_AUC_Gap",
            "std_test_score",
        ]

        search_results = (
            search_results
            .sort_values(
                "CV_PR_AUC",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Other classification: ROC-AUC
    # --------------------------------------------------------

    else:

        search_results[
            "CV_ROC_AUC"
        ] = (
            search_results[
                "mean_test_score"
            ]
        )

        search_results[
            "Train_ROC_AUC"
        ] = (
            search_results[
                "mean_train_score"
            ]
        )

        search_results[
            "ROC_AUC_Gap"
        ] = (
            search_results[
                "Train_ROC_AUC"
            ]
            -
            search_results[
                "CV_ROC_AUC"
            ]
        )

        display_columns = [
            "params",
            "CV_ROC_AUC",
            "Train_ROC_AUC",
            "ROC_AUC_Gap",
            "std_test_score",
        ]

        search_results = (
            search_results
            .sort_values(
                "CV_ROC_AUC",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Print top combinations
    # --------------------------------------------------------

    print(
        "\nTop parameter combinations:"
    )

    print(
        search_results[
            display_columns
        ]
        .head(10)
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # ADDITIONAL CV METRICS FOR BEST MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "BEST MODEL CROSS-VALIDATION METRICS"
    )
    print("=" * 70)

    best_cv_results = (
        run_cross_validation(
            pipeline=(
                search.best_estimator_
            ),
            problem_type=problem_type,
            X_train=X_train,
            y_train=y_train,
        )
    )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        search.best_estimator_,
        search_results,
        search,
    )