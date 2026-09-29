import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
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
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
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
    },
}


# ============================================================
# SCORING
# ============================================================

def get_scoring(problem_type):

    if problem_type == "regression":

        return {
            "rmse":
                "neg_root_mean_squared_error",

            "mae":
                "neg_mean_absolute_error",

            "r2":
                "r2",
        }

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
    }


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
        problem_type
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    rows = []

    for metric in scoring:

        train_values = (
            scores[f"train_{metric}"]
        )

        val_values = (
            scores[f"test_{metric}"]
        )

        # sklearn returns negative error metrics
        if metric in [
            "rmse",
            "mae",
        ]:

            train_values = (
                -train_values
            )

            val_values = (
                -val_values
            )

        rows.append({
            "metric": metric,

            "train_mean":
                np.mean(train_values),

            "train_std":
                np.std(train_values),

            "cv_mean":
                np.mean(val_values),

            "cv_std":
                np.std(val_values),
        })

    result = pd.DataFrame(
        rows
    )

    print(
        result
        .round(4)
        .to_string(index=False)
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
    print(f"MODEL TUNING: {model_name}")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Get parameter grid
    # --------------------------------------------------------

    if problem_type == "regression":

        param_grid = (
            REGRESSION_PARAM_GRIDS.get(
                model_name,
                {},
            )
        )

        scoring = (
            "neg_root_mean_squared_error"
        )

    else:

        param_grid = (
            CLASSIFICATION_PARAM_GRIDS.get(
                model_name,
                {},
            )
        )

        scoring = "f1_weighted"

    # --------------------------------------------------------
    # No meaningful hyperparameters
    # --------------------------------------------------------

    if not param_grid:

        print(
            f"\n{model_name} has no meaningful "
            "hyperparameter grid configured."
        )

        print(
            "\nRunning cross-validation instead..."
        )

        cv_results = run_cross_validation(
            pipeline=pipeline,
            problem_type=problem_type,
            X_train=X_train,
            y_train=y_train,
        )

        print(
            "\nFitting model on full training set..."
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
    # Grid search
    # --------------------------------------------------------

    print("\nParameter grid:")

    for param, values in param_grid.items():

        print(
            f"  {param}: {values}"
        )

    print(
        "\nRunning GridSearchCV..."
    )

    start = time.time()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    elapsed = (
        time.time()
        - start
    )

    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    print(
        f"\nSearch time: "
        f"{elapsed:.2f} seconds"
    )

    print("\nBest parameters:")

    for key, value in (
        search.best_params_.items()
    ):

        print(
            f"  {key}: {value}"
        )

    if problem_type == "regression":

        best_score = (
            -search.best_score_
        )

        print(
            f"\nBest CV RMSE: "
            f"{best_score:.4f}"
        )

    else:

        print(
            f"\nBest CV F1: "
            f"{search.best_score_:.4f}"
        )

    # --------------------------------------------------------
    # Show top search results
    # --------------------------------------------------------

    search_results = pd.DataFrame(
        search.cv_results_
    )

    if problem_type == "regression":

        search_results[
            "CV_RMSE"
        ] = (
            -search_results[
                "mean_test_score"
            ]
        )

        columns = [
            "params",
            "CV_RMSE",
            "std_test_score",
            "mean_train_score",
        ]

        search_results = (
            search_results
            .sort_values(
                "CV_RMSE"
            )
        )

    else:

        columns = [
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
        ]

        search_results = (
            search_results
            .sort_values(
                "mean_test_score",
                ascending=False,
            )
        )

    print("\nTop parameter combinations:")

    print(
        search_results[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    return (
        search.best_estimator_,
        search_results,
        search,
    )