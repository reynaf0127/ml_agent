import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


class UpliftModel:
    """
    T-Learner uplift / CATE model.

    Estimates:

        mu1(X) = E[Y | T=1, X]
        mu0(X) = E[Y | T=0, X]

    Individual treatment effect:

        CATE(X) = mu1(X) - mu0(X)

    For binary outcomes:

        uplift =
            P(Y=1 | T=1, X)
            -
            P(Y=1 | T=0, X)

    Two separate outcome models are trained:

        Treated model:
            X -> Y using T=1 observations

        Control model:
            X -> Y using T=0 observations

    Important:
    Only pre-treatment features should be supplied.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/uplift",
        random_state=42,
        max_iter=2000,
    ):

        self.output_dir = output_dir
        self.random_state = random_state
        self.max_iter = max_iter

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        self.treated_model = None
        self.control_model = None

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate(
        self,
        X,
        treatment,
        outcome,
    ):

        if not isinstance(
            X,
            pd.DataFrame,
        ):

            raise TypeError(
                "X must be a pandas DataFrame."
            )

        treatment = pd.Series(
            treatment,
            index=X.index,
        )

        outcome = pd.Series(
            outcome,
            index=X.index,
        )

        if treatment.isna().any():

            raise ValueError(
                "Treatment contains missing values."
            )

        if outcome.isna().any():

            raise ValueError(
                "Outcome contains missing values."
            )

        treatment_values = set(
            treatment
            .unique()
            .tolist()
        )

        if not treatment_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "Treatment must be binary 0/1. "
                f"Found: {treatment_values}"
            )

        outcome_values = set(
            outcome
            .unique()
            .tolist()
        )

        if not outcome_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "Current uplift implementation "
                "requires binary outcome 0/1. "
                f"Found: {outcome_values}"
            )

        if treatment.nunique() < 2:

            raise ValueError(
                "Both treatment and control "
                "observations are required."
            )

    # ========================================================
    # BUILD OUTCOME MODEL
    # ========================================================

    def _build_model(
        self,
        preprocessor,
    ):

        outcome_model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=None,
        )

        pipeline = Pipeline([
            (
                "preprocessor",
                clone(
                    preprocessor
                ),
            ),
            (
                "model",
                outcome_model,
            ),
        ])

        return pipeline

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X_train,
        treatment_train,
        outcome_train,
        preprocessor,
        sample_weight=None,
    ):

        print("\n" + "=" * 70)
        print("UPLIFT MODEL: T-LEARNER")
        print("=" * 70)

        self._validate(
            X=X_train,
            treatment=treatment_train,
            outcome=outcome_train,
        )

        treatment_train = pd.Series(
            np.asarray(
                treatment_train
            ),
            index=X_train.index,
        )

        outcome_train = pd.Series(
            np.asarray(
                outcome_train
            ),
            index=X_train.index,
        )

        treated_mask = (
            treatment_train == 1
        )

        control_mask = (
            treatment_train == 0
        )

        print(
            f"\nTraining rows: "
            f"{len(X_train):,}"
        )

        print(
            f"Treated rows: "
            f"{treated_mask.sum():,}"
        )

        print(
            f"Control rows: "
            f"{control_mask.sum():,}"
        )

        print(
            "\nTreated outcome rate: "
            f"{outcome_train[treated_mask].mean():.4%}"
        )

        print(
            "Control outcome rate: "
            f"{outcome_train[control_mask].mean():.4%}"
        )

        # ----------------------------------------------------
        # Build independent pipelines
        # ----------------------------------------------------

        self.treated_model = (
            self._build_model(
                preprocessor=preprocessor,
            )
        )

        self.control_model = (
            self._build_model(
                preprocessor=preprocessor,
            )
        )

        # ====================================================
        # OPTIONAL PROPENSITY WEIGHTS
        # ====================================================

        treated_weights = None
        control_weights = None

        if sample_weight is not None:

            sample_weight = pd.Series(
                np.asarray(
                    sample_weight,
                    dtype=float,
                ),
                index=X_train.index,
            )

            treated_weights = (
                sample_weight[
                    treated_mask
                ]
                .to_numpy()
            )

            control_weights = (
                sample_weight[
                    control_mask
                ]
                .to_numpy()
            )

            print(
                "\n[UPLIFT] Using sample weights "
                "during outcome-model training."
            )

        # ====================================================
        # FIT TREATED MODEL
        # ====================================================

        print(
            "\n[UPLIFT] Fitting treated "
            "outcome model..."
        )

        treated_fit_params = {}

        if treated_weights is not None:

            treated_fit_params[
                "model__sample_weight"
            ] = treated_weights

        self.treated_model.fit(
            X_train.loc[
                treated_mask
            ],
            outcome_train.loc[
                treated_mask
            ],
            **treated_fit_params,
        )

        # ====================================================
        # FIT CONTROL MODEL
        # ====================================================

        print(
            "[UPLIFT] Fitting control "
            "outcome model..."
        )

        control_fit_params = {}

        if control_weights is not None:

            control_fit_params[
                "model__sample_weight"
            ] = control_weights

        self.control_model.fit(
            X_train.loc[
                control_mask
            ],
            outcome_train.loc[
                control_mask
            ],
            **control_fit_params,
        )

        print(
            "\n[UPLIFT] T-Learner fitted."
        )

        return self

    # ========================================================
    # POSITIVE CLASS PROBABILITY
    # ========================================================

    @staticmethod
    def _positive_probability(
        pipeline,
        X,
    ):

        probabilities = (
            pipeline.predict_proba(
                X
            )
        )

        classes = (
            pipeline.named_steps[
                "model"
            ]
            .classes_
        )

        positive_index = np.where(
            classes == 1
        )[0]

        if len(
            positive_index
        ) != 1:

            raise ValueError(
                "Could not identify "
                "outcome class 1."
            )

        return np.asarray(
            probabilities[
                :,
                positive_index[0],
            ],
            dtype=float,
        )

    # ========================================================
    # POTENTIAL OUTCOMES
    # ========================================================

    def predict_potential_outcomes(
        self,
        X,
    ):

        if (
            self.treated_model is None
            or
            self.control_model is None
        ):

            raise ValueError(
                "Uplift model has not "
                "been fitted."
            )

        mu1 = (
            self._positive_probability(
                pipeline=(
                    self.treated_model
                ),
                X=X,
            )
        )

        mu0 = (
            self._positive_probability(
                pipeline=(
                    self.control_model
                ),
                X=X,
            )
        )

        return (
            mu1,
            mu0,
        )

    # ========================================================
    # PREDICT UPLIFT
    # ========================================================

    def predict_uplift(
        self,
        X,
    ):

        (
            mu1,
            mu0,
        ) = (
            self.predict_potential_outcomes(
                X=X
            )
        )

        uplift = (
            mu1
            - mu0
        )

        return {
            "mu1":
                mu1,

            "mu0":
                mu0,

            "uplift":
                uplift,
        }

    # ========================================================
    # OUTCOME MODEL METRICS
    # ========================================================

    def _outcome_metrics(
        self,
        y_true,
        probability,
    ):

        y_true = np.asarray(
            y_true
        )

        probability = np.asarray(
            probability
        )

        metrics = {
            "rows":
                len(y_true),

            "outcome_rate":
                float(
                    np.mean(
                        y_true
                    )
                ),

            "predicted_rate":
                float(
                    np.mean(
                        probability
                    )
                ),
        }

        # ----------------------------------------------------
        # Some small subsets may contain only one class.
        # ----------------------------------------------------

        if len(
            np.unique(
                y_true
            )
        ) >= 2:

            metrics[
                "roc_auc"
            ] = float(
                roc_auc_score(
                    y_true,
                    probability,
                )
            )

            metrics[
                "pr_auc"
            ] = float(
                average_precision_score(
                    y_true,
                    probability,
                )
            )

            metrics[
                "log_loss"
            ] = float(
                log_loss(
                    y_true,
                    probability,
                    labels=[
                        0,
                        1,
                    ],
                )
            )

        else:

            metrics[
                "roc_auc"
            ] = np.nan

            metrics[
                "pr_auc"
            ] = np.nan

            metrics[
                "log_loss"
            ] = np.nan

        return metrics

    # ========================================================
    # EVALUATE OBSERVED OUTCOMES
    # ========================================================

    def evaluate_outcome_models(
        self,
        X,
        treatment,
        outcome,
        split_name,
    ):

        treatment = np.asarray(
            treatment
        )

        outcome = np.asarray(
            outcome
        )

        predictions = (
            self.predict_uplift(
                X=X
            )
        )

        mu1 = predictions[
            "mu1"
        ]

        mu0 = predictions[
            "mu0"
        ]

        rows = []

        # ----------------------------------------------------
        # Treated observations
        # ----------------------------------------------------

        treated_mask = (
            treatment == 1
        )

        if treated_mask.sum() > 0:

            treated_metrics = (
                self._outcome_metrics(
                    y_true=(
                        outcome[
                            treated_mask
                        ]
                    ),
                    probability=(
                        mu1[
                            treated_mask
                        ]
                    ),
                )
            )

            rows.append({
                "split":
                    split_name,

                "model":
                    "treated_outcome_model",

                **treated_metrics,
            })

        # ----------------------------------------------------
        # Control observations
        # ----------------------------------------------------

        control_mask = (
            treatment == 0
        )

        if control_mask.sum() > 0:

            control_metrics = (
                self._outcome_metrics(
                    y_true=(
                        outcome[
                            control_mask
                        ]
                    ),
                    probability=(
                        mu0[
                            control_mask
                        ]
                    ),
                )
            )

            rows.append({
                "split":
                    split_name,

                "model":
                    "control_outcome_model",

                **control_metrics,
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # CREATE SCORING TABLE
    # ========================================================

    def create_scoring_table(
        self,
        X,
        treatment,
        outcome,
        groups=None,
        propensity=None,
        original_df=None,
    ):

        predictions = (
            self.predict_uplift(
                X=X
            )
        )

        if original_df is not None:

            result = (
                original_df
                .loc[
                    X.index
                ]
                .copy()
            )

        else:

            result = (
                X.copy()
            )

        result[
            "observed_treatment"
        ] = np.asarray(
            treatment
        )

        result[
            "observed_outcome"
        ] = np.asarray(
            outcome
        )

        result[
            "predicted_outcome_treated"
        ] = predictions[
            "mu1"
        ]

        result[
            "predicted_outcome_control"
        ] = predictions[
            "mu0"
        ]

        result[
            "uplift_score"
        ] = predictions[
            "uplift"
        ]

        if groups is not None:

            result[
                "causal_group"
            ] = np.asarray(
                groups
            )

        if propensity is not None:

            result[
                "propensity_score"
            ] = np.asarray(
                propensity
            )

        return result

    # ========================================================
    # UPLIFT SUMMARY
    # ========================================================

    def summarize_uplift(
        self,
        score_df,
    ):

        uplift = (
            score_df[
                "uplift_score"
            ]
            .astype(float)
        )

        summary = pd.DataFrame([
            {
                "observations":
                    len(
                        uplift
                    ),

                "mean_uplift":
                    uplift.mean(),

                "std_uplift":
                    uplift.std(),

                "min_uplift":
                    uplift.min(),

                "q01":
                    uplift.quantile(
                        0.01
                    ),

                "q05":
                    uplift.quantile(
                        0.05
                    ),

                "q10":
                    uplift.quantile(
                        0.10
                    ),

                "q25":
                    uplift.quantile(
                        0.25
                    ),

                "median":
                    uplift.quantile(
                        0.50
                    ),

                "q75":
                    uplift.quantile(
                        0.75
                    ),

                "q90":
                    uplift.quantile(
                        0.90
                    ),

                "q95":
                    uplift.quantile(
                        0.95
                    ),

                "q99":
                    uplift.quantile(
                        0.99
                    ),

                "max_uplift":
                    uplift.max(),

                "positive_uplift_pct":
                    (
                        uplift
                        > 0
                    ).mean(),

                "negative_uplift_pct":
                    (
                        uplift
                        < 0
                    ).mean(),
            }
        ])

        return summary

    # ========================================================
    # UPLIFT DISTRIBUTION PLOT
    # ========================================================

    def plot_uplift_distribution(
        self,
        score_df,
        split_name,
    ):

        uplift = (
            score_df[
                "uplift_score"
            ]
            .astype(float)
            .to_numpy()
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            uplift,
            bins=60,
            alpha=0.75,
        )

        plt.axvline(
            0,
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "Estimated Individual Treatment Effect"
        )

        plt.ylabel(
            "Observations"
        )

        plt.title(
            "Estimated Uplift Distribution "
            f"({split_name})"
        )

        plt.grid(
            alpha=0.2
        )

        path = os.path.join(
            self.output_dir,
            f"uplift_distribution_"
            f"{split_name}.png",
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "[UPLIFT] Saved uplift distribution:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # UPLIFT DECILES
    # ========================================================

    def uplift_deciles(
        self,
        score_df,
        n_bins=10,
    ):
        """
        Rank observations from highest predicted uplift
        to lowest predicted uplift.

        This is descriptive only.

        Formal causal uplift evaluation is implemented
        separately using Qini / uplift curves.
        """

        result = (
            score_df
            .copy()
            .sort_values(
                "uplift_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        n_bins = min(
            n_bins,
            len(result),
        )

        if n_bins < 1:

            return pd.DataFrame()

        result[
            "uplift_decile"
        ] = (
            pd.qcut(
                np.arange(
                    len(result)
                ),
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

        rows = []

        for decile, subset in (
            result.groupby(
                "uplift_decile"
            )
        ):

            treated = (
                subset[
                    "observed_treatment"
                ]
                == 1
            )

            control = (
                subset[
                    "observed_treatment"
                ]
                == 0
            )

            treated_rate = (
                subset.loc[
                    treated,
                    "observed_outcome",
                ]
                .mean()
            )

            control_rate = (
                subset.loc[
                    control,
                    "observed_outcome",
                ]
                .mean()
            )

            rows.append({
                "uplift_decile":
                    int(
                        decile
                    ),

                "observations":
                    len(
                        subset
                    ),

                "mean_predicted_uplift":
                    subset[
                        "uplift_score"
                    ]
                    .mean(),

                "treated_observations":
                    int(
                        treated.sum()
                    ),

                "control_observations":
                    int(
                        control.sum()
                    ),

                "treated_outcome_rate":
                    treated_rate,

                "control_outcome_rate":
                    control_rate,

                "observed_rate_difference":
                    (
                        treated_rate
                        - control_rate
                    ),
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    def save_models(
        self,
    ):

        if (
            self.treated_model is None
            or
            self.control_model is None
        ):

            raise ValueError(
                "No fitted uplift models "
                "available."
            )

        treated_path = os.path.join(
            self.output_dir,
            "treated_outcome_model.joblib",
        )

        control_path = os.path.join(
            self.output_dir,
            "control_outcome_model.joblib",
        )

        joblib.dump(
            self.treated_model,
            treated_path,
        )

        joblib.dump(
            self.control_model,
            control_path,
        )

        print(
            "\n[UPLIFT] Saved treated model:"
            f"\n{treated_path}"
        )

        print(
            "[UPLIFT] Saved control model:"
            f"\n{control_path}"
        )

        return {
            "treated_model":
                treated_path,

            "control_model":
                control_path,
        }

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        preprocessor,
        split_results,
        propensity_results=None,
        original_df=None,
        use_training_weights=False,
        training_weights=None,
    ):

        print("\n" + "=" * 70)
        print("UPLIFT / CATE MODEL")
        print("=" * 70)

        print(
            "\nMethod: T-Learner"
        )

        print(
            "Outcome model: Logistic Regression"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        self.fit(
            X_train=(
                split_results[
                    "X_train"
                ]
            ),

            treatment_train=(
                split_results[
                    "treatment_train"
                ]
            ),

            outcome_train=(
                split_results[
                    "outcome_train"
                ]
            ),

            preprocessor=preprocessor,

            sample_weight=(
                training_weights
                if use_training_weights
                else None
            ),
        )

        # ====================================================
        # EVALUATE OUTCOME MODELS
        # ====================================================

        metrics_frames = []

        split_mapping = {
            "train": (
                "X_train",
                "treatment_train",
                "outcome_train",
            ),

            "validation": (
                "X_val",
                "treatment_val",
                "outcome_val",
            ),

            "test": (
                "X_test",
                "treatment_test",
                "outcome_test",
            ),
        }

        for (
            split_name,
            keys,
        ) in split_mapping.items():

            (
                X_key,
                treatment_key,
                outcome_key,
            ) = keys

            metrics = (
                self.evaluate_outcome_models(
                    X=(
                        split_results[
                            X_key
                        ]
                    ),

                    treatment=(
                        split_results[
                            treatment_key
                        ]
                    ),

                    outcome=(
                        split_results[
                            outcome_key
                        ]
                    ),

                    split_name=(
                        split_name
                    ),
                )
            )

            metrics_frames.append(
                metrics
            )

        metrics_df = pd.concat(
            metrics_frames,
            ignore_index=True,
        )

        print("\n" + "=" * 70)
        print("UPLIFT OUTCOME MODEL PERFORMANCE")
        print("=" * 70)

        print(
            "\n"
            + metrics_df
            .round(4)
            .to_string(
                index=False
            )
        )

        metrics_path = os.path.join(
            self.output_dir,
            "outcome_model_metrics.csv",
        )

        metrics_df.to_csv(
            metrics_path,
            index=False,
        )

        # ====================================================
        # TEST SCORING
        # ====================================================

        X_test = (
            split_results[
                "X_test"
            ]
        )

        treatment_test = (
            split_results[
                "treatment_test"
            ]
        )

        outcome_test = (
            split_results[
                "outcome_test"
            ]
        )

        groups_test = (
            split_results.get(
                "groups_test"
            )
        )

        # ----------------------------------------------------
        # Propensity scores
        # ----------------------------------------------------

        test_propensity = None

        if propensity_results is not None:

            try:

                test_propensity = (
                    propensity_results[
                        "scores"
                    ][
                        "test"
                    ][
                        "propensity"
                    ]
                )

            except (
                KeyError,
                TypeError,
            ):

                print(
                    "\n[UPLIFT] Test propensity "
                    "scores unavailable."
                )

        score_df = (
            self.create_scoring_table(
                X=X_test,
                treatment=(
                    treatment_test
                ),
                outcome=(
                    outcome_test
                ),
                groups=(
                    groups_test
                ),
                propensity=(
                    test_propensity
                ),
                original_df=(
                    original_df
                ),
            )
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary_df = (
            self.summarize_uplift(
                score_df
            )
        )

        print("\n" + "=" * 70)
        print("TEST UPLIFT DISTRIBUTION")
        print("=" * 70)

        print(
            "\n"
            + summary_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # DECILES
        # ====================================================

        deciles_df = (
            self.uplift_deciles(
                score_df=score_df,
                n_bins=10,
            )
        )

        print("\n" + "=" * 70)
        print("UPLIFT DECILES")
        print("=" * 70)

        print(
            "\n"
            + deciles_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # SAVE
        # ====================================================

        score_path = os.path.join(
            self.output_dir,
            "uplift_test_scores.csv",
        )

        score_df.to_csv(
            score_path,
            index=False,
        )

        summary_path = os.path.join(
            self.output_dir,
            "uplift_summary.csv",
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        deciles_path = os.path.join(
            self.output_dir,
            "uplift_deciles.csv",
        )

        deciles_df.to_csv(
            deciles_path,
            index=False,
        )

        plot_path = (
            self.plot_uplift_distribution(
                score_df=score_df,
                split_name="test",
            )
        )

        model_paths = (
            self.save_models()
        )

        print(
            "\n[UPLIFT] Saved scoring table:"
            f"\n{score_path}"
        )

        print(
            "[UPLIFT] Saved summary:"
            f"\n{summary_path}"
        )

        print(
            "[UPLIFT] Saved deciles:"
            f"\n{deciles_path}"
        )

        return {
            "treated_model":
                self.treated_model,

            "control_model":
                self.control_model,

            "metrics":
                metrics_df,

            "scores":
                score_df,

            "summary":
                summary_df,

            "deciles":
                deciles_df,

            "metrics_path":
                metrics_path,

            "score_path":
                score_path,

            "summary_path":
                summary_path,

            "deciles_path":
                deciles_path,

            "plot_path":
                plot_path,

            "model_paths":
                model_paths,
        }