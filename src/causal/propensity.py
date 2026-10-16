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


class PropensityModel:
    """
    Estimate propensity scores:

        e(X) = P(T = 1 | X)

    where:
        T = treatment assignment
        X = pre-treatment covariates

    Important:
    The propensity model predicts treatment assignment,
    NOT the business outcome.

    The fitted model is later used for:
    - overlap diagnostics
    - positivity checks
    - IPW / stabilized IPW
    - matching
    - doubly robust estimators

    This class intentionally does NOT perform weighting or
    matching. Those belong in separate causal modules.
    """

    def __init__(
        self,
        output_dir="artifacts/causal/propensity",
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

        self.pipeline = None

    # ========================================================
    # BUILD MODEL
    # ========================================================

    def build_pipeline(
        self,
        preprocessor,
    ):

        model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,

            # Do NOT automatically use class_weight="balanced".
            #
            # For propensity estimation we want calibrated
            # treatment probabilities representing the actual
            # treatment assignment mechanism.
            class_weight=None,
        )

        self.pipeline = Pipeline([
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "model",
                model,
            ),
        ])

        return self.pipeline

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X_train,
        treatment_train,
        preprocessor,
    ):

        print("\n" + "=" * 70)
        print("PROPENSITY SCORE MODEL")
        print("=" * 70)

        print(
            f"\nTraining rows: "
            f"{len(X_train):,}"
        )

        print(
            f"Features: "
            f"{X_train.shape[1]:,}"
        )

        treatment_rate = (
            treatment_train.mean()
        )

        print(
            f"Treatment rate: "
            f"{treatment_rate:.4%}"
        )

        # ----------------------------------------------------
        # Validate treatment
        # ----------------------------------------------------

        treatment_values = set(
            pd.Series(
                treatment_train
            )
            .dropna()
            .unique()
            .tolist()
        )

        if not treatment_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "Propensity model requires "
                "binary treatment coded as 0/1."
            )

        # ----------------------------------------------------
        # Build pipeline
        # ----------------------------------------------------

        self.build_pipeline(
            preprocessor=preprocessor,
        )

        print(
            "\nFitting treatment-assignment model..."
        )

        self.pipeline.fit(
            X_train,
            treatment_train,
        )

        print(
            "\n[PROPENSITY] Model fitted."
        )

        return self.pipeline

    # ========================================================
    # PREDICT PROPENSITY
    # ========================================================

    def predict_propensity(
        self,
        X,
    ):

        if self.pipeline is None:

            raise ValueError(
                "Propensity model has not been fitted."
            )

        probabilities = (
            self.pipeline.predict_proba(
                X
            )
        )

        classes = (
            self.pipeline.named_steps[
                "model"
            ].classes_
        )

        positive_index = (
            np.where(
                classes == 1
            )[0]
        )

        if len(positive_index) != 1:

            raise ValueError(
                "Could not identify treatment=1 "
                "probability."
            )

        scores = probabilities[
            :,
            positive_index[0],
        ]

        return np.asarray(
            scores,
            dtype=float,
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(
        self,
        X,
        treatment,
        split_name,
    ):

        propensity = (
            self.predict_propensity(
                X
            )
        )

        roc_auc = roc_auc_score(
            treatment,
            propensity,
        )

        pr_auc = average_precision_score(
            treatment,
            propensity,
        )

        loss = log_loss(
            treatment,
            propensity,
        )

        metrics = {
            "split": split_name,
            "rows": len(X),
            "treatment_rate":
                float(
                    treatment.mean()
                ),
            "roc_auc":
                float(
                    roc_auc
                ),
            "pr_auc":
                float(
                    pr_auc
                ),
            "log_loss":
                float(
                    loss
                ),
            "propensity_mean":
                float(
                    np.mean(
                        propensity
                    )
                ),
            "propensity_min":
                float(
                    np.min(
                        propensity
                    )
                ),
            "propensity_max":
                float(
                    np.max(
                        propensity
                    )
                ),
        }

        return (
            metrics,
            propensity,
        )

    # ========================================================
    # PROPENSITY DISTRIBUTION
    # ========================================================

    def propensity_summary(
        self,
        propensity,
        treatment,
        split_name,
    ):

        result = pd.DataFrame({
            "propensity_score":
                propensity,

            "treatment":
                np.asarray(
                    treatment
                ),
        })

        quantiles = [
            0.00,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ]

        rows = []

        for treatment_value in [
            0,
            1,
        ]:

            subset = (
                result.loc[
                    result[
                        "treatment"
                    ] == treatment_value,
                    "propensity_score",
                ]
            )

            row = {
                "split": split_name,
                "treatment":
                    treatment_value,
                "n":
                    len(subset),
                "mean":
                    subset.mean(),
                "std":
                    subset.std(),
            }

            for q in quantiles:

                key = (
                    f"q{int(q * 100):02d}"
                )

                row[key] = (
                    subset.quantile(q)
                )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # EXTREME PROPENSITY DIAGNOSTICS
    # ========================================================

    def extreme_score_summary(
        self,
        propensity,
        treatment,
        split_name,
    ):

        propensity = np.asarray(
            propensity
        )

        treatment = np.asarray(
            treatment
        )

        thresholds = [
            0.01,
            0.025,
            0.05,
            0.10,
        ]

        rows = []

        for threshold in thresholds:

            low = (
                propensity
                < threshold
            )

            high = (
                propensity
                > (
                    1 - threshold
                )
            )

            extreme = (
                low
                | high
            )

            rows.append({
                "split":
                    split_name,

                "threshold":
                    threshold,

                "below_threshold":
                    int(
                        low.sum()
                    ),

                "above_threshold":
                    int(
                        high.sum()
                    ),

                "extreme_total":
                    int(
                        extreme.sum()
                    ),

                "extreme_pct":
                    float(
                        extreme.mean()
                    ),
            })

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # OVERLAP PLOT
    # ========================================================

    def plot_propensity_overlap(
        self,
        propensity,
        treatment,
        split_name,
    ):

        propensity = np.asarray(
            propensity
        )

        treatment = np.asarray(
            treatment
        )

        control_scores = (
            propensity[
                treatment == 0
            ]
        )

        treated_scores = (
            propensity[
                treatment == 1
            ]
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            control_scores,
            bins=50,
            density=True,
            alpha=0.5,
            label="Control",
        )

        plt.hist(
            treated_scores,
            bins=50,
            density=True,
            alpha=0.5,
            label="Treated",
        )

        plt.xlabel(
            "Estimated Propensity Score"
        )

        plt.ylabel(
            "Density"
        )

        plt.title(
            "Propensity Score Overlap "
            f"({split_name})"
        )

        plt.legend()

        plt.grid(
            alpha=0.2
        )

        path = os.path.join(
            self.output_dir,
            f"propensity_overlap_"
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
            f"[PROPENSITY] Saved overlap plot:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # SAVE SCORES
    # ========================================================
    def save_scores(
        self,
        X,
        propensity,
        treatment,
        split_name,
        treatment_col,
        outcome=None,
        outcome_col=None,
        groups=None,
        group_col=None,
    ):

        if isinstance(
            X,
            pd.DataFrame,
        ):

            score_df = (
                X.copy()
                .reset_index(drop=True)
            )

        else:

            score_df = pd.DataFrame(
                X
            )

        # --------------------------------------------------------
        # Treatment
        # --------------------------------------------------------

        score_df[
            treatment_col
        ] = np.asarray(
            treatment
        )

        # --------------------------------------------------------
        # Outcome
        # --------------------------------------------------------

        if (
            outcome is not None
            and outcome_col is not None
        ):

            score_df[
                outcome_col
            ] = np.asarray(
                outcome
            )

        # --------------------------------------------------------
        # Group
        # --------------------------------------------------------

        if (
            groups is not None
            and group_col is not None
        ):

            score_df[
                group_col
            ] = np.asarray(
                groups
            )

        # --------------------------------------------------------
        # Propensity
        # --------------------------------------------------------

        score_df[
            "propensity_score"
        ] = np.asarray(
            propensity,
            dtype=float,
        )

        path = os.path.join(
            self.output_dir,
            f"propensity_scores_{split_name}.csv",
        )

        score_df.to_csv(
            path,
            index=False,
        )

        return (
            score_df,
            path,
        )
    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save_model(
        self,
    ):

        if self.pipeline is None:

            raise ValueError(
                "No fitted propensity model "
                "available."
            )

        path = os.path.join(
            self.output_dir,
            "propensity_model.joblib",
        )

        joblib.dump(
            self.pipeline,
            path,
        )

        print(
            f"\n[PROPENSITY] "
            f"Saved model:"
            f"\n{path}"
        )

        return path

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        preprocessor,
        split_results,
        treatment_col="treatment",
        outcome_col="outcome",
        group_col=None,
    ):

        print("\n" + "=" * 70)
        print("PROPENSITY SCORE ESTIMATION")
        print("=" * 70)

        # ====================================================
        # FIT TRAIN ONLY
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
            preprocessor=preprocessor,
        )

        # ====================================================
        # EVALUATE ALL SPLITS
        # ====================================================

        metrics_rows = []

        score_results = {}

        split_mapping = {
            "train": (
                "X_train",
                "treatment_train",
                "outcome_train",
                "groups_train",
            ),

            "validation": (
                "X_val",
                "treatment_val",
                "outcome_val",
                "groups_val",
            ),

            "test": (
                "X_test",
                "treatment_test",
                "outcome_test",
                "groups_test",
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
                group_key,
            ) = keys

            X = (
                split_results[
                    X_key
                ]
            )

            treatment = (
                split_results[
                    treatment_key
                ]
            )

            outcome = (
                split_results[
                    outcome_key
                ]
            )

            groups = (
                split_results[
                    group_key
                ]
            )

            (
                metrics,
                propensity,
            ) = self.evaluate(
                X=X,
                treatment=treatment,
                split_name=split_name,
            )

            metrics_rows.append(
                metrics
            )

            # -----------------------------------------------
            # Distribution
            # -----------------------------------------------

            distribution = (
                self.propensity_summary(
                    propensity=propensity,
                    treatment=treatment,
                    split_name=split_name,
                )
            )

            # -----------------------------------------------
            # Extreme scores
            # -----------------------------------------------

            extreme = (
                self.extreme_score_summary(
                    propensity=propensity,
                    treatment=treatment,
                    split_name=split_name,
                )
            )

            # -----------------------------------------------
            # Save scores
            # -----------------------------------------------

            (
                score_df,
                score_path,
            ) = self.save_scores(
                X=X,

                propensity=propensity,

                treatment=treatment,

                split_name=split_name,

                treatment_col=(
                    treatment_col
                ),

                outcome=outcome,

                outcome_col=(
                    outcome_col
                ),

                groups=groups,

                group_col=(
                    group_col
                ),
            )

            # -----------------------------------------------
            # Plot
            # -----------------------------------------------

            plot_path = (
                self.plot_propensity_overlap(
                    propensity=propensity,
                    treatment=treatment,
                    split_name=split_name,
                )
            )

            score_results[
                split_name
            ] = {
                "scores":
                    score_df,

                "propensity":
                    propensity,

                "distribution":
                    distribution,

                "extreme":
                    extreme,

                "score_path":
                    score_path,

                "plot_path":
                    plot_path,
            }

        # ====================================================
        # METRICS TABLE
        # ====================================================

        metrics_df = pd.DataFrame(
            metrics_rows
        )

        print("\n" + "=" * 70)
        print("PROPENSITY MODEL PERFORMANCE")
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
            "propensity_metrics.csv",
        )

        metrics_df.to_csv(
            metrics_path,
            index=False,
        )

        # ====================================================
        # TEST DISTRIBUTION
        # ====================================================

        print("\n" + "=" * 70)
        print("TEST PROPENSITY DISTRIBUTION")
        print("=" * 70)

        test_distribution = (
            score_results[
                "test"
            ][
                "distribution"
            ]
        )

        print(
            "\n"
            + test_distribution
            .round(4)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # EXTREME SCORES
        # ====================================================

        print("\n" + "=" * 70)
        print("TEST EXTREME PROPENSITY SCORES")
        print("=" * 70)

        test_extreme = (
            score_results[
                "test"
            ][
                "extreme"
            ]
        )

        print(
            "\n"
            + test_extreme
            .round(4)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # SAVE MODEL
        # ====================================================

        model_path = (
            self.save_model()
        )

        return {
            "model":
                self.pipeline,

            "metrics":
                metrics_df,

            "scores":
                score_results,

            "model_path":
                model_path,

            "metrics_path":
                metrics_path,
        }