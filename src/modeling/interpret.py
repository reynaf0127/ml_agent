import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.inspection import permutation_importance
from statsmodels.nonparametric.smoothers_lowess import lowess

class ModelInterpreter:

    def __init__(
        self,
        output_dir="artifacts/model_interpretation",
    ):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # ========================================================
    # SAVE / SHOW PLOT
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
            f"[INTERPRETATION] Saved: {path}"
        )

        # Important for automated pipeline:
        # do NOT use plt.show()
        plt.close("all")

    # ========================================================
    # RIDGE / LINEAR COEFFICIENTS
    # ========================================================

    def linear_coefficients(
        self,
        pipeline,
        model_name,
    ):

        print("\n" + "=" * 70)
        print("MODEL COEFFICIENTS")
        print("=" * 70)

        preprocessor = (
            pipeline.named_steps[
                "preprocessor"
            ]
        )

        model = (
            pipeline.named_steps[
                "model"
            ]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        coefficients = (
            np.asarray(model.coef_)
            .ravel()
        )

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
        })

        coef_df[
            "abs_coefficient"
        ] = (
            coef_df["coefficient"]
            .abs()
        )

        coef_df = (
            coef_df
            .sort_values(
                "abs_coefficient",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        print(
            coef_df
            .head(30)
            .round(4)
            .to_string(index=False)
        )

        # Save table
        path = os.path.join(
            self.output_dir,
            f"{model_name}_coefficients.csv",
        )

        coef_df.to_csv(
            path,
            index=False,
        )

        print(
            f"\nSaved coefficients to:\n{path}"
        )

        # Plot top coefficients
        top = (
            coef_df
            .head(20)
            .sort_values(
                "coefficient"
            )
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.barh(
            top["feature"],
            top["coefficient"],
        )

        plt.xlabel(
            "Coefficient"
        )

        plt.title(
            f"{model_name}: "
            "Top Model Coefficients"
        )

        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            f"{model_name}_coefficients.png",
        )

        self._finish_plot(
            f"{model_name}_coefficients.png"
        )

        return coef_df

    # ========================================================
    # PERMUTATION IMPORTANCE
    # ========================================================

    def permutation_importance_plot(
        self,
        pipeline,
        model_name,
        X,
        y,
        problem_type,
    ):

        print("\n" + "=" * 70)
        print("PERMUTATION IMPORTANCE")
        print("=" * 70)

        if problem_type == "regression":
            scoring = (
                "neg_root_mean_squared_error"
            )
        else:
            scoring = "f1_weighted"

        result = permutation_importance(
            pipeline,
            X,
            y,
            scoring=scoring,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({
            "feature": X.columns,

            "importance_mean":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        })

        importance_df = (
            importance_df
            .sort_values(
                "importance_mean",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        print(
            importance_df
            .round(4)
            .to_string(index=False)
        )

        # Save
        path = os.path.join(
            self.output_dir,
            f"{model_name}_permutation_importance.csv",
        )

        importance_df.to_csv(
            path,
            index=False,
        )

        # Plot
        top = (
            importance_df
            .head(20)
            .sort_values(
                "importance_mean"
            )
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.barh(
            top["feature"],
            top["importance_mean"],
            xerr=top["importance_std"],
        )

        plt.xlabel(
            "Permutation Importance"
        )

        plt.title(
            f"{model_name}: "
            "Permutation Feature Importance"
        )

        plt.tight_layout()

        self._finish_plot(
            f"{model_name}_permutation_importance.png"
        )

        print(
            "[PERMUTATION] Complete."
        )

        return importance_df

    # ========================================================
    # SHAP ANALYSIS
    # ========================================================

    def shap_analysis(
        self,
        pipeline,
        model_name,
        X,
        max_samples=2000,
        top_n=15,
    ):

        print("\n" + "=" * 70)
        print("SHAP MODEL INTERPRETATION")
        print("=" * 70)

        # ----------------------------------------------------
        # Extract fitted pipeline components
        # ----------------------------------------------------

        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        model = pipeline.named_steps[
            "model"
        ]

        # ----------------------------------------------------
        # Sample data for SHAP
        # ----------------------------------------------------

        if len(X) > max_samples:

            X_sample = X.sample(
                n=max_samples,
                random_state=42,
            )

        else:

            X_sample = X.copy()

        print(
            f"\nSHAP sample size: "
            f"{len(X_sample):,}"
        )

        # ----------------------------------------------------
        # Transform raw features
        # ----------------------------------------------------

        X_processed = (
            preprocessor.transform(
                X_sample
            )
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        # Make sure we have a DataFrame
        X_processed = pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X_sample.index,
        )

        # ----------------------------------------------------
        # Create SHAP explainer
        #
        # shap.Explainer automatically chooses an appropriate
        # explainer for many sklearn models.
        # ----------------------------------------------------

        explainer = shap.Explainer(
            model,
            X_processed,
        )

        shap_values = explainer(
            X_processed
        )

        # ====================================================
        # GLOBAL FEATURE IMPORTANCE
        # ====================================================

        importance = np.abs(
            shap_values.values
        ).mean(axis=0)

        shap_importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": importance,
        })

        shap_importance = (
            shap_importance
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        print("\nTop SHAP features:\n")

        print(
            shap_importance
            .head(top_n)
            .round(4)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Save importance
        # ----------------------------------------------------

        path = os.path.join(
            self.output_dir,
            f"{model_name}_shap_importance.csv",
        )

        shap_importance.to_csv(
            path,
            index=False,
        )

        # ============================================================
        # SHAP BAR PLOT
        # ============================================================

        print(
            "\n[SHAP] Creating importance bar plot..."
        )

        shap.plots.bar(
            shap_values,
            max_display=top_n,
            show=False,
        )

        plt.title(
            f"{model_name}: "
            "SHAP Feature Importance"
        )

        print(
            "[SHAP] Importance plot created. "
            "Saving..."
        )

        self._finish_plot(
            f"{model_name}_shap_importance.png"
        )

        print(
            "[SHAP] Importance plot complete."
        )


        # ============================================================
        # SHAP BEESWARM
        # ============================================================

        print(
            "\n[SHAP] Creating beeswarm plot..."
        )

        shap.plots.beeswarm(
            shap_values,
            max_display=top_n,
            show=False,
        )

        print(
            "[SHAP] Beeswarm created. Saving..."
        )

        self._finish_plot(
            f"{model_name}_shap_beeswarm.png"
        )

        print(
            "[SHAP] Beeswarm complete."
        )

        return (
            shap_values,
            shap_importance,
            X_processed,
        )

    # ========================================================
    # SHAP FEATURE RELATIONSHIPS
    # ========================================================

    def shap_dependence_plots(
        self,
        shap_values,
        shap_importance,
        model_name,
        top_n=None,
        ncols=3,
    ):

        print("\n" + "=" * 70)
        print("SHAP FEATURE RELATIONSHIPS")
        print("=" * 70)

        # ========================================================
        # FEATURES TO PLOT
        # ========================================================

        features = (
            shap_importance["feature"]
            .tolist()
        )

        # Optional: only plot top N
        if top_n is not None:
            features = features[:top_n]

        n_features = len(features)

        nrows = int(
            np.ceil(
                n_features / ncols
            )
        )

        print(
            f"\nCreating SHAP dependence plots for "
            f"{n_features} features..."
        )

        print(
            f"Layout: {nrows} rows × {ncols} columns"
        )

        # ========================================================
        # CREATE FIGURE
        # ========================================================

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                6 * ncols,
                4.5 * nrows,
            ),
        )

        axes = np.array(
            axes
        ).reshape(-1)

        # ========================================================
        # PLOT EACH FEATURE
        # ========================================================

        for i, feature in enumerate(
            features
        ):

            ax = axes[i]

            print(
                f"  [{i + 1}/{n_features}] "
                f"{feature}"
            )

            # Get feature position
            feature_index = list(
                shap_values.feature_names
            ).index(
                feature
            )

            # Actual transformed feature values
            x = shap_values.data[
                :,
                feature_index
            ]

            # SHAP contribution
            y = shap_values.values[
                :,
                feature_index
            ]

            # -----------------------------------------------
            # Scatter
            # -----------------------------------------------

            ax.scatter(
                x,
                y,
                alpha=0.35,
                s=18,
            )

            # ------------------------------------------------
            # LOWESS smoothed relationship
            # ------------------------------------------------

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            x_clean = np.asarray(x)[valid]
            y_clean = np.asarray(y)[valid]

            if (
                len(x_clean) >= 20
                and len(np.unique(x_clean)) >= 5
            ):

                smooth = lowess(
                    endog=y_clean,
                    exog=x_clean,
                    frac=0.25,
                    return_sorted=True,
                )

                ax.plot(
                    smooth[:, 0],
                    smooth[:, 1],
                    linewidth=2,
                )

            # SHAP = 0 reference
            ax.axhline(
                y=0,
                linestyle="--",
                linewidth=1,
            )

            # -----------------------------------------------
            # SHAP = 0 reference
            # -----------------------------------------------

            ax.axhline(
                y=0,
                linestyle="--",
                linewidth=1,
            )

            # -----------------------------------------------
            # Labels
            # -----------------------------------------------

            ax.set_title(
                feature,
                fontsize=10,
            )

            ax.set_xlabel(
                "Feature value"
            )

            ax.set_ylabel(
                "SHAP value"
            )

            ax.grid(
                alpha=0.2
            )

        # ========================================================
        # REMOVE UNUSED PANELS
        # ========================================================

        for j in range(
            n_features,
            len(axes),
        ):

            fig.delaxes(
                axes[j]
            )

        # ========================================================
        # TITLE
        # ========================================================

        fig.suptitle(
            f"{model_name}: "
            "SHAP Feature Relationships",
            fontsize=16,
            y=1.01,
        )

        plt.tight_layout()

        # ========================================================
        # SAVE
        # ========================================================

        path = os.path.join(
            self.output_dir,
            f"{model_name}_shap_all_relationships.png",
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        print(
            f"\n[INTERPRETATION] Saved: {path}"
        )

    # ========================================================
    # INDIVIDUAL SHAP EXPLANATION
    # ========================================================

    def shap_individual_explanation(
        self,
        shap_values,
        row=0,
    ):

        print("\n" + "=" * 70)
        print(f"INDIVIDUAL SHAP EXPLANATION: ROW {row}")
        print("=" * 70)

        shap.plots.waterfall(
            shap_values[row],
            max_display=15,
            show=False,
        )

        self._finish_plot(
            f"shap_individual_row_{row}.png"
        )