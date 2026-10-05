import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from scipy import sparse
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
    # SAVE / CLOSE PLOT
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

        # Never block the terminal.
        plt.close("all")

        return path

    # ========================================================
    # SAFE FEATURE NAMES
    # ========================================================

    def _get_feature_names(
        self,
        preprocessor,
        X_processed,
    ):
        """
        Retrieve transformed feature names.

        Custom transformers such as feature hashing may not
        fully support sklearn's get_feature_names_out().
        This method therefore uses several fallbacks.
        """

        # ----------------------------------------------------
        # Best case: pandas output already has names
        # ----------------------------------------------------

        if isinstance(
            X_processed,
            pd.DataFrame,
        ):

            return (
                X_processed.columns
                .astype(str)
                .tolist()
            )

        # ----------------------------------------------------
        # Try sklearn feature-name API
        # ----------------------------------------------------

        try:

            names = (
                preprocessor
                .get_feature_names_out()
            )

            names = (
                np.asarray(names)
                .astype(str)
                .tolist()
            )

            if (
                len(names)
                == X_processed.shape[1]
            ):

                return names

        except Exception as e:

            print(
                "\n[INTERPRETATION] "
                "Could not retrieve feature names "
                "from sklearn."
            )

            print(
                f"[INTERPRETATION] Reason: {e}"
            )

        # ----------------------------------------------------
        # Final fallback
        # ----------------------------------------------------

        print(
            "[INTERPRETATION] "
            "Using generated transformed "
            "feature names."
        )

        return [
            f"feature_{i}"
            for i in range(
                X_processed.shape[1]
            )
        ]

    # ========================================================
    # SAFE TRANSFORMED DATA
    # ========================================================

    def _get_processed_data(
        self,
        preprocessor,
        X,
        dense=True,
    ):
        """
        Transform raw X and safely recover feature names.

        SHAP generally works more reliably with a dense
        DataFrame for the relatively small interpretation
        sample.

        Do not use dense=True on the entire training dataset
        when hashing / one-hot encoding creates many columns.
        """

        X_processed = (
            preprocessor.transform(X)
        )

        feature_names = (
            self._get_feature_names(
                preprocessor=preprocessor,
                X_processed=X_processed,
            )
        )

        # ----------------------------------------------------
        # Already pandas
        # ----------------------------------------------------

        if isinstance(
            X_processed,
            pd.DataFrame,
        ):

            result = X_processed.copy()

            result.columns = (
                result.columns
                .astype(str)
            )

            return (
                result,
                result.columns.tolist(),
            )

        # ----------------------------------------------------
        # Sparse matrix
        # ----------------------------------------------------

        if sparse.issparse(
            X_processed
        ):

            if dense:

                X_processed = (
                    X_processed.toarray()
                )

            else:

                return (
                    X_processed,
                    feature_names,
                )

        # ----------------------------------------------------
        # Convert dense result to DataFrame
        # ----------------------------------------------------

        X_processed = np.asarray(
            X_processed
        )

        if (
            X_processed.shape[1]
            != len(feature_names)
        ):

            print(
                "\n[INTERPRETATION] "
                "Feature-name count does not match "
                "transformed columns."
            )

            print(
                "Transformed columns:",
                X_processed.shape[1],
            )

            print(
                "Feature names:",
                len(feature_names),
            )

            feature_names = [
                f"feature_{i}"
                for i in range(
                    X_processed.shape[1]
                )
            ]

        X_processed = pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X.index,
        )

        return (
            X_processed,
            feature_names,
        )

    # ========================================================
    # SAFE COEFFICIENT ARRAY
    # ========================================================

    def _get_coefficients(
        self,
        model,
    ):
        """
        Return one coefficient per transformed feature.

        Binary LogisticRegression:
            coef_ shape = (1, n_features)

        Regression:
            coef_ shape = (n_features,)

        Multiclass classification has multiple coefficient
        vectors. For global importance we use mean absolute
        coefficient across classes.
        """

        coef = np.asarray(
            model.coef_
        )

        if coef.ndim == 1:

            return coef

        if coef.shape[0] == 1:

            return coef[0]

        # Multiclass fallback
        return np.mean(
            np.abs(coef),
            axis=0,
        )

    # ========================================================
    # LINEAR / RIDGE / LOGISTIC COEFFICIENTS
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

        # ----------------------------------------------------
        # Number of coefficients
        # ----------------------------------------------------

        coefficients = (
            self._get_coefficients(
                model
            )
        )

        # ----------------------------------------------------
        # Try feature names
        # ----------------------------------------------------

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            feature_names = (
                np.asarray(
                    feature_names
                )
                .astype(str)
                .tolist()
            )

        except Exception as e:

            print(
                "\n[COEFFICIENTS] "
                "Could not retrieve transformed "
                "feature names."
            )

            print(
                f"[COEFFICIENTS] Reason: {e}"
            )

            feature_names = [
                f"feature_{i}"
                for i in range(
                    len(coefficients)
                )
            ]

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if (
            len(feature_names)
            != len(coefficients)
        ):

            print(
                "\n[COEFFICIENTS] "
                "Feature-name mismatch."
            )

            print(
                "Names:",
                len(feature_names),
            )

            print(
                "Coefficients:",
                len(coefficients),
            )

            print(
                "[COEFFICIENTS] "
                "Using generated names."
            )

            feature_names = [
                f"feature_{i}"
                for i in range(
                    len(coefficients)
                )
            ]

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        coef_df = pd.DataFrame({
            "feature":
                feature_names,

            "coefficient":
                coefficients,
        })

        coef_df[
            "abs_coefficient"
        ] = (
            coef_df[
                "coefficient"
            ]
            .abs()
        )

        coef_df = (
            coef_df
            .sort_values(
                "abs_coefficient",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print(
            coef_df
            .head(30)
            .round(4)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        path = os.path.join(
            self.output_dir,
            f"{model_name}_coefficients.csv",
        )

        coef_df.to_csv(
            path,
            index=False,
        )

        print(
            "\n[COEFFICIENTS] Saved:"
        )

        print(path)

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

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

        plt.axvline(
            x=0,
            linestyle="--",
            linewidth=1,
        )

        plt.xlabel(
            "Coefficient"
        )

        plt.title(
            f"{model_name}: "
            "Top Model Coefficients"
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

        # ----------------------------------------------------
        # Scoring
        # ----------------------------------------------------

        if problem_type == "regression":

            scoring = (
                "neg_root_mean_squared_error"
            )

            metric_label = (
                "Increase in RMSE"
            )

        else:

            # Better than weighted F1 for imbalanced
            # binary classification.
            scoring = (
                "average_precision"
            )

            metric_label = (
                "Decrease in PR-AUC"
            )

        print(
            f"\nScoring metric: {scoring}"
        )

        # ----------------------------------------------------
        # Calculate
        # ----------------------------------------------------

        result = (
            permutation_importance(
                pipeline,
                X,
                y,
                scoring=scoring,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
        )

        importance_df = pd.DataFrame({
            "feature":
                X.columns,

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
            .reset_index(
                drop=True
            )
        )

        print(
            importance_df
            .round(6)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        path = os.path.join(
            self.output_dir,
            f"{model_name}_permutation_importance.csv",
        )

        importance_df.to_csv(
            path,
            index=False,
        )

        print(
            f"\n[PERMUTATION] Saved: {path}"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

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
            xerr=top[
                "importance_std"
            ],
        )

        plt.axvline(
            x=0,
            linestyle="--",
            linewidth=1,
        )

        plt.xlabel(
            metric_label
        )

        plt.title(
            f"{model_name}: "
            "Permutation Feature Importance"
        )

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
        # Pipeline
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        if len(X) > max_samples:

            X_sample = X.sample(
                n=max_samples,
                random_state=42,
            )

        else:

            X_sample = X.copy()

        print(
            "\nSHAP sample size:",
            f"{len(X_sample):,}",
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        (
            X_processed,
            feature_names,
        ) = self._get_processed_data(
            preprocessor=preprocessor,
            X=X_sample,
            dense=True,
        )

        print(
            "\n[SHAP] Processed shape:",
            X_processed.shape,
        )

        print(
            "[SHAP] Number of features:",
            len(feature_names),
        )

        print(
            "\n[SHAP] First transformed features:"
        )

        for feature in (
            feature_names[:30]
        ):

            print(
                f"  - {feature}"
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if (
            X_processed.shape[1]
            != len(feature_names)
        ):

            raise ValueError(
                "SHAP feature-name mismatch: "
                f"{X_processed.shape[1]} columns "
                f"but {len(feature_names)} names."
            )

        # ----------------------------------------------------
        # SHAP explainer
        # ----------------------------------------------------

        print(
            "\n[SHAP] Building explainer..."
        )

        explainer = shap.Explainer(
            model,
            X_processed,
        )

        print(
            "[SHAP] Calculating SHAP values..."
        )

        shap_values = explainer(
            X_processed
        )

        # ----------------------------------------------------
        # Classification can occasionally produce
        # 3D SHAP values.
        # ----------------------------------------------------

        values = np.asarray(
            shap_values.values
        )

        if values.ndim == 3:

            print(
                "\n[SHAP] Multi-output SHAP "
                "values detected."
            )

            # For binary classification use positive class.
            if values.shape[2] == 2:

                shap_values = (
                    shap_values[:, :, 1]
                )

                values = np.asarray(
                    shap_values.values
                )

            else:

                # Generic multiclass:
                # global importance will be averaged
                # across outputs later.
                pass

        # ====================================================
        # GLOBAL IMPORTANCE
        # ====================================================

        values = np.asarray(
            shap_values.values
        )

        if values.ndim == 2:

            importance = (
                np.abs(values)
                .mean(axis=0)
            )

        elif values.ndim == 3:

            importance = (
                np.abs(values)
                .mean(axis=(0, 2))
            )

        else:

            raise ValueError(
                "Unexpected SHAP value shape: "
                f"{values.shape}"
            )

        shap_importance = (
            pd.DataFrame({
                "feature":
                    feature_names,

                "mean_abs_shap":
                    importance,
            })
        )

        shap_importance = (
            shap_importance
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print(
            "\nTop SHAP features:\n"
        )

        print(
            shap_importance
            .head(top_n)
            .round(6)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Save table
        # ----------------------------------------------------

        path = os.path.join(
            self.output_dir,
            f"{model_name}_shap_importance.csv",
        )

        shap_importance.to_csv(
            path,
            index=False,
        )

        print(
            f"\n[SHAP] Saved importance: {path}"
        )

        # ====================================================
        # SHAP BAR
        # ====================================================

        print(
            "\n[SHAP] Creating importance "
            "bar plot..."
        )

        try:

            shap.plots.bar(
                shap_values,
                max_display=top_n,
                show=False,
            )

            plt.title(
                f"{model_name}: "
                "SHAP Feature Importance"
            )

            self._finish_plot(
                f"{model_name}_shap_importance.png"
            )

        except Exception as e:

            print(
                "[SHAP] Standard bar plot "
                f"failed: {e}"
            )

            # Manual fallback
            top = (
                shap_importance
                .head(top_n)
                .sort_values(
                    "mean_abs_shap"
                )
            )

            plt.figure(
                figsize=(9, 8)
            )

            plt.barh(
                top["feature"],
                top["mean_abs_shap"],
            )

            plt.xlabel(
                "Mean |SHAP value|"
            )

            plt.title(
                f"{model_name}: "
                "SHAP Feature Importance"
            )

            self._finish_plot(
                f"{model_name}_shap_importance.png"
            )

        print(
            "[SHAP] Importance plot complete."
        )

        # ====================================================
        # BEESWARM
        # ====================================================

        print(
            "\n[SHAP] Creating beeswarm plot..."
        )

        try:

            shap.plots.beeswarm(
                shap_values,
                max_display=top_n,
                show=False,
            )

            self._finish_plot(
                f"{model_name}_shap_beeswarm.png"
            )

            print(
                "[SHAP] Beeswarm complete."
            )

        except Exception as e:

            print(
                "[SHAP] Beeswarm skipped."
            )

            print(
                f"[SHAP] Reason: {e}"
            )

        # ====================================================
        # RETURN
        # ====================================================

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

        # ----------------------------------------------------
        # Feature names available from SHAP
        # ----------------------------------------------------

        shap_feature_names = (
            shap_values.feature_names
        )

        if shap_feature_names is None:

            shap_feature_names = [
                f"feature_{i}"
                for i in range(
                    shap_values.values.shape[1]
                )
            ]

        shap_feature_names = (
            list(
                shap_feature_names
            )
        )

        # ----------------------------------------------------
        # Features ranked by SHAP importance
        # ----------------------------------------------------

        features = (
            shap_importance[
                "feature"
            ]
            .tolist()
        )

        features = [
            feature
            for feature in features
            if feature
            in shap_feature_names
        ]

        if top_n is not None:

            features = (
                features[:top_n]
            )

        n_features = len(
            features
        )

        if n_features == 0:

            print(
                "\n[SHAP] No valid features "
                "available for relationship plots."
            )

            return

        ncols = min(
            ncols,
            n_features,
        )

        nrows = int(
            np.ceil(
                n_features
                / ncols
            )
        )

        print(
            "\nCreating SHAP dependence plots "
            f"for {n_features} features..."
        )

        print(
            f"Layout: "
            f"{nrows} rows × "
            f"{ncols} columns"
        )

        # ----------------------------------------------------
        # Figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                6 * ncols,
                4.5 * nrows,
            ),
            squeeze=False,
        )

        axes = (
            np.asarray(axes)
            .reshape(-1)
        )

        shap_data = np.asarray(
            shap_values.data
        )

        shap_matrix = np.asarray(
            shap_values.values
        )

        # Binary / normal case expected here.
        if shap_matrix.ndim != 2:

            print(
                "\n[SHAP] Relationship plots "
                "currently require 2D SHAP values."
            )

            plt.close(fig)

            return

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        for i, feature in enumerate(
            features
        ):

            ax = axes[i]

            print(
                f"  [{i + 1}/{n_features}] "
                f"{feature}"
            )

            feature_index = (
                shap_feature_names
                .index(feature)
            )

            x = np.asarray(
                shap_data[
                    :,
                    feature_index
                ]
            )

            y = np.asarray(
                shap_matrix[
                    :,
                    feature_index
                ]
            )

            # ------------------------------------------------
            # Convert values to numeric if possible
            # ------------------------------------------------

            try:

                x = x.astype(float)

            except (
                ValueError,
                TypeError,
            ):

                print(
                    f"    [SHAP] Skipping "
                    f"{feature}: "
                    "non-numeric transformed values."
                )

                ax.set_visible(
                    False
                )

                continue

            # ------------------------------------------------
            # Valid values
            # ------------------------------------------------

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            x_clean = x[valid]
            y_clean = y[valid]

            if len(x_clean) == 0:

                ax.set_visible(
                    False
                )

                continue

            # ------------------------------------------------
            # Scatter
            # ------------------------------------------------

            ax.scatter(
                x_clean,
                y_clean,
                alpha=0.30,
                s=16,
            )

            # ------------------------------------------------
            # LOWESS
            # ------------------------------------------------

            unique_values = (
                np.unique(
                    x_clean
                )
            )

            if (
                len(x_clean) >= 20
                and len(
                    unique_values
                ) >= 5
            ):

                try:

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

                except Exception as e:

                    print(
                        f"    [LOWESS] "
                        f"Skipped: {e}"
                    )

            # ------------------------------------------------
            # Zero reference
            # ------------------------------------------------

            ax.axhline(
                y=0,
                linestyle="--",
                linewidth=1,
            )

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            ax.set_title(
                feature,
                fontsize=10,
            )

            ax.set_xlabel(
                "Transformed feature value"
            )

            ax.set_ylabel(
                "SHAP value"
            )

            ax.grid(
                alpha=0.2
            )

        # ----------------------------------------------------
        # Unused panels
        # ----------------------------------------------------

        for j in range(
            n_features,
            len(axes),
        ):

            fig.delaxes(
                axes[j]
            )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        fig.suptitle(
            f"{model_name}: "
            "SHAP Feature Relationships",
            fontsize=16,
            y=1.01,
        )

        fig.tight_layout()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        path = os.path.join(
            self.output_dir,
            f"{model_name}_shap_all_relationships.png",
        )

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        print(
            f"\n[INTERPRETATION] Saved: "
            f"{path}"
        )

        return path

    # ========================================================
    # INDIVIDUAL SHAP EXPLANATION
    # ========================================================

    def shap_individual_explanation(
        self,
        shap_values,
        row=0,
    ):

        print("\n" + "=" * 70)
        print(
            "INDIVIDUAL SHAP EXPLANATION: "
            f"ROW {row}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Validate row
        # ----------------------------------------------------

        if row < 0:

            row = 0

        if row >= len(
            shap_values
        ):

            print(
                "\n[SHAP] Requested row "
                "is outside the SHAP sample."
            )

            print(
                "[SHAP] Using row 0."
            )

            row = 0

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        try:

            shap.plots.waterfall(
                shap_values[row],
                max_display=15,
                show=False,
            )

            self._finish_plot(
                f"shap_individual_row_{row}.png"
            )

        except Exception as e:

            print(
                "\n[SHAP] Individual "
                "waterfall plot failed."
            )

            print(
                f"[SHAP] Reason: {e}"
            )