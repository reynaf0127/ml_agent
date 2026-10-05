import os
from datetime import datetime
import re
import numpy as np
import pandas as pd


class ModelPredictor:

    def __init__(
        self,
        output_dir="artifacts/predictions",
    ):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    def predict(
        self,
        pipeline,
        X,
        original_df,
        problem_type,
        model_name,
        target_col,
    ):

        print("\n" + "=" * 70)
        print("GENERATING FINAL PREDICTIONS")
        print("=" * 70)

        print(
            f"\nModel: {model_name}"
        )

        print(
            f"Problem type: {problem_type}"
        )

        print(
            f"Rows to score: {len(X):,}"
        )

        # ----------------------------------------------------
        # Make predictions
        # ----------------------------------------------------

        predictions = pipeline.predict(
            X
        )

        # ----------------------------------------------------
        # Start with original data
        # ----------------------------------------------------

        output_df = (
            original_df
            .loc[X.index]
            .copy()
        )

        output_df[
            "model_prediction"
        ] = predictions

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if problem_type == "classification":

            if hasattr(
                pipeline,
                "predict_proba",
            ):

                probabilities = (
                    pipeline.predict_proba(
                        X
                    )
                )

                model = (
                    pipeline.named_steps[
                        "model"
                    ]
                )

                classes = model.classes_

                # --------------------------------------------
                # Add probability for every class
                # --------------------------------------------

                for i, class_name in enumerate(
                    classes
                ):

                    # Convert integer-like floats:
                    # 0.0 -> 0
                    # 1.0 -> 1
                    if isinstance(
                        class_name,
                        (float, np.floating),
                    ) and float(class_name).is_integer():

                        class_name = int(
                            class_name
                        )

                    safe_class = str(
                        class_name
                    )

                    # BigQuery-safe column name
                    safe_class = re.sub(
                        r"[^A-Za-z0-9_]",
                        "_",
                        safe_class,
                    )

                    output_df[
                        f"probability_{safe_class}"
                    ] = probabilities[:, i]

                # --------------------------------------------
                # Binary classification convenience column
                # --------------------------------------------

                if len(classes) == 2:

                    output_df[
                        "prediction_probability"
                    ] = probabilities[:, 1]

        # ====================================================
        # METADATA
        # ====================================================

        output_df[
            "prediction_model"
        ] = model_name

        output_df[
            "prediction_target"
        ] = target_col

        output_df[
            "prediction_timestamp"
        ] = pd.Timestamp.utcnow()

        # ----------------------------------------------------
        # Print preview
        # ----------------------------------------------------

        print(
            "\nPrediction preview:"
        )

        display_cols = [
            target_col,
            "model_prediction",
        ]

        if (
            "prediction_probability"
            in output_df.columns
        ):
            display_cols.append(
                "prediction_probability"
            )

        print(
            output_df[
                display_cols
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print(
            f"\nOutput shape: "
            f"{output_df.shape}"
        )

        return output_df

    # ========================================================
    # SAVE LOCALLY
    # ========================================================

    def save(
        self,
        prediction_df,
        model_name,
        target_col,
    ):

        filename = (
            f"{model_name}_"
            f"{target_col}_predictions.parquet"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        prediction_df.to_parquet(
            path,
            index=False,
        )

        print(
            f"\n[PREDICTION] Saved locally:"
        )

        print(
            path
        )

        return path