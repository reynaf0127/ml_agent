import json
import os

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import (
    build_preprocessor,
)


class CausalPreprocessor:
    """
    Configure preprocessing specifically for causal covariates.

    This module is independent from the prediction workflow.

    Responsibilities:
    - infer initial feature types
    - allow user review/update
    - choose one-hot vs feature hashing
    - choose numerical transformations
    - save causal preprocessing configuration
    - reuse existing causal configuration
    - build sklearn preprocessing pipeline
    """

    def __init__(
        self,
        df,
        feature_cols,
        config_path=(
            "data/metadata/"
            "causal_preprocessing.json"
        ),
        onehot_threshold=100,
        hash_n_features=256,
    ):

        self.df = df

        self.feature_cols = [
            col
            for col in feature_cols
            if col in df.columns
        ]

        self.config_path = (
            config_path
        )

        self.onehot_threshold = (
            onehot_threshold
        )

        self.hash_n_features = (
            hash_n_features
        )

        self.config = None

    # ========================================================
    # INPUT PARSER
    # ========================================================

    def _parse_columns(
        self,
        value,
    ):

        value = (
            value
            .strip()
        )

        if not value:
            return []

        columns = [
            col.strip()
            for col in value.split(",")
            if col.strip()
        ]

        unknown = [
            col
            for col in columns
            if col not in self.feature_cols
        ]

        if unknown:

            raise ValueError(
                "Unknown causal features: "
                f"{unknown}"
            )

        return list(
            dict.fromkeys(
                columns
            )
        )

    # ========================================================
    # INITIAL TYPE INFERENCE
    # ========================================================

    def _infer_initial_types(
        self,
    ):

        categorical = []

        numerical = []

        boolean = []

        for col in self.feature_cols:

            series = (
                self.df[col]
            )

            unique_count = (
                series
                .dropna()
                .nunique()
            )

            # -----------------------------------------------
            # Boolean
            # -----------------------------------------------

            if (
                pd.api.types.is_bool_dtype(
                    series
                )
                or unique_count <= 2
            ):

                boolean.append(
                    col
                )

            # -----------------------------------------------
            # Native categorical/string
            # -----------------------------------------------

            elif (
                pd.api.types.is_object_dtype(
                    series
                )
                or pd.api.types.is_string_dtype(
                    series
                )
                or isinstance(
                    series.dtype,
                    pd.CategoricalDtype,
                )
            ):

                categorical.append(
                    col
                )

            # -----------------------------------------------
            # Numeric
            # -----------------------------------------------

            else:

                numerical.append(
                    col
                )

        return (
            categorical,
            numerical,
            boolean,
        )

    # ========================================================
    # PRINT CARDINALITY
    # ========================================================

    def _print_feature_summary(
        self,
    ):

        print("\n" + "=" * 70)
        print("CAUSAL FEATURE TYPE SUMMARY")
        print("=" * 70)

        rows = []

        for col in self.feature_cols:

            series = (
                self.df[col]
            )

            rows.append({
                "feature":
                    col,

                "dtype":
                    str(
                        series.dtype
                    ),

                "unique":
                    series
                    .nunique(
                        dropna=True
                    ),

                "missing_pct":
                    series
                    .isna()
                    .mean(),

                "example":
                    (
                        series
                        .dropna()
                        .iloc[0]
                        if (
                            series
                            .notna()
                            .any()
                        )
                        else None
                    ),
            })

        summary = pd.DataFrame(
            rows
        )

        print(
            "\n"
            + summary
            .to_string(
                index=False
            )
        )

        return summary

    # ========================================================
    # DEFAULT ENCODING
    # ========================================================

    def _default_encoding(
        self,
        categorical,
    ):

        onehot = []

        hashing = []

        for col in categorical:

            n_unique = (
                self.df[col]
                .nunique(
                    dropna=True
                )
            )

            if (
                n_unique
                <= self.onehot_threshold
            ):

                onehot.append(
                    col
                )

            else:

                hashing.append(
                    col
                )

        return (
            onehot,
            hashing,
        )

    # ========================================================
    # CREATE NEW CONFIG
    # ========================================================

    def create_config(
        self,
    ):

        print("\n" + "=" * 70)
        print("CAUSAL PREPROCESSING CONFIGURATION")
        print("=" * 70)

        self._print_feature_summary()

        (
            categorical,
            numerical,
            boolean,
        ) = self._infer_initial_types()

        print(
            "\nAutomatically detected:"
        )

        print(
            f"\nCategorical:\n"
            f"{categorical}"
        )

        print(
            f"\nNumerical:\n"
            f"{numerical}"
        )

        print(
            f"\nBoolean:\n"
            f"{boolean}"
        )

        # ====================================================
        # USER TYPE CORRECTION
        # ====================================================

        print(
            "\nSome integer/numeric columns may actually "
            "represent categorical IDs."
        )

        print(
            "Examples include campaign IDs, category IDs, "
            "region codes, product IDs, etc."
        )

        convert_to_cat = (
            self._parse_columns(
                input(
                    "\nEnter numerical columns that "
                    "should be treated as categorical "
                    "(comma separated, Enter for none):\n> "
                )
            )
        )

        # Remove from other groups first

        numerical = [
            col
            for col in numerical
            if col not in convert_to_cat
        ]

        boolean = [
            col
            for col in boolean
            if col not in convert_to_cat
        ]

        categorical = list(
            dict.fromkeys(
                categorical
                + convert_to_cat
            )
        )

        # ====================================================
        # BOOLEAN CORRECTION
        # ====================================================

        convert_to_bool = (
            self._parse_columns(
                input(
                    "\nEnter columns that should "
                    "be treated as boolean "
                    "(comma separated, Enter for none):\n> "
                )
            )
        )

        categorical = [
            col
            for col in categorical
            if col not in convert_to_bool
        ]

        numerical = [
            col
            for col in numerical
            if col not in convert_to_bool
        ]

        boolean = list(
            dict.fromkeys(
                boolean
                + convert_to_bool
            )
        )

        # ====================================================
        # ENCODING PLAN
        # ====================================================

        (
            onehot,
            hashing,
        ) = self._default_encoding(
            categorical
        )

        print("\n" + "=" * 70)
        print("PROPOSED CAUSAL ENCODING")
        print("=" * 70)

        print(
            f"\nOne-hot encoding:\n"
            f"{onehot}"
        )

        print(
            f"\nFeature hashing:\n"
            f"{hashing}"
        )

        update_encoding = input(
            "\nUpdate categorical encoding? "
            "(y/N):\n> "
        ).strip().lower()

        if update_encoding in [
            "y",
            "yes",
        ]:

            onehot = (
                self._parse_columns(
                    input(
                        "\nOne-hot columns:\n> "
                    )
                )
            )

            hashing = (
                self._parse_columns(
                    input(
                        "\nFeature hashing columns:\n> "
                    )
                )
            )

            # -----------------------------------------------
            # Must only contain categorical variables
            # -----------------------------------------------

            invalid = (
                set(
                    onehot
                    + hashing
                )
                - set(
                    categorical
                )
            )

            if invalid:

                raise ValueError(
                    "Encoding columns must be "
                    "categorical features. "
                    f"Invalid: {sorted(invalid)}"
                )

            overlap = (
                set(onehot)
                & set(hashing)
            )

            if overlap:

                raise ValueError(
                    "Columns cannot be both one-hot "
                    "and hashed: "
                    f"{sorted(overlap)}"
                )

        # ====================================================
        # NUMERICAL TRANSFORMS
        # ====================================================

        print("\n" + "=" * 70)
        print("CAUSAL NUMERICAL PREPROCESSING")
        print("=" * 70)

        print(
            f"\nNumerical features:\n"
            f"{numerical}"
        )

        log_cols = (
            self._parse_columns(
                input(
                    "\nNumerical columns to log transform "
                    "(Enter for none):\n> "
                )
            )
        )

        clip_cols = (
            self._parse_columns(
                input(
                    "\nNumerical columns to quantile clip "
                    "(Enter for none):\n> "
                )
            )
        )

        # -----------------------------------------------
        # Validate numerical transforms
        # -----------------------------------------------

        invalid_log = (
            set(log_cols)
            - set(numerical)
        )

        invalid_clip = (
            set(clip_cols)
            - set(numerical)
        )

        if invalid_log:

            raise ValueError(
                "Log-transform columns must "
                "be numerical. "
                f"Invalid: {sorted(invalid_log)}"
            )

        if invalid_clip:

            raise ValueError(
                "Clip columns must "
                "be numerical. "
                f"Invalid: {sorted(invalid_clip)}"
            )

        # ====================================================
        # CONFIG
        # ====================================================

        self.config = {
            "features":
                self.feature_cols,

            "categorical":
                categorical,

            "numerical":
                numerical,

            "boolean":
                boolean,

            "onehot":
                onehot,

            "hash":
                hashing,

            "ordinal":
                [],

            "manual_mapping":
                [],

            "log_transform":
                log_cols,

            "quantile_clip":
                clip_cols,

            "hash_n_features":
                self.hash_n_features,
        }

        self._validate_config()

        self.save_config()

        return self.config

    # ========================================================
    # VALIDATE CONFIG
    # ========================================================

    def _validate_config(
        self,
    ):

        if self.config is None:

            raise ValueError(
                "Causal preprocessing "
                "configuration is empty."
            )

        feature_set = set(
            self.feature_cols
        )

        categorical = set(
            self.config[
                "categorical"
            ]
        )

        numerical = set(
            self.config[
                "numerical"
            ]
        )

        boolean = set(
            self.config[
                "boolean"
            ]
        )

        # -----------------------------------------------
        # Check type overlap
        # -----------------------------------------------

        if (
            categorical
            & numerical
            or categorical
            & boolean
            or numerical
            & boolean
        ):

            raise ValueError(
                "A causal feature cannot belong "
                "to multiple feature types."
            )

        configured = (
            categorical
            | numerical
            | boolean
        )

        missing = (
            feature_set
            - configured
        )

        extra = (
            configured
            - feature_set
        )

        if missing:

            raise ValueError(
                "Causal features missing from "
                "preprocessing configuration: "
                f"{sorted(missing)}"
            )

        if extra:

            raise ValueError(
                "Unknown features in causal "
                "preprocessing configuration: "
                f"{sorted(extra)}"
            )

        # -----------------------------------------------
        # Check categorical encoding
        # -----------------------------------------------

        encoded = (
            set(
                self.config[
                    "onehot"
                ]
            )
            | set(
                self.config[
                    "hash"
                ]
            )
        )

        unencoded_cat = (
            categorical
            - encoded
        )

        if unencoded_cat:

            raise ValueError(
                "Categorical causal features "
                "without an encoding strategy: "
                f"{sorted(unencoded_cat)}"
            )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    def save_config(
        self,
    ):

        directory = os.path.dirname(
            self.config_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            self.config_path,
            "w",
        ) as f:

            json.dump(
                self.config,
                f,
                indent=4,
            )

        print(
            "\n[CAUSAL PREPROCESSING] "
            "Configuration saved:"
        )

        print(
            self.config_path
        )

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    def load_config(
        self,
    ):

        if not os.path.exists(
            self.config_path
        ):
            return None

        with open(
            self.config_path,
            "r",
        ) as f:

            config = json.load(
                f
            )

        # ----------------------------------------------------
        # Important:
        # only reuse config when it matches current features.
        # ----------------------------------------------------

        saved_features = set(
            config.get(
                "features",
                [],
            )
        )

        current_features = set(
            self.feature_cols
        )

        if (
            saved_features
            != current_features
        ):

            print(
                "\n[CAUSAL PREPROCESSING] "
                "Existing configuration does not "
                "match current causal features."
            )

            print(
                "A new configuration will be created."
            )

            return None

        self.config = (
            config
        )

        self._validate_config()

        return self.config

    # ========================================================
    # DISPLAY CONFIG
    # ========================================================

    def display_config(
        self,
    ):

        print("\n" + "=" * 70)
        print("CAUSAL PREPROCESSING PLAN")
        print("=" * 70)

        for key in [
            "categorical",
            "numerical",
            "boolean",
            "onehot",
            "hash",
            "log_transform",
            "quantile_clip",
        ]:

            print(
                f"\n{key}:"
            )

            print(
                self.config.get(
                    key,
                    []
                )
            )

    # ========================================================
    # BUILD SKLEARN PREPROCESSOR
    # ========================================================

    def build(
        self,
    ):

        if self.config is None:

            raise ValueError(
                "No causal preprocessing "
                "configuration available."
            )

        config = (
            self.config
        )

        return build_preprocessor(
            cat_cols=(
                config[
                    "categorical"
                ]
            ),

            num_cols=(
                config[
                    "numerical"
                ]
            ),

            bool_cols=(
                config[
                    "boolean"
                ]
            ),

            ordinal_cols=(
                config[
                    "ordinal"
                ]
            ),

            map_cols=(
                config[
                    "manual_mapping"
                ]
            ),

            onehot_cols=(
                config[
                    "onehot"
                ]
            ),

            hash_cols=(
                config[
                    "hash"
                ]
            ),

            log_cols=(
                config[
                    "log_transform"
                ]
            ),

            clip_cols=(
                config[
                    "quantile_clip"
                ]
            ),

            hash_n_features=(
                config[
                    "hash_n_features"
                ]
            ),
        )

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def run(
        self,
    ):

        existing = (
            self.load_config()
        )

        if existing is not None:

            self.display_config()

            reuse = input(
                "\nReuse this causal preprocessing "
                "configuration? (Y/n):\n> "
            ).strip().lower()

            if reuse in [
                "",
                "y",
                "yes",
            ]:

                print(
                    "\n[CAUSAL PREPROCESSING] "
                    "Using existing configuration."
                )

                return (
                    self.build(),
                    self.config,
                )

        # ----------------------------------------------------
        # New configuration
        # ----------------------------------------------------

        self.create_config()

        self.display_config()

        print(
            "\n[CAUSAL PREPROCESSING] "
            "New configuration ready."
        )

        return (
            self.build(),
            self.config,
        )