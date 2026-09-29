import json
import os

import pandas as pd


class CategoricalPreprocessor:

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        excl_cols: list[str],
        path: str = "data/metadata/categorical_preprocessing.json",
    ):
        self.df = df

        self.cat_cols = cat_cols
        self.excl_cols = excl_cols

        self.path = path

        self.ordinal_cols = []
        self.map_cols = []
        self.onehot_cols = []

    # --------------------------------------------------
    # Select ordinal columns
    # --------------------------------------------------

    def ask_ordinal_columns(self):

        print("\n" + "=" * 60)
        print("ORDINAL COLUMN SELECTION")
        print("=" * 60)

        print("\nAvailable categorical columns:")

        for i, col in enumerate(
            self.cat_cols,
            start=1,
        ):
            print(f"{i}. {col}")

        user_input = input(
            "\nEnter ordinal columns "
            "(comma separated, or Enter for none):\n> "
        ).strip()

        if not user_input:
            self.ordinal_cols = []
            return

        self.ordinal_cols = [
            col.strip()
            for col in user_input.split(",")
            if col.strip()
        ]

        invalid = [
            col
            for col in self.ordinal_cols
            if col not in self.cat_cols
        ]

        if invalid:
            raise ValueError(
                f"Invalid ordinal columns: {invalid}"
            )

    # --------------------------------------------------
    # Select manually mapped columns
    # --------------------------------------------------

    def ask_map_columns(self):

        available = [
            col
            for col in self.cat_cols
            if col not in self.ordinal_cols
        ]

        print("\n" + "=" * 60)
        print("MANUAL MAPPING COLUMN SELECTION")
        print("=" * 60)

        print("\nAvailable columns:")

        for i, col in enumerate(
            available,
            start=1,
        ):
            print(f"{i}. {col}")

        user_input = input(
            "\nEnter columns using manual mappings "
            "(comma separated, or Enter for none):\n> "
        ).strip()

        if not user_input:
            self.map_cols = []
            return

        self.map_cols = [
            col.strip()
            for col in user_input.split(",")
            if col.strip()
        ]

        invalid = [
            col
            for col in self.map_cols
            if col not in available
        ]

        if invalid:
            raise ValueError(
                f"Invalid mapping columns: {invalid}"
            )

    # --------------------------------------------------
    # Determine one-hot columns
    # --------------------------------------------------

    def detect_onehot_columns(self):

        self.onehot_cols = [
            col
            for col in self.cat_cols
            if col not in self.ordinal_cols
            and col not in self.map_cols
            and col not in self.excl_cols
        ]

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print("CATEGORICAL PREPROCESSING PLAN")
        print("=" * 60)

        print("\nOrdinal + scale:")
        print(self.ordinal_cols)

        print("\nManual mapping:")
        print(self.map_cols)

        print("\nOne-hot encoding:")
        print(self.onehot_cols)

        print("\nIgnored:")
        print(self.excl_cols)

    # --------------------------------------------------
    # Save configuration
    # --------------------------------------------------

    def save(self):

        os.makedirs(
            os.path.dirname(self.path),
            exist_ok=True,
        )

        config = {
            "ordinal_columns": self.ordinal_cols,
            "map_columns": self.map_cols,
            "onehot_columns": self.onehot_cols,
        }

        with open(self.path, "w") as f:
            json.dump(
                config,
                f,
                indent=4,
            )

        print(
            f"\n[CONFIG] Saved to {self.path}"
        )

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self):
        # --------------------------------------------------
        # Existing preprocessing configuration
        # --------------------------------------------------
        if os.path.exists(self.path):

            self.load()

            print("\n" + "=" * 60)
            print("EXISTING CATEGORICAL PREPROCESSING PLAN")
            print("=" * 60)

            self.summary()

            update = input(
                "\nDo you want to update this plan? (y/N):\n> "
            ).strip().lower()

            if update not in ["y", "yes"]:

                print(
                    "\n[CONFIG] Using existing "
                    "categorical preprocessing plan."
                )

                return (
                    self.ordinal_cols,
                    self.map_cols,
                    self.onehot_cols,
                )

        # --------------------------------------------------
        # Create / update plan
        # --------------------------------------------------
        self.ask_ordinal_columns()
        self.ask_map_columns()
        self.detect_onehot_columns()
        self.summary()
        self.save()
        return (
            self.ordinal_cols,
            self.map_cols,
            self.onehot_cols,
        )

    def load(self):
        with open(self.path, "r") as f:
            config = json.load(f)
        self.ordinal_cols = config.get(
            "ordinal_columns",
            [],
        )
        self.map_cols = config.get(
            "map_columns",
            [],
        )
        self.onehot_cols = config.get(
            "onehot_columns",
            [],
        )