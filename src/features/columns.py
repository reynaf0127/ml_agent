import json
import os

import pandas as pd


class ColumnManager:

    def __init__(
        self,
        df: pd.DataFrame,
        path: str = "data/metadata/columns.json",
    ):
        self.df = df
        self.path = path

        self.excl_cols = []
        self.cat_cols = []
        self.num_cols = []
        self.bool_cols = []

    # --------------------------------------------------
    # Ask user for excluded columns
    # --------------------------------------------------

    def ask_excluded_columns(self):
        """
        Ask user to manually specify excluded columns
        in the terminal.
        """

        print("\n" + "=" * 60)
        print("COLUMN SELECTION")
        print("=" * 60)

        print("\nAvailable columns:")

        for col in self.df.columns:
            print(f"  - {col}")

        user_input = input(
            "\nEnter columns to exclude "
            "(comma separated, or press Enter for none):\n> "
        ).strip()

        if not user_input:
            self.excl_cols = []
            return

        self.excl_cols = [
            col.strip()
            for col in user_input.split(",")
            if col.strip()
        ]

        # Validate
        invalid_cols = [
            col
            for col in self.excl_cols
            if col not in self.df.columns
        ]

        if invalid_cols:
            raise ValueError(
                f"Columns not found in dataset: {invalid_cols}"
            )

    # --------------------------------------------------
    # Detect categorical / numerical
    # --------------------------------------------------

    def detect_columns(self):

        self.cat_cols = (
            self.df
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "string",
                    "bool",
                ]
            )
            .columns
            .difference(self.excl_cols)
            .tolist()
        )

        self.num_cols = (
            self.df
            .select_dtypes(
                include="number"
            )
            .columns
            .difference(self.excl_cols)
            .tolist()
        )
    def prepare_for_eda(self):

        self.ask_excluded_columns()
        self.detect_columns()
        return (
            self.excl_cols,
            self.cat_cols,
            self.num_cols,
        )

    def ask_boolean_columns(self):
        """
        After EDA, allow user to reclassify numerical
        columns as boolean columns.
        """

        print("\n" + "=" * 60)
        print("BOOLEAN COLUMN SELECTION")
        print("=" * 60)

        print("\nCurrent numerical columns:")

        for i, col in enumerate(
            self.num_cols,
            start=1,
        ):
            print(f"  {i}. {col}")

        user_input = input(
            "\nAfter reviewing the EDA, enter numerical columns "
            "that should be treated as boolean "
            "(comma separated, or press Enter for none):\n> "
        ).strip()

        if not user_input:
            self.bool_cols = []
            return

        selected = [
            col.strip()
            for col in user_input.split(",")
            if col.strip()
        ]

        invalid = [
            col
            for col in selected
            if col not in self.num_cols
        ]

        if invalid:
            raise ValueError(
                f"These columns are not in num_cols: {invalid}"
            )

        self.bool_cols = selected

        # Remove them from numerical features
        self.num_cols = [
            col
            for col in self.num_cols
            if col not in self.bool_cols
        ]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save(self):

        os.makedirs(
            os.path.dirname(self.path),
            exist_ok=True,
        )

        metadata = {
            "excluded_columns": self.excl_cols,
            "categorical_columns": self.cat_cols,
            "numerical_columns": self.num_cols,
        }

        with open(self.path, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )

        print(
            f"\n[METADATA] Saved to {self.path}"
        )

    # --------------------------------------------------
    # Load existing configuration
    # --------------------------------------------------

    def load(self):

        with open(self.path) as f:
            metadata = json.load(f)

        self.excl_cols = metadata["excluded_columns"]
        self.cat_cols = metadata["categorical_columns"]
        self.num_cols = metadata["numerical_columns"]

    # --------------------------------------------------
    # Main workflow
    # --------------------------------------------------
    def run(self):

        # --------------------------------------------------
        # Existing configuration
        # --------------------------------------------------

        if os.path.exists(self.path):

            self.load()

            print("\n" + "=" * 60)
            print("EXISTING COLUMN CONFIGURATION")
            print("=" * 60)

            self.summary()

            update = input(
                "\nUpdate column configuration? (y/N):\n> "
            ).strip().lower()

            if update not in ["y", "yes"]:

                print("\n[CONFIG] Using existing column configuration.")

                return (
                    self.excl_cols,
                    self.cat_cols,
                    self.num_cols,
                    self.bool_cols,
                )

        # --------------------------------------------------
        # New / updated configuration
        # --------------------------------------------------

        self.ask_excluded_columns()

        self.detect_columns()

        # Don't ask bool yet if you're doing this after EDA.
        # bool selection happens in finalize()

        return (
            self.excl_cols,
            self.cat_cols,
            self.num_cols,
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print("COLUMN SUMMARY")
        print("=" * 60)

        print(
            f"\nExcluded ({len(self.excl_cols)}):"
        )
        print(self.excl_cols)

        print(
            f"\nCategorical ({len(self.cat_cols)}):"
        )
        print(self.cat_cols)

        print(
            f"\nNumerical ({len(self.num_cols)}):"
        )
        print(self.num_cols)

    def finalize(self):
        self._move_numeric_to_categorical()
        self.ask_boolean_columns()
        self.save()
        self.summary()
        return (
            self.excl_cols,
            self.cat_cols,
            self.num_cols,
            self.bool_cols,
        )

    def load_existing(self) -> bool:
        """
        Load existing configuration and ask whether
        the user wants to update it.

        Returns:
            True  -> use existing configuration
            False -> rebuild configuration
        """

        if not os.path.exists(self.path):
            return False

        self.load()

        print("\n" + "=" * 60)
        print("EXISTING COLUMN CONFIGURATION")
        print("=" * 60)

        self.summary()

        update = input(
            "\nDo you want to update column configuration? (y/N):\n> "
        ).strip().lower()

        if update in ["y", "yes"]:
            print("\n[CONFIG] Rebuilding column configuration...")
            return False

        print("\n[CONFIG] Using existing configuration.")

        return True

    def _move_numeric_to_categorical(
        self,
    ):
        """
        Allow numeric-coded categorical variables to be
        manually moved from num_cols to cat_cols.

        Examples:
            campaign_id
            product_id
            zip_code
            cat1, cat2, ...
        """

        print("\n" + "=" * 60)
        print("NUMERIC → CATEGORICAL OVERRIDE")
        print("=" * 60)

        print(
            "\nCurrent numerical columns:"
        )

        for col in self.num_cols:
            print(f"  - {col}")

        value = input(
            "\nEnter numerical columns that should "
            "be treated as categorical "
            "(comma separated, Enter for none):\n> "
        ).strip()

        if not value:
            return

        columns = [
            col.strip()
            for col in value.split(",")
            if col.strip()
        ]

        invalid = [
            col
            for col in columns
            if col not in self.num_cols
        ]

        if invalid:
            raise ValueError(
                "These columns are not currently "
                f"numerical columns: {invalid}"
            )

        for col in columns:

            self.num_cols.remove(
                col
            )

            if col not in self.cat_cols:
                self.cat_cols.append(
                    col
                )

        print(
            "\nMoved to categorical:"
        )

        for col in columns:
            print(f"  - {col}")