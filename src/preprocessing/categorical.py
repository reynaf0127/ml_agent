import json
import os


class CategoricalPreprocessor:

    def __init__(
        self,
        df,
        cat_cols,
        excl_cols,
        config_path=(
            "data/metadata/"
            "categorical_preprocessing.json"
        ),
    ):

        self.df = df

        self.cat_cols = [
            col
            for col in cat_cols
            if col not in excl_cols
        ]

        self.excl_cols = excl_cols

        self.config_path = (
            config_path
        )

        self.ordinal_cols = []
        self.map_cols = []
        self.onehot_cols = []
        self.hash_cols = []

    # ========================================================
    # INPUT HELPER
    # ========================================================

    def _parse_columns(
        self,
        value,
    ):

        if not value.strip():
            return []

        cols = [
            col.strip()
            for col in value.split(",")
            if col.strip()
        ]

        invalid = [
            col
            for col in cols
            if col not in self.cat_cols
        ]

        if invalid:

            raise ValueError(
                "Unknown categorical columns: "
                f"{invalid}"
            )

        return cols

    # ========================================================
    # LOAD
    # ========================================================

    def _load(
        self,
    ):

        if not os.path.exists(
            self.config_path
        ):
            return False

        with open(
            self.config_path,
            "r",
        ) as f:

            config = json.load(f)

        self.ordinal_cols = (
            config.get(
                "ordinal_cols",
                [],
            )
        )

        self.map_cols = (
            config.get(
                "map_cols",
                [],
            )
        )

        self.onehot_cols = (
            config.get(
                "onehot_cols",
                [],
            )
        )

        self.hash_cols = (
            config.get(
                "hash_cols",
                [],
            )
        )

        return True

    # ========================================================
    # SAVE
    # ========================================================

    def _save(
        self,
    ):

        os.makedirs(
            os.path.dirname(
                self.config_path
            ),
            exist_ok=True,
        )

        config = {
            "ordinal_cols":
                self.ordinal_cols,

            "map_cols":
                self.map_cols,

            "onehot_cols":
                self.onehot_cols,

            "hash_cols":
                self.hash_cols,
        }

        with open(
            self.config_path,
            "w",
        ) as f:

            json.dump(
                config,
                f,
                indent=4,
            )

        print(
            "\n[CONFIG] "
            f"Saved to {self.config_path}"
        )

    # ========================================================
    # PRINT PLAN
    # ========================================================

    def _print_plan(
        self,
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            "CATEGORICAL PREPROCESSING PLAN"
        )

        print(
            "=" * 60
        )

        print(
            "\nOrdinal + scale:"
        )
        print(
            self.ordinal_cols
        )

        print(
            "\nManual mapping:"
        )
        print(
            self.map_cols
        )

        print(
            "\nOne-hot encoding:"
        )
        print(
            self.onehot_cols
        )

        print(
            "\nFeature hashing:"
        )
        print(
            self.hash_cols
        )

        print(
            "\nIgnored:"
        )
        print(
            self.excl_cols
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def _update(
        self,
    ):

        print(
            "\nAvailable categorical columns:"
        )

        for col in self.cat_cols:

            cardinality = (
                self.df[col]
                .nunique(
                    dropna=False
                )
            )

            print(
                f"  - {col}: "
                f"{cardinality:,} unique"
            )

        print(
            "\nEnter ordinal columns "
            "(comma separated, Enter for none):"
        )

        self.ordinal_cols = (
            self._parse_columns(
                input("> ")
            )
        )

        print(
            "\nEnter manual mapping columns "
            "(comma separated, Enter for none):"
        )

        self.map_cols = (
            self._parse_columns(
                input("> ")
            )
        )

        print(
            "\nEnter feature hashing columns "
            "(comma separated, Enter for none):"
        )

        self.hash_cols = (
            self._parse_columns(
                input("> ")
            )
        )

        # ----------------------------------------
        # Check duplicates
        # ----------------------------------------

        selected = (
            self.ordinal_cols
            + self.map_cols
            + self.hash_cols
        )

        duplicates = {
            col
            for col in selected
            if selected.count(col) > 1
        }

        if duplicates:

            raise ValueError(
                "Columns assigned to multiple "
                "categorical strategies: "
                f"{duplicates}"
            )

        # ----------------------------------------
        # Everything else becomes one-hot
        # ----------------------------------------

        self.onehot_cols = [
            col
            for col in self.cat_cols
            if col not in selected
        ]

        self._save()

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
    ):

        exists = self._load()

        if exists:

            self._print_plan()

            update = input(
                "\nDo you want to update "
                "this plan? (y/N):\n> "
            ).strip().lower()

            if update in [
                "y",
                "yes",
            ]:

                self._update()

        else:

            print(
                "\nNo categorical "
                "preprocessing configuration found."
            )

            self._update()

        self._print_plan()

        return (
            self.ordinal_cols,
            self.map_cols,
            self.onehot_cols,
            self.hash_cols,
        )