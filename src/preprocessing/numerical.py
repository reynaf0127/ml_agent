import json
import os


class NumericalPreprocessor:

    def __init__(
        self,
        df,
        num_cols,
        path="data/metadata/numerical_preprocessing.json",
    ):

        self.df = df
        self.num_cols = num_cols
        self.path = path

        self.log_cols = []
        self.clip_cols = []

    def ask_log_columns(self):

        print("\n" + "=" * 60)
        print("LOG TRANSFORMATION")
        print("=" * 60)

        print("\nNumerical columns:")

        for col in self.num_cols:
            print(f"  - {col}")

        user_input = input(
            "\nEnter columns to log transform "
            "(comma separated, Enter for none):\n> "
        ).strip()

        if not user_input:
            self.log_cols = []
            return

        self.log_cols = [
            x.strip()
            for x in user_input.split(",")
            if x.strip()
        ]

        invalid = [
            col
            for col in self.log_cols
            if col not in self.num_cols
        ]

        if invalid:
            raise ValueError(
                f"Invalid numerical columns: {invalid}"
            )

    def ask_clip_columns(self):

        print("\n" + "=" * 60)
        print("OUTLIER CLIPPING")
        print("=" * 60)

        user_input = input(
            "\nEnter columns to quantile clip "
            "(comma separated, Enter for none):\n> "
        ).strip()

        if not user_input:
            self.clip_cols = []
            return

        self.clip_cols = [
            x.strip()
            for x in user_input.split(",")
            if x.strip()
        ]

        invalid = [
            col
            for col in self.clip_cols
            if col not in self.num_cols
        ]

        if invalid:
            raise ValueError(
                f"Invalid numerical columns: {invalid}"
            )

    def save(self):

        os.makedirs(
            os.path.dirname(self.path),
            exist_ok=True,
        )

        config = {
            "log_columns": self.log_cols,
            "clip_columns": self.clip_cols,

            "clip_quantiles": {
                "lower": 0.01,
                "upper": 0.99,
            },

            "scaler": "standard",
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

    def run(self):

        # --------------------------------------------------
        # Existing configuration
        # --------------------------------------------------

        if os.path.exists(self.path):

            with open(self.path, "r") as f:
                config = json.load(f)

            self.log_cols = config.get(
                "log_columns",
                [],
            )

            self.clip_cols = config.get(
                "clip_columns",
                [],
            )

            print("\n" + "=" * 60)
            print("EXISTING NUMERICAL PREPROCESSING PLAN")
            print("=" * 60)

            print("\nLog transform:")
            print(self.log_cols)

            print("\nQuantile clip:")
            print(self.clip_cols)

            print("\nScaler:")
            print(config.get("scaler", "standard"))

            update = input(
                "\nDo you want to update this plan? (y/N):\n> "
            ).strip().lower()

            if update not in ["y", "yes"]:

                print(
                    "\n[CONFIG] Using existing numerical "
                    "preprocessing plan."
                )

                return (
                    self.log_cols,
                    self.clip_cols,
                )

        # --------------------------------------------------
        # Create / update plan
        # --------------------------------------------------

        self.ask_log_columns()

        self.ask_clip_columns()

        self.save()

        return (
            self.log_cols,
            self.clip_cols,
        )