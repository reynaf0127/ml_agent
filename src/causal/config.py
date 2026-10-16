# src/causal/config.py


class CausalConfig:
    """
    Configuration for causal / uplift modeling.

    Defines:
    - treatment column
    - outcome column
    - group / entity column
    - columns that must never enter the causal feature matrix
    """

    def __init__(
        self,
        treatment_col,
        outcome_col,
        group_col=None,
        time_col=None,
    ):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.group_col = group_col
        self.time_col = time_col

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self, df):

        print("\n" + "=" * 70)
        print("CAUSAL CONFIGURATION")
        print("=" * 70)

        required = [
            self.treatment_col,
            self.outcome_col,
        ]

        if self.group_col:
            required.append(
                self.group_col
            )

        if self.time_col:
            required.append(
                self.time_col
            )

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required causal columns: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Treatment validation
        # ----------------------------------------------------

        treatment_values = (
            df[self.treatment_col]
            .dropna()
            .unique()
        )

        if not set(treatment_values).issubset(
            {0, 1, False, True}
        ):

            raise ValueError(
                f"Treatment column "
                f"'{self.treatment_col}' "
                "must be binary."
            )

        # ----------------------------------------------------
        # Outcome validation
        # ----------------------------------------------------

        outcome_values = (
            df[self.outcome_col]
            .dropna()
            .unique()
        )

        if not set(outcome_values).issubset(
            {0, 1, False, True}
        ):

            raise ValueError(
                f"Outcome column "
                f"'{self.outcome_col}' "
                "must currently be binary."
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            f"\nTreatment: "
            f"{self.treatment_col}"
        )

        print(
            f"Outcome: "
            f"{self.outcome_col}"
        )

        print(
            f"Group: "
            f"{self.group_col}"
        )

        print(
            f"Time: "
            f"{self.time_col}"
        )

        print(
            "\nTreatment distribution:"
        )

        print(
            df[self.treatment_col]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .round(4)
        )

        print(
            "\nOutcome distribution:"
        )

        print(
            df[self.outcome_col]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .round(4)
        )

        print(
            "\nObserved outcome by treatment:"
        )

        observed = (
            df
            .groupby(
                self.treatment_col
            )[self.outcome_col]
            .agg([
                "count",
                "mean",
            ])
        )

        print(
            observed
            .round(4)
            .to_string()
        )

        return True

    # ========================================================
    # COLUMNS THAT MUST NOT BE FEATURES
    # ========================================================

    def protected_columns(self):

        cols = {
            self.treatment_col,
            self.outcome_col,
        }

        if self.group_col:
            cols.add(
                self.group_col
            )

        if self.time_col:
            cols.add(
                self.time_col
            )

        return cols

    # ========================================================
    # BUILD FEATURE LIST
    # ========================================================

    def get_feature_columns(
        self,
        df,
        additional_exclude=None,
    ):

        excluded = set(
            self.protected_columns()
        )

        if additional_exclude:

            excluded.update(
                additional_exclude
            )

        features = [
            col
            for col in df.columns
            if col not in excluded
        ]

        return features