class CausalFeatureSelector:

    def __init__(
        self,
        df,
        config,
    ):
        self.df = df
        self.config = config

    # ========================================================
    # AUTOMATIC EXCLUSIONS
    # ========================================================

    def get_automatic_exclusions(
        self,
    ):

        exclude = {
            self.config.treatment_col,
            self.config.outcome_col,
        }

        if self.config.group_col:
            exclude.add(
                self.config.group_col
            )

        if self.config.time_col:
            exclude.add(
                self.config.time_col
            )

        return exclude

    # ========================================================
    # CANDIDATE FEATURES
    # ========================================================

    def get_candidate_features(
        self,
    ):

        exclude = (
            self.get_automatic_exclusions()
        )

        return [
            col
            for col in self.df.columns
            if col not in exclude
        ]

    # ========================================================
    # USER REVIEW
    # ========================================================

    def run(
        self,
    ):

        print("\n" + "=" * 70)
        print("CAUSAL FEATURE SELECTION")
        print("=" * 70)

        candidates = (
            self.get_candidate_features()
        )

        print(
            "\nCandidate covariates:"
        )

        for col in candidates:
            print(f"  - {col}")

        print(
            "\nCausal covariates should contain ONLY "
            "information available BEFORE treatment."
        )

        print(
            "\nExclude variables that are:"
            "\n  - created after treatment"
            "\n  - consequences of treatment"
            "\n  - future information"
            "\n  - identifiers"
            "\n  - leakage variables"
        )

        exclude_input = input(
            "\nEnter additional columns to exclude "
            "(comma separated, Enter for none):\n> "
        ).strip()

        if exclude_input:

            user_exclude = [
                col.strip()
                for col in exclude_input.split(",")
                if col.strip()
            ]

        else:
            user_exclude = []

        # ----------------------------------------------------
        # Validate names
        # ----------------------------------------------------

        unknown = [
            col
            for col in user_exclude
            if col not in candidates
        ]

        if unknown:

            raise ValueError(
                "Unknown causal feature columns: "
                f"{unknown}"
            )

        # ----------------------------------------------------
        # Final feature set
        # ----------------------------------------------------

        final_features = [
            col
            for col in candidates
            if col not in user_exclude
        ]

        print("\n" + "=" * 70)
        print("FINAL CAUSAL FEATURES")
        print("=" * 70)

        print(
            f"\nFeatures: {len(final_features)}"
        )

        for col in final_features:
            print(f"  - {col}")

        return (
            final_features,
            user_exclude,
        )