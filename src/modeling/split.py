from sklearn.model_selection import train_test_split


def split_data(
    X,
    y,
    problem_type,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Split X and y into train, validation, and test sets.

    Default:
        Train      = 70%
        Validation = 15%
        Test       = 15%

    For classification:
        stratify is used to preserve class proportions.

    For regression:
        random split is used.
    """

    # --------------------------------------------------
    # Validate split sizes
    # --------------------------------------------------

    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "train_size + val_size + test_size must equal 1."
        )

    # --------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------

    temp_size = val_size + test_size

    stratify = (
        y
        if problem_type == "classification"
        else None
    )

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify,
    )

    # --------------------------------------------------
    # Second split:
    #
    # temporary data -> validation + test
    #
    # With 15/15:
    # 50% of temp -> validation
    # 50% of temp -> test
    # --------------------------------------------------

    test_fraction_of_temp = (
        test_size / temp_size
    )

    temp_stratify = (
        y_temp
        if problem_type == "classification"
        else None
    )

    (
        X_val,
        X_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_of_temp,
        random_state=random_state,
        stratify=temp_stratify,
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    total_rows = len(X)

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    print(
        f"\nTrain:      "
        f"{len(X_train):,} rows "
        f"({len(X_train) / total_rows:.1%})"
    )

    print(
        f"Validation: "
        f"{len(X_val):,} rows "
        f"({len(X_val) / total_rows:.1%})"
    )

    print(
        f"Test:       "
        f"{len(X_test):,} rows "
        f"({len(X_test) / total_rows:.1%})"
    )

    # --------------------------------------------------
    # Classification class distribution
    # --------------------------------------------------

    if problem_type == "classification":

        print("\nClass distribution:")

        print("\nTrain:")
        print(
            y_train
            .value_counts(normalize=True)
            .round(3)
        )

        print("\nValidation:")
        print(
            y_val
            .value_counts(normalize=True)
            .round(3)
        )

        print("\nTest:")
        print(
            y_test
            .value_counts(normalize=True)
            .round(3)
        )

    # --------------------------------------------------
    # Return
    # --------------------------------------------------

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )