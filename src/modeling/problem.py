import pandas as pd


def detect_problem_type(
    y: pd.Series,
) -> str:

    n_unique = y.nunique()

    # Classification-like targets
    if (
        pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        return "classification"

    # Integer targets with relatively few classes
    if (
        pd.api.types.is_integer_dtype(y)
        and n_unique <= 20
    ):
        return "classification"

    return "regression"

def confirm_problem_type(
    y,
):

    detected = detect_problem_type(y)

    print("\n" + "=" * 60)
    print("PROBLEM TYPE")
    print("=" * 60)

    print(f"\nTarget: {y.name}")
    print(f"dtype: {y.dtype}")
    print(f"Unique values: {y.nunique():,}")
    print(f"\nDetected: {detected.upper()}")

    answer = input(
        "\nIs this correct? (Y/n):\n> "
    ).strip().lower()

    if answer in ["", "y", "yes"]:
        return detected

    new_type = input(
        "\nEnter problem type "
        "(classification/regression):\n> "
    ).strip().lower()

    if new_type not in [
        "classification",
        "regression",
    ]:
        raise ValueError(
            "Problem type must be classification or regression."
        )

    return new_type