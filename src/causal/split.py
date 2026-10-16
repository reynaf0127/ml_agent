import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
)


def causal_group_split(
    df,
    group_col,
    treatment_col,
    outcome_col,
    feature_cols,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):

    print("\n" + "=" * 70)
    print("CAUSAL GROUP-AWARE DATA SPLIT")
    print("=" * 70)

    # ========================================================
    # VALIDATION
    # ========================================================

    if not np.isclose(
        train_size + val_size + test_size,
        1.0,
    ):
        raise ValueError(
            "train_size + val_size + test_size "
            "must equal 1."
        )

    required_cols = (
        feature_cols
        + [
            group_col,
            treatment_col,
            outcome_col,
        ]
    )

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns for causal split: "
            f"{missing}"
        )

    # ========================================================
    # MODEL DATA
    # ========================================================

    model_df = (
        df[
            list(
                dict.fromkeys(
                    required_cols
                )
            )
        ]
        .copy()
    )

    groups = model_df[
        group_col
    ]

    # ========================================================
    # TRAIN VS TEMP
    # ========================================================

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        first_split.split(
            model_df,
            groups=groups,
        )
    )

    train_df = (
        model_df
        .iloc[train_idx]
        .copy()
    )

    temp_df = (
        model_df
        .iloc[temp_idx]
        .copy()
    )

    # ========================================================
    # VALIDATION VS TEST
    # ========================================================

    remaining_size = (
        val_size + test_size
    )

    relative_val_size = (
        val_size / remaining_size
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=random_state + 1,
    )

    val_idx, test_idx = next(
        second_split.split(
            temp_df,
            groups=temp_df[group_col],
        )
    )

    val_df = (
        temp_df
        .iloc[val_idx]
        .copy()
    )

    test_df = (
        temp_df
        .iloc[test_idx]
        .copy()
    )

    # ========================================================
    # VERIFY GROUP ISOLATION
    # ========================================================

    train_groups = set(
        train_df[group_col]
    )

    val_groups = set(
        val_df[group_col]
    )

    test_groups = set(
        test_df[group_col]
    )

    train_val_overlap = (
        train_groups
        & val_groups
    )

    train_test_overlap = (
        train_groups
        & test_groups
    )

    val_test_overlap = (
        val_groups
        & test_groups
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise RuntimeError(
            "Group leakage detected between "
            "causal data splits."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summarize_split(
        name,
        split_df,
    ):

        return {
            "split": name,

            "rows":
                len(split_df),

            "groups":
                split_df[
                    group_col
                ].nunique(),

            "treatment_rate":
                split_df[
                    treatment_col
                ].mean(),

            "outcome_rate":
                split_df[
                    outcome_col
                ].mean(),

            "treated_outcome_rate":
                split_df.loc[
                    split_df[
                        treatment_col
                    ] == 1,
                    outcome_col,
                ].mean(),

            "control_outcome_rate":
                split_df.loc[
                    split_df[
                        treatment_col
                    ] == 0,
                    outcome_col,
                ].mean(),
        }

    summary = pd.DataFrame([
        summarize_split(
            "train",
            train_df,
        ),

        summarize_split(
            "validation",
            val_df,
        ),

        summarize_split(
            "test",
            test_df,
        ),
    ])

    print("\nSplit summary:\n")

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nGroup overlap:"
    )

    print(
        f"  Train / validation: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"  Train / test: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"  Validation / test: "
        f"{len(val_test_overlap)}"
    )

    print(
        "\n[CAUSAL] "
        "Group-aware split complete."
    )

    # ========================================================
    # SEPARATE COMPONENTS
    # ========================================================

    def unpack(
        split_df,
    ):

        X = (
            split_df[
                feature_cols
            ]
            .copy()
        )

        treatment = (
            split_df[
                treatment_col
            ]
            .copy()
        )

        outcome = (
            split_df[
                outcome_col
            ]
            .copy()
        )

        groups = (
            split_df[
                group_col
            ]
            .copy()
        )

        return (
            X,
            treatment,
            outcome,
            groups,
        )

    (
        X_train,
        treatment_train,
        outcome_train,
        groups_train,
    ) = unpack(
        train_df
    )

    (
        X_val,
        treatment_val,
        outcome_val,
        groups_val,
    ) = unpack(
        val_df
    )

    (
        X_test,
        treatment_test,
        outcome_test,
        groups_test,
    ) = unpack(
        test_df
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,

        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "treatment_train":
            treatment_train,

        "treatment_val":
            treatment_val,

        "treatment_test":
            treatment_test,

        "outcome_train":
            outcome_train,

        "outcome_val":
            outcome_val,

        "outcome_test":
            outcome_test,

        "groups_train":
            groups_train,

        "groups_val":
            groups_val,

        "groups_test":
            groups_test,

        "summary": summary,
    }