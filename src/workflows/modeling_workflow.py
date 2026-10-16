from src.modeling.problem import (
    confirm_problem_type,
)

from src.modeling.models import (
    get_available_models,
)

from src.modeling.split import (
    split_data,
)

from src.modeling.train import (
    train_models,
)

from src.modeling.evaluate import (
    ModelEvaluator,
)

from src.modeling.tune import (
    tune_model,
)

from src.modeling.finalize import (
    finalize_model,
    load_final_model,
)


def prepare_model_data(
    df,
    target_col,
    column_config,
):

    y = df[
        target_col
    ].copy()

    problem_type = (
        confirm_problem_type(
            y
        )
    )

    cat_cols = [
        col
        for col in column_config["cat_cols"]
        if col != target_col
    ]

    num_cols = [
        col
        for col in column_config["num_cols"]
        if col != target_col
    ]

    bool_cols = [
        col
        for col in column_config["bool_cols"]
        if col != target_col
    ]

    feature_cols = (
        cat_cols
        + num_cols
        + bool_cols
    )

    X = df[
        feature_cols
    ].copy()

    print("\n" + "=" * 60)
    print("MODEL DATA")
    print("=" * 60)

    print(
        f"\nTarget: {target_col}"
    )

    print(
        f"Problem type: "
        f"{problem_type}"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"\nNumber of features: "
        f"{len(feature_cols)}"
    )

    updated_config = (
        column_config.copy()
    )

    updated_config[
        "cat_cols"
    ] = cat_cols

    updated_config[
        "num_cols"
    ] = num_cols

    updated_config[
        "bool_cols"
    ] = bool_cols

    return (
        X,
        y,
        problem_type,
        feature_cols,
        updated_config,
    )


def create_model_split(
    X,
    y,
    problem_type,
):

    return split_data(
        X=X,
        y=y,
        problem_type=problem_type,
    )


def run_model_training(
    models,
    preprocessor,
    problem_type,
    X_train,
    y_train,
    X_val,
    y_val,
):

    return train_models(

        models=models,

        preprocessor=preprocessor,

        problem_type=problem_type,

        X_train=X_train,

        y_train=y_train,

        X_val=X_val,

        y_val=y_val,
    )


def evaluate_baseline_models(
    fitted_models,
    problem_type,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print("\n" + "=" * 70)
    print("DETAILED MODEL EVALUATION")
    print("=" * 70)

    print(
        "\nAvailable fitted models:"
    )

    for name in fitted_models:

        print(
            f"  - {name}"
        )

    model_name = input(
        "\nEnter model to evaluate in detail "
        "(Enter to skip):\n> "
    ).strip()

    if not model_name:

        return

    if model_name not in fitted_models:

        print(
            f"\n[EVALUATION] "
            f"Unknown model: {model_name}"
        )

        return

    evaluator = ModelEvaluator(
        problem_type=problem_type
    )

    evaluator.evaluate(

        model_name=model_name,

        pipeline=fitted_models[
            model_name
        ],

        X_train=X_train,

        y_train=y_train,

        X_val=X_val,

        y_val=y_val,
    )


def tune_and_finalize(
    models,
    preprocessor,
    problem_type,
    target_col,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    print(
        "\nAvailable models:"
    )

    for name in models:

        print(
            f"  - {name}"
        )

    model_name = input(
        "\nEnter model to tune "
        "(Enter to skip):\n> "
    ).strip()

    final_model = None
    final_model_name = None
    final_metrics = None
    final_metadata = None

    if model_name:

        if model_name not in models:

            print(
                f"\n[TUNING] "
                f"Unknown model: "
                f"{model_name}"
            )

        else:

            (
                tuned_model,
                tuning_results,
                tuning_search,
            ) = tune_model(

                model_name=model_name,

                model=models[
                    model_name
                ],

                preprocessor=preprocessor,

                problem_type=problem_type,

                X_train=X_train,

                y_train=y_train,
            )

            finalize = input(
                f"\nFinalize {model_name} "
                "using the selected "
                "hyperparameters? (y/N):\n> "
            ).strip().lower()

            if finalize in [
                "y",
                "yes",
            ]:

                (
                    final_model,
                    final_metrics,
                    final_metadata,
                ) = finalize_model(

                    model=tuned_model,

                    model_name=model_name,

                    problem_type=problem_type,

                    target_col=target_col,

                    X_train=X_train,

                    y_train=y_train,

                    X_val=X_val,

                    y_val=y_val,

                    X_test=X_test,

                    y_test=y_test,
                )

                final_model_name = (
                    model_name
                )

    # ========================================================
    # TRY EXISTING FINAL MODEL
    # ========================================================

    if final_model is None:

        (
            loaded_model,
            loaded_model_name,
            loaded_metadata,
        ) = load_final_model()

        if loaded_model is not None:

            saved_target = (
                loaded_metadata.get(
                    "target"
                )
            )

            if (
                saved_target
                and saved_target != target_col
            ):

                print(
                    "\n[MODEL] Existing finalized "
                    "model target does not match "
                    "current target."
                )

                print(
                    f"Saved target: "
                    f"{saved_target}"
                )

                print(
                    f"Current target: "
                    f"{target_col}"
                )

            else:

                final_model = (
                    loaded_model
                )

                final_model_name = (
                    loaded_model_name
                )

                final_metadata = (
                    loaded_metadata
                )

    return (
        final_model,
        final_model_name,
        final_metrics,
        final_metadata,
    )