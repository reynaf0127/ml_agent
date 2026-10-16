from src.modeling.interpret import (
    ModelInterpreter,
)


def run_interpretation(
    final_model,
    final_model_name,
    problem_type,
    X_val,
    y_val,
    output_dir,
):

    if final_model is None:

        return

    print("\n" + "=" * 70)
    print("MODEL INTERPRETATION")
    print("=" * 70)

    print(
        f"\nFinal model: "
        f"{final_model_name}"
    )

    choice = input(
        "\nRun model interpretation? "
        "(Y/n):\n> "
    ).strip().lower()

    if choice not in [
        "",
        "y",
        "yes",
    ]:

        return

    interpreter = ModelInterpreter(
        output_dir=output_dir
    )

    # ========================================================
    # COEFFICIENTS
    # ========================================================

    if final_model_name in [
        "linear_regression",
        "ridge",
        "logistic_regression",
    ]:

        interpreter.linear_coefficients(

            pipeline=final_model,

            model_name=final_model_name,
        )

    # ========================================================
    # PERMUTATION
    # ========================================================

    interpreter.permutation_importance_plot(

        pipeline=final_model,

        model_name=final_model_name,

        X=X_val,

        y=y_val,

        problem_type=problem_type,
    )

    # ========================================================
    # SHAP
    # ========================================================

    run_shap = input(
        "\nRun SHAP interpretation? "
        "(Y/n):\n> "
    ).strip().lower()

    if run_shap not in [
        "",
        "y",
        "yes",
    ]:

        return

    (
        shap_values,
        shap_importance,
        X_shap,
    ) = interpreter.shap_analysis(

        pipeline=final_model,

        model_name=final_model_name,

        X=X_val,

        max_samples=500,

        top_n=15,
    )

    interpreter.shap_dependence_plots(

        shap_values=shap_values,

        shap_importance=shap_importance,

        model_name=final_model_name,

        top_n=None,

        ncols=3,
    )

    explain_row = input(
        "\nEnter SHAP row number "
        "(Enter for row 0):\n> "
    ).strip()

    try:

        explain_row = (
            int(explain_row)
            if explain_row
            else 0
        )

    except ValueError:

        explain_row = 0

    if (
        explain_row < 0
        or explain_row >= len(
            shap_values
        )
    ):

        explain_row = 0

    interpreter.shap_individual_explanation(

        shap_values=shap_values,

        row=explain_row,
    )