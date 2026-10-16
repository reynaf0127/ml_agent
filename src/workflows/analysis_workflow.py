from src.causal.config import CausalConfig


def choose_analysis_type():

    print("\n" + "=" * 70)
    print("ANALYSIS OBJECTIVE")
    print("=" * 70)

    print(
        """
1. Prediction
2. Causal / uplift
3. Both
"""
    )

    analysis_choice = input(
        "Select analysis type [1/2/3]: "
    ).strip()

    analysis_map = {
        "1": "prediction",
        "2": "causal",
        "3": "both",
    }

    analysis_type = (
        analysis_map.get(
            analysis_choice
        )
    )

    if analysis_type is None:

        raise ValueError(
            "Please select 1, 2, or 3."
        )

    print(
        f"\nSelected analysis: "
        f"{analysis_type}"
    )

    return analysis_type


def configure_causal_analysis(
    df,
    analysis_type,
):

    if analysis_type not in [
        "causal",
        "both",
    ]:

        return None, None

    print("\n" + "=" * 70)
    print("CAUSAL MODEL CONFIGURATION")
    print("=" * 70)

    treatment_col = input(
        "\nEnter treatment column:\n> "
    ).strip()

    outcome_col = input(
        "\nEnter outcome column:\n> "
    ).strip()

    group_col = input(
        "\nEnter group/entity column "
        "(Enter for none):\n> "
    ).strip()

    time_col = input(
        "\nEnter decision-time column "
        "(Enter for none):\n> "
    ).strip()

    causal_config = CausalConfig(

        treatment_col=treatment_col,

        outcome_col=outcome_col,

        group_col=(
            group_col or None
        ),

        time_col=(
            time_col or None
        ),
    )

    causal_config.validate(
        df
    )

    causal_features = (
        causal_config
        .get_feature_columns(
            df=df
        )
    )

    print("\n" + "=" * 70)
    print("CAUSAL FEATURES")
    print("=" * 70)

    for feature in causal_features:

        print(
            f"  - {feature}"
        )

    return (
        causal_config,
        causal_features,
    )