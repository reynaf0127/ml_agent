from src.features.columns import ColumnManager
from src.profiling.profiler import DatasetProfiler
from src.profiling.categorical_eda import CategoricalEDA
from src.profiling.numerical_eda import NumericalEDA
from src.profiling.correlation import CorrelationAnalysis


def run_profile(
    df,
):

    print(
        f"\nDataset loaded: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    profiler = DatasetProfiler(
        df
    )

    profiler.run()


def run_eda(
    df,
):

    column_manager = ColumnManager(
        df
    )

    use_existing = (
        column_manager.load_existing()
    )

    if use_existing:

        excl_cols = (
            column_manager.excl_cols
        )

        cat_cols = (
            column_manager.cat_cols
        )

        num_cols = (
            column_manager.num_cols
        )

        bool_cols = (
            column_manager.bool_cols
        )

    else:

        (
            excl_cols,
            cat_cols,
            num_cols,
        ) = column_manager.prepare_for_eda()

        bool_cols = []

    categorical_eda = CategoricalEDA(
        df=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

    categorical_eda.run()

    numerical_eda = NumericalEDA(
        df=df,
        num_cols=num_cols,
    )

    numerical_eda.run()

    if not use_existing:

        (
            excl_cols,
            cat_cols,
            num_cols,
            bool_cols,
        ) = column_manager.finalize()

    return {
        "excl_cols":
            excl_cols,

        "cat_cols":
            cat_cols,

        "num_cols":
            num_cols,

        "bool_cols":
            bool_cols,
    }


def run_correlation(
    df,
    column_config,
):

    analysis = CorrelationAnalysis(

        df=df,

        num_cols=column_config[
            "num_cols"
        ],

        bool_cols=column_config[
            "bool_cols"
        ],

        excl_cols=column_config[
            "excl_cols"
        ],
    )

    target_col, corr, vif = (
        analysis.run()
    )

    return (
        target_col,
        corr,
        vif,
    )