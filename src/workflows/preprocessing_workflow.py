import numpy as np

from scipy import sparse

from src.preprocessing.categorical import (
    CategoricalPreprocessor,
)

from src.preprocessing.numerical import (
    NumericalPreprocessor,
)

from src.preprocessing.pipeline import (
    build_preprocessor,
)


def configure_preprocessing(
    df,
    column_config,
):

    categorical = (
        CategoricalPreprocessor(

            df=df,

            cat_cols=column_config[
                "cat_cols"
            ],

            excl_cols=column_config[
                "excl_cols"
            ],
        )
    )

    (
        ordinal_cols,
        map_cols,
        onehot_cols,
        hash_cols,
    ) = categorical.run()

    numerical = (
        NumericalPreprocessor(

            df=df,

            num_cols=column_config[
                "num_cols"
            ],
        )
    )

    (
        log_cols,
        clip_cols,
    ) = numerical.run()

    return {

        "ordinal_cols":
            ordinal_cols,

        "map_cols":
            map_cols,

        "onehot_cols":
            onehot_cols,

        "hash_cols":
            hash_cols,

        "log_cols":
            log_cols,

        "clip_cols":
            clip_cols,
    }


def build_model_preprocessor(
    X,
    column_config,
    preprocessing_config,
):

    valid_features = set(
        X.columns
    )

    def valid(
        columns,
    ):

        return [
            col
            for col in columns
            if col in valid_features
        ]

    return build_preprocessor(

        cat_cols=valid(
            column_config[
                "cat_cols"
            ]
        ),

        num_cols=valid(
            column_config[
                "num_cols"
            ]
        ),

        bool_cols=valid(
            column_config[
                "bool_cols"
            ]
        ),

        ordinal_cols=valid(
            preprocessing_config[
                "ordinal_cols"
            ]
        ),

        map_cols=valid(
            preprocessing_config[
                "map_cols"
            ]
        ),

        onehot_cols=valid(
            preprocessing_config[
                "onehot_cols"
            ]
        ),

        hash_cols=valid(
            preprocessing_config[
                "hash_cols"
            ]
        ),

        log_cols=valid(
            preprocessing_config[
                "log_cols"
            ]
        ),

        clip_cols=valid(
            preprocessing_config[
                "clip_cols"
            ]
        ),

        hash_n_features=256,
    )


def test_preprocessor(
    preprocessor,
    X_train,
    X_val,
    X_test,
):

    print("\n" + "=" * 60)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 60)

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "\nRaw train shape:",
        X_train.shape,
    )

    print(
        "Processed train shape:",
        X_train_processed.shape,
    )

    print(
        "Processed validation shape:",
        X_val_processed.shape,
    )

    print(
        "Processed test shape:",
        X_test_processed.shape,
    )

    is_sparse = sparse.issparse(
        X_train_processed
    )

    print(
        "\nSparse matrix:",
        is_sparse,
    )

    if is_sparse:

        values = (
            X_train_processed.data
        )

    else:

        values = np.asarray(
            X_train_processed
        )

    missing_count = int(
        np.isnan(values).sum()
    )

    infinite_count = int(
        np.isinf(values).sum()
    )

    print(
        "\nMissing values after preprocessing:",
        missing_count,
    )

    print(
        "Infinite values after preprocessing:",
        infinite_count,
    )

    if missing_count > 0:

        raise ValueError(
            "Preprocessing produced missing values."
        )

    if infinite_count > 0:

        raise ValueError(
            "Preprocessing produced infinite values."
        )

    if is_sparse:

        total_cells = (
            X_train_processed.shape[0]
            * X_train_processed.shape[1]
        )

        density = (
            X_train_processed.nnz
            / total_cells
        )

        print(
            "\nNon-zero values:",
            f"{X_train_processed.nnz:,}",
        )

        print(
            "Matrix density:",
            f"{density:.4%}",
        )

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        print(
            "\nNumber of processed features:",
            len(feature_names),
        )

        print(
            "\nFirst 30 processed features:"
        )

        for feature in feature_names[:30]:

            print(
                f"  - {feature}"
            )

    except Exception as e:

        print(
            "\n[PREPROCESSING] "
            "Could not retrieve feature names: "
            f"{e}"
        )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
    )