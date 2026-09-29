import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)

from sklearn.impute import SimpleImputer

from config.category_mappings import (
    CATEGORY_MAPPINGS,
)

from src.preprocessing.transformers import (
    QuantileClipper,
    CategoryMapper,
)


# ============================================================
# LOG TRANSFORM
# ============================================================

def safe_log1p(X):
    """
    Apply log1p to numerical values.

    Assumes selected log columns are appropriate for log
    transformation (normally non-negative right-skewed data).
    """

    return np.log1p(X)


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(
    cat_cols,
    num_cols,
    bool_cols,
    ordinal_cols,
    map_cols,
    onehot_cols,
    log_cols,
    clip_cols,
):
    """
    Build preprocessing pipeline using the feature configuration
    supplied by the current modeling run.

    This function does NOT independently load saved column lists.
    That prevents stale columns or the current target from leaking
    back into the preprocessing pipeline.
    """

    transformers = []

    # ========================================================
    # SAFETY / NORMALIZATION
    # ========================================================

    cat_cols = list(
        dict.fromkeys(cat_cols)
    )

    num_cols = list(
        dict.fromkeys(num_cols)
    )

    bool_cols = list(
        dict.fromkeys(bool_cols)
    )

    ordinal_cols = list(
        dict.fromkeys(ordinal_cols)
    )

    map_cols = list(
        dict.fromkeys(map_cols)
    )

    onehot_cols = list(
        dict.fromkeys(onehot_cols)
    )

    log_cols = list(
        dict.fromkeys(log_cols)
    )

    clip_cols = list(
        dict.fromkeys(clip_cols)
    )

    # ========================================================
    # CATEGORICAL SAFETY
    # ========================================================

    categorical_set = set(
        cat_cols
    )

    ordinal_cols = [
        col
        for col in ordinal_cols
        if col in categorical_set
    ]

    map_cols = [
        col
        for col in map_cols
        if col in categorical_set
    ]

    onehot_cols = [
        col
        for col in onehot_cols
        if col in categorical_set
    ]

    # --------------------------------------------------------
    # Prevent overlap
    # --------------------------------------------------------

    ordinal_set = set(
        ordinal_cols
    )

    map_set = set(
        map_cols
    )

    onehot_cols = [
        col
        for col in onehot_cols
        if (
            col not in ordinal_set
            and col not in map_set
        )
    ]

    # ========================================================
    # ORDINAL COLUMNS
    # ========================================================

    if ordinal_cols:

        ordinal_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown=(
                            "use_encoded_value"
                        ),
                        unknown_value=-1,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "ordinal",
                ordinal_pipeline,
                ordinal_cols,
            )
        )

    # ========================================================
    # MANUALLY MAPPED CATEGORICAL COLUMNS
    # ========================================================

    for col in map_cols:

        if col not in CATEGORY_MAPPINGS:

            raise ValueError(
                f"No CATEGORY_MAPPINGS entry "
                f"found for mapped column: {col}"
            )

        mapping_pipeline = Pipeline(
            steps=[
                (
                    "mapper",
                    CategoryMapper(
                        mappings={
                            col: CATEGORY_MAPPINGS[col]
                        }
                    ),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                f"mapped_{col}",
                mapping_pipeline,
                [col],
            )
        )

    # ========================================================
    # ONE-HOT CATEGORICAL COLUMNS
    # ========================================================

    if onehot_cols:

        onehot_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "onehot",
                onehot_pipeline,
                onehot_cols,
            )
        )

    # ========================================================
    # NUMERICAL COLUMN GROUPS
    # ========================================================

    numerical_set = set(
        num_cols
    )

    log_cols = [
        col
        for col in log_cols
        if col in numerical_set
    ]

    clip_cols = [
        col
        for col in clip_cols
        if col in numerical_set
    ]

    # --------------------------------------------------------
    # Columns requiring BOTH log + clipping
    # --------------------------------------------------------

    log_clip_cols = [
        col
        for col in log_cols
        if col in set(clip_cols)
    ]

    # --------------------------------------------------------
    # Log only
    # --------------------------------------------------------

    log_only_cols = [
        col
        for col in log_cols
        if col not in set(clip_cols)
    ]

    # --------------------------------------------------------
    # Clip only
    # --------------------------------------------------------

    clip_only_cols = [
        col
        for col in clip_cols
        if col not in set(log_cols)
    ]

    # --------------------------------------------------------
    # Standard numerical columns
    # --------------------------------------------------------

    special_num_cols = set(
        log_cols + clip_cols
    )

    standard_num_cols = [
        col
        for col in num_cols
        if col not in special_num_cols
    ]

    # ========================================================
    # STANDARD NUMERICAL
    # ========================================================

    if standard_num_cols:

        standard_numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                standard_numeric_pipeline,
                standard_num_cols,
            )
        )

    # ========================================================
    # LOG ONLY
    # ========================================================

    if log_only_cols:

        log_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "log",
                    FunctionTransformer(
                        safe_log1p,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "log_numeric",
                log_pipeline,
                log_only_cols,
            )
        )

    # ========================================================
    # CLIP ONLY
    # ========================================================

    if clip_only_cols:

        clip_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "clip",
                    QuantileClipper(),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "clip_numeric",
                clip_pipeline,
                clip_only_cols,
            )
        )

    # ========================================================
    # LOG + CLIP
    # ========================================================

    if log_clip_cols:

        log_clip_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "clip",
                    QuantileClipper(),
                ),
                (
                    "log",
                    FunctionTransformer(
                        safe_log1p,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "log_clip_numeric",
                log_clip_pipeline,
                log_clip_cols,
            )
        )

    # ========================================================
    # BOOLEAN
    # ========================================================

    if bool_cols:

        bool_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "boolean",
                bool_pipeline,
                bool_cols,
            )
        )

    # ========================================================
    # FINAL COLUMN TRANSFORMER
    # ========================================================

    if not transformers:

        raise ValueError(
            "No preprocessing transformers "
            "were created. Check feature lists."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,

        # Ignore anything that was not explicitly selected.
        remainder="drop",

        verbose_feature_names_out=False,
    )

    return preprocessor