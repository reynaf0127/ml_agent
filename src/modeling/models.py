from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


def get_available_models(
    problem_type,
):

    if problem_type == "classification":

        return {
            "logistic_regression": LogisticRegression(
                max_iter=2000
            ),

            "decision_tree": DecisionTreeClassifier(
                random_state=42
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),

            "gradient_boosting": GradientBoostingClassifier(
                random_state=42
            ),
        }

    return {
        "linear_regression": LinearRegression(),

        "ridge": Ridge(),

        "decision_tree": DecisionTreeRegressor(
            random_state=42
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            random_state=42
        ),
    }