import streamlit as st
import pandas as pd
import time

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ==========================================================
# Parameter Grids — kept small and cloud-safe
# ==========================================================
# Slow models (XGBoost, CatBoost, Gradient Boosting, Random
# Forest, Extra Trees) use deliberately small grids to avoid
# timeouts on constrained cloud environments.

PARAM_GRIDS = {

    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    },

    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    },

    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    },

    "Extra Trees": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    },

    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },

    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },

    "AdaBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 1.0]
    },

    "Gradient Boosting": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },

    "XGBoost": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },

    "LightGBM": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31]
    },

    "CatBoost": {
        "iterations": [100],
        "learning_rate": [0.05, 0.1],
        "depth": [4, 6]
    },

    # ---- Regression versions ----

    "Linear Regression": {},

    "Ridge": {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },

    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1, 10]
    },

    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8]
    },

    "SVR": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
}

# Models known to be slow per-fit — used to warn users and
# force lighter search settings automatically.
SLOW_MODELS = {"XGBoost", "CatBoost", "Gradient Boosting", "Random Forest", "Extra Trees"}


def has_tunable_params(model_name):

    grid = PARAM_GRIDS.get(model_name, {})

    return len(grid) > 0


def is_slow_model(model_name):

    return model_name in SLOW_MODELS


def tune_hyperparameters(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    problem_type,
    search_type="Grid Search",
    cv=3
):
    """
    Runs GridSearchCV or RandomizedSearchCV on the given model using its
    predefined parameter grid. Uses n_jobs=1 throughout to stay safe on
    constrained cloud environments (Streamlit Cloud kills processes that
    request more parallel workers than available, often silently).
    """

    param_grid = PARAM_GRIDS.get(model_name, {})

    if not param_grid:

        return {
            "error": f"{model_name} has no tunable hyperparameters defined."
        }

    scoring = "accuracy" if problem_type == "Classification" else "r2"

    # ---- BEFORE: score with default/current model ----

    if problem_type == "Classification":

        from sklearn.metrics import accuracy_score

        before_pred = model.predict(X_test)
        before_score = accuracy_score(y_test, before_pred)

    else:

        from sklearn.metrics import r2_score

        before_pred = model.predict(X_test)
        before_score = r2_score(y_test, before_pred)

    # ---- Reduce cv folds automatically for slow models ----

    effective_cv = 2 if is_slow_model(model_name) else cv

    # ---- Run the search ----

    start = time.time()

    if search_type == "Grid Search":

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=effective_cv,
            n_jobs=1
        )

    else:  # Randomized Search — faster for large grids

        n_iter = 5 if is_slow_model(model_name) else 8

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            cv=effective_cv,
            n_iter=n_iter,
            n_jobs=1,
            random_state=42
        )

    search.fit(X_train, y_train)

    tuning_time = round(time.time() - start, 3)

    tuned_model = search.best_estimator_

    # ---- AFTER: score with tuned model ----

    after_pred = tuned_model.predict(X_test)

    if problem_type == "Classification":

        after_score = accuracy_score(y_test, after_pred)

    else:

        after_score = r2_score(y_test, after_pred)

    improvement = after_score - before_score

    return {
        "before_score": round(before_score, 4),
        "after_score": round(after_score, 4),
        "improvement": round(improvement, 4),
        "best_params": search.best_params_,
        "tuned_model": tuned_model,
        "tuning_time": tuning_time,
        "score_label": "Accuracy" if problem_type == "Classification" else "R2 Score"
    }