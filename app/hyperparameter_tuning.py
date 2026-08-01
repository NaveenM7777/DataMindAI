import streamlit as st
import pandas as pd
import time

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ==========================================================
# Parameter Grids — one per model type
# ==========================================================

PARAM_GRIDS = {

    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"]
    },

    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    },

    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    },

    "Extra Trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    },

    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"]
    },

    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },

    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5]
    },

    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },

    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [15, 31, 63]
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
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
}

# Models that are known to be slow during tuning
SLOW_MODELS = ["SVM", "SVR", "XGBoost", "LightGBM", "Gradient Boosting", "CatBoost"]


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
    Runs GridSearchCV or RandomizedSearchCV on the given model.

    Returns a dict with before_score, after_score, best_params,
    tuned_model, tuning_time, improvement, score_label.
    """

    param_grid = PARAM_GRIDS.get(model_name, {})

    if not param_grid:
        return {
            "error": f"{model_name} has no tunable hyperparameters defined."
        }

    scoring = "accuracy" if problem_type == "Classification" else "r2"

    # ---- BEFORE: score with current model ----
    if problem_type == "Classification":
        from sklearn.metrics import accuracy_score
        before_pred = model.predict(X_test)
        before_score = accuracy_score(y_test, before_pred)
    else:
        from sklearn.metrics import r2_score
        before_pred = model.predict(X_test)
        before_score = r2_score(y_test, before_pred)

    # ---- Run the search ----
    start = time.time()

    if search_type == "Grid Search":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=10,
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
        "before_score":  round(before_score, 4),
        "after_score":   round(after_score, 4),
        "improvement":   round(improvement, 4),
        "best_params":   search.best_params_,
        "tuned_model":   tuned_model,
        "tuning_time":   tuning_time,
        "score_label":   "Accuracy" if problem_type == "Classification" else "R2 Score"
    }