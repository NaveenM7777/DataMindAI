import time
import pandas as pd
import streamlit as st
# ==========================================================
# Classification Models
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ==========================================================
# Regression Models
# ==========================================================

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ==========================================================
# Metrics
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================================
# Optional Models
# ==========================================================

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False


# ==========================================================
# Get Models
# ==========================================================

def get_models(problem_type, training_mode):

    models = {}

    # ======================================================
    # Classification
    # ======================================================

    if problem_type == "Classification":

        if training_mode == "⚡ Quick Train":

            models = {

                "Logistic Regression":
                    LogisticRegression(max_iter=1000),

            }

        else:

            models = {

                "Logistic Regression":
                    LogisticRegression(max_iter=1000),

                "Decision Tree":
                    DecisionTreeClassifier(random_state=42),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    ),

                "Extra Trees":
                    ExtraTreesClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    ),

                "KNN":
                    KNeighborsClassifier(),

                "SVM":
                    SVC(probability=True),

                "Naive Bayes":
                    GaussianNB(),

                "AdaBoost":
                    AdaBoostClassifier(random_state=42),

                "Gradient Boosting":
                    GradientBoostingClassifier(random_state=42)

            }

            if XGBOOST_AVAILABLE:

                models["XGBoost"] = XGBClassifier(
                    random_state=42,
                    verbosity=0
                )

            if LIGHTGBM_AVAILABLE:

                models["LightGBM"] = LGBMClassifier(
                    random_state=42,
                    verbose=-1
                )

            if CATBOOST_AVAILABLE:

                models["CatBoost"] = CatBoostClassifier(
                    random_state=42,
                    verbose=False
                )

    # ======================================================
    # Regression
    # ======================================================

    else:

        if training_mode == "⚡ Quick Train":

            models = {

                "Linear Regression":
                    LinearRegression(),

                "Decision Tree":
                    DecisionTreeRegressor(random_state=42),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    )

            }

        else:

            models = {

                "Linear Regression":
                    LinearRegression(),

                "Ridge":
                    Ridge(),

                "Lasso":
                    Lasso(),

                "ElasticNet":
                    ElasticNet(),

                "Decision Tree":
                    DecisionTreeRegressor(random_state=42),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    ),

                "Extra Trees":
                    ExtraTreesRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    ),

                "KNN":
                    KNeighborsRegressor(),

                "SVR":
                    SVR(),

                "AdaBoost":
                    AdaBoostRegressor(random_state=42),

                "Gradient Boosting":
                    GradientBoostingRegressor(random_state=42)

            }

            if XGBOOST_AVAILABLE:

                models["XGBoost"] = XGBRegressor(
                    random_state=42,
                    verbosity=0
                )

            if LIGHTGBM_AVAILABLE:

                models["LightGBM"] = LGBMRegressor(
                    random_state=42,
                    verbose=-1
                )

            if CATBOOST_AVAILABLE:

                models["CatBoost"] = CatBoostRegressor(
                    random_state=42,
                    verbose=False
                )

    return models


# ==========================================================
# STOP HERE
# ==========================================================
# ==========================================================
# Train All Models
# ==========================================================

def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type,
    training_mode
):

    models = get_models(
        problem_type,
        training_mode
    )

    results = []

    best_model_name = None
    best_model = None
    best_prediction = None
    best_score = float("-inf")

    # ======================================================
    # Train Every Model
    # ======================================================

    for name, model in models.items():
        print(f"\nTraining Model : {name}")
        try:

            start = time.time()

            model.fit(
                X_train,
                y_train
            )
            print(f"{name} trained successfully")

            prediction = model.predict(
                X_test
            )

            end = time.time()

            training_time = round(
                end - start,
                3
            )

            # ==========================================
            # Classification
            # ==========================================

            if problem_type == "Classification":

                accuracy = accuracy_score(
                    y_test,
                    prediction
                )

                precision = precision_score(
                    y_test,
                    prediction,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    prediction,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    prediction,
                    average="weighted",
                    zero_division=0
                )

                results.append({

                    "Model": name,

                    "Accuracy": round(
                        accuracy,
                        4
                    ),

                    "Precision": round(
                        precision,
                        4
                    ),

                    "Recall": round(
                        recall,
                        4
                    ),

                    "F1 Score": round(
                        f1,
                        4
                    ),

                    "Training Time (s)": training_time

                })

                if accuracy > best_score:

                    best_score = accuracy

                    best_model_name = name

                    best_model = model

                    best_prediction = prediction

            # ==========================================
            # Regression
            # ==========================================

            else:

                r2 = r2_score(
                    y_test,
                    prediction
                )

                mae = mean_absolute_error(
                    y_test,
                    prediction
                )

                rmse = mean_squared_error(
                    y_test,
                    prediction
                )** 0.5

                results.append({

                    "Model": name,

                    "R2 Score": round(
                        r2,
                        4
                    ),

                    "MAE": round(
                        mae,
                        4
                    ),

                    "RMSE": round(
                        rmse,
                        4
                    ),

                    "Training Time (s)": training_time

                })

                if r2 > best_score:

                    best_score = r2

                    best_model_name = name

                    best_model = model

                    best_prediction = prediction

        except Exception as e:

            print("=" * 70)
            print(f"FAILED MODEL : {name}")
            print(f"ERROR TYPE   : {type(e).__name__}")
            print(f"ERROR        : {e}")
            print("=" * 70)

    # ======================================================
    # STOP HERE
    # ======================================================
    # ======================================================
    # Create Results DataFrame
    # ======================================================

    result_df = pd.DataFrame(results)

    if result_df.empty:

        raise Exception(
            "No models were trained successfully."
        )

    # ======================================================
    # Sort Results
    # ======================================================

    if problem_type == "Classification":

        result_df = result_df.sort_values(
            by="Accuracy",
            ascending=False
        )

    else:

        result_df = result_df.sort_values(
            by="R2 Score",
            ascending=False
        )

    result_df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================================
    # Add Rank
    # ======================================================

    result_df.insert(
        0,
        "Rank",
        range(
            1,
            len(result_df) + 1
        )
    )

    # ======================================================
    # Console Output
    # ======================================================

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(result_df)
    print("=" * 70)
    print(f"Total Models Trained : {len(result_df)}")
    print(f"Best Model           : {best_model_name}")
    print("=" * 70)

    # ======================================================
    # Return
    # ======================================================

    return (

        result_df,

        best_model_name,

        best_model,

        best_prediction

    )