from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Classification Metrics
# ==========================================================

def classification_metrics(y_true, y_pred):

    return {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

    }


# ==========================================================
# Regression Metrics
# ==========================================================

def regression_metrics(y_true, y_pred):

    return {

        "R2 Score": r2_score(
            y_true,
            y_pred
        ),

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "RMSE": mean_squared_error(
            y_true,
            y_pred
        ) ** 0.5

    }


# ==========================================================
# Confusion Matrix
# ==========================================================

def get_confusion_matrix(y_true, y_pred):

    return confusion_matrix(
        y_true,
        y_pred
    )


# ==========================================================
# Classification Report
# ==========================================================

def get_classification_report(y_true, y_pred):

    return classification_report(
        y_true,
        y_pred,
        output_dict=True
    )


# ==========================================================
# ROC AUC
# ==========================================================

def get_roc_auc(y_true, probabilities):

    try:

        return roc_auc_score(
            y_true,
            probabilities
        )

    except:

        return None