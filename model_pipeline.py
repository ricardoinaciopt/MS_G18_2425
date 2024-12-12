import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import os
import json
import argparse
from joblib import dump
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def compute_eod(y_true, y_pred, group):
    tpr_group1 = recall_score(y_true[group == 0], y_pred[group == 0])
    tpr_group2 = recall_score(y_true[group == 1], y_pred[group == 1])
    return tpr_group2 - tpr_group1


def compute_dmr(y_true, y_pred, group):
    fnr_group1 = 1 - recall_score(y_true[group == 0], y_pred[group == 0])
    fnr_group2 = 1 - recall_score(y_true[group == 1], y_pred[group == 1])
    fpr_group1 = 1 - precision_score(y_true[group == 0], y_pred[group == 0])
    fpr_group2 = 1 - precision_score(y_true[group == 1], y_pred[group == 1])
    return (fnr_group2 + fpr_group2) - (fnr_group1 + fpr_group1)


def main(csv_path, use_unbalance=False, scale_pos_weight=None, calibrate=False):
    data = pd.read_csv(csv_path, index_col=[0, 1, 2])
    data["group"] = data.index.get_level_values("group").map({"A": 1, "B": 0})
    data["sex"] = data["sex"].map({"M": 1, "F": 0})

    X = data[
        [
            "wealth",
            "career_years",
            "sex",
            "job_status",
            "has_disease",
            "has_car",
            "has_house",
            "num_children",
            "personal_luxuries",
            "health_care_cost",
        ]
    ]
    y = data["group"]
    data.to_csv("processed_df.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
        "LightGBM": LGBMClassifier(
            random_state=42,
            is_unbalance=use_unbalance,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
        ),
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 6, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 6, 10, None],
            "gamma": [0, 0.1, 0.3],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 6, 10, -1],
            "num_leaves": [31, 50, 100],
        },
    }

    metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "ROC AUC": [],
        "EOD": [],
        "DMR": [],
    }

    classification_reports = {}
    for model_name, model in models.items():
        print(f"\n\nTraining {model_name}...")
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grids[model_name],
            random_state=42,
            n_jobs=-1,
            n_iter=10,
            cv=5,
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Calibration, if specified
        if calibrate:
            best_model = CalibratedClassifierCV(best_model, cv=5, method="sigmoid")
            best_model.fit(X_train, y_train)

        # Save the model
        os.makedirs("models", exist_ok=True)
        dump(
            best_model, f"models/{model_name}_{'calibrated' if calibrate else 'nl'}.pkl"
        )

        # Predictions and evaluation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        eod = compute_eod(y_test, y_pred, y_test)
        dmr = compute_dmr(y_test, y_pred, y_test)

        # Log metrics
        metrics["Model"].append(model_name)
        metrics["Accuracy"].append(accuracy)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["F1 Score"].append(f1)
        metrics["ROC AUC"].append(roc_auc)
        metrics["EOD"].append(eod)
        metrics["DMR"].append(dmr)

        classification_reports[model_name] = classification_report(
            y_test, y_pred, output_dict=True
        )

        print(
            f"Performance Metrics for {model_name}:\n\tAccuracy: {accuracy:.4f}\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1: {f1:.4f}\n\tROC AUC: {roc_auc:.4f}"
        )
        print(
            f"Fairness Metrics:\n\tEqual Opportunity Difference: {eod:.3f}.\n\tMisclassification Rate: {dmr:.3f}"
        )
        print(classification_report(y_test, y_pred))

    # Save metrics results
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open("results/classification_reports.json", "w") as f:
        json.dump(classification_reports, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on simulation data."
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--use_unbalance", action="store_true", help="Use is_unbalance for LightGBM"
    )
    parser.add_argument(
        "--scale_pos_weight",
        type=float,
        help="Set scale_pos_weight for LightGBM and XGBoost",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Use CalibratedClassifierCV for prediction calibration",
    )
    args = parser.parse_args()

    main(
        args.csv_path,
        use_unbalance=args.use_unbalance,
        scale_pos_weight=args.scale_pos_weight,
        calibrate=args.calibrate,
    )
