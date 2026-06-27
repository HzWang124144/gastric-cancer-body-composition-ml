"""
Hyperparameter tuning and model training for gastric cancer body composition ML study.

This script performs hyperparameter tuning for candidate machine-learning models
using 5-fold stratified cross-validation and ROC-AUC as the optimization metric.

Candidate models:
- Logistic Regression
- Support Vector Classifier
- Multilayer Perceptron
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Notes:
- This script assumes that X and y have already been prepared after preprocessing
  and imputation.
- y should be a binary outcome coded as 0/1.
- The original patient-level dataset is not included in this public repository
  because of privacy and ethical restrictions.
"""

import os
import time
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Global settings
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

USE_RANDOMIZED_SEARCH = True
RANDOM_SEARCH_ITER = 50

OUTPUT_DIR = "outputs/model_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. Input data
# =============================================================================
# X and y should be generated from the preprocessing/imputation script.
# Example:
# X = pd.read_csv("data/processed_features.csv")
# y = pd.read_csv("data/outcome.csv")["outcome"]

# The following two lines are placeholders and should be replaced by authorized users.
# X = ...
# y = ...

if "X" not in globals() or "y" not in globals():
    raise ValueError(
        "X and y are not defined. Please load the preprocessed feature matrix X "
        "and binary outcome vector y before running this script."
    )


# =============================================================================
# 3. Train-test split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("=" * 80)
print("Train-test split")
print("=" * 80)
print(f"Training set: {X_train.shape}")
print(f"Internal test set: {X_test.shape}")
print(f"Training positive cases: {np.sum(y_train == 1)}")
print(f"Training negative cases: {np.sum(y_train == 0)}")
print(f"Test positive cases: {np.sum(y_test == 1)}")
print(f"Test negative cases: {np.sum(y_test == 0)}")


# =============================================================================
# 4. Cross-validation strategy
# =============================================================================

cv_strategy = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# =============================================================================
# 5. Candidate models
# =============================================================================
# Scaling is included inside Pipeline for models that are sensitive to feature scale.
# Tree-based models are trained without standardization.

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=10000,
            random_state=RANDOM_STATE
        ))
    ]),

    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            probability=True,
            random_state=RANDOM_STATE,
            max_iter=10000
        ))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=1000
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    ),

    "CatBoost": CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=0,
        eval_metric="AUC"
    )
}


# =============================================================================
# 6. Hyperparameter search spaces
# =============================================================================

param_grids = {
    "Logistic Regression": [
        {
            "clf__solver": ["liblinear"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": np.logspace(-3, 3, 7),
            "clf__class_weight": [None, "balanced"]
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": np.logspace(-3, 3, 7),
            "clf__class_weight": [None, "balanced"]
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["elasticnet"],
            "clf__C": np.logspace(-3, 3, 7),
            "clf__l1_ratio": [0.25, 0.50, 0.75],
            "clf__class_weight": [None, "balanced"]
        }
    ],

    "SVC": [
        {
            "clf__kernel": ["linear"],
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"]
        },
        {
            "clf__kernel": ["rbf"],
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "clf__class_weight": [None, "balanced"]
        },
        {
            "clf__kernel": ["poly"],
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "clf__degree": [2, 3, 4],
            "clf__coef0": [0.0, 0.5, 1.0],
            "clf__class_weight": [None, "balanced"]
        }
    ],

    "MLP": {
        "clf__hidden_layer_sizes": [
            (50,),
            (100,),
            (50, 50),
            (100, 50),
            (100, 100, 50)
        ],
        "clf__activation": ["relu", "tanh", "logistic"],
        "clf__solver": ["adam", "sgd"],
        "clf__alpha": [0.0001, 0.001, 0.01, 0.1],
        "clf__batch_size": [32, 64, 128, "auto"],
        "clf__learning_rate": ["constant", "invscaling", "adaptive"],
        "clf__learning_rate_init": [0.001, 0.01, 0.1],
        "clf__max_iter": [500, 1000, 2000],
        "clf__early_stopping": [True, False],
        "clf__validation_fraction": [0.1, 0.2],
        "clf__beta_1": [0.9, 0.95, 0.99],
        "clf__beta_2": [0.999, 0.9999]
    },

    "Random Forest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced"]
    },

    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "reg_lambda": [1, 1.5, 2, 3],
        "scale_pos_weight": [1, 2, 3]
    },

    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 7, -1],
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
        "boosting_type": ["gbdt", "dart"],
        "objective": ["binary"],
        "scale_pos_weight": [1, 2, 3]
    },

    "CatBoost": {
        "iterations": [100, 200, 300, 500],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.001, 0.01, 0.1, 0.3],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [32, 64, 128],
        "random_strength": [0, 0.1, 1],
        "bagging_temperature": [0, 0.5, 1],
        "loss_function": ["Logloss", "CrossEntropy"]
    }
}


# =============================================================================
# 7. Hyperparameter tuning
# =============================================================================

best_models = {}
search_results = {}
summary_rows = []

print("\n" + "=" * 80)
print("Hyperparameter tuning started")
print("=" * 80)

for model_name, model in models.items():
    print("\n" + "-" * 80)
    print(f"Tuning model: {model_name}")
    print("-" * 80)

    start_time = time.time()

    if USE_RANDOMIZED_SEARCH:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[model_name],
            n_iter=RANDOM_SEARCH_ITER,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
            return_train_score=True,
            error_score=np.nan
        )
    else:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score=np.nan
        )

    try:
        search.fit(X_train, y_train)

        elapsed_time = time.time() - start_time

        best_models[model_name] = search.best_estimator_
        search_results[model_name] = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
            "time_seconds": float(elapsed_time),
            "cv_results": search.cv_results_
        }

        summary_rows.append({
            "Model": model_name,
            "Best_CV_AUC": search.best_score_,
            "Time_seconds": elapsed_time,
            "Best_params": search.best_params_
        })

        print(f"Best CV AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Tuning time: {elapsed_time:.2f} seconds")

    except Exception as e:
        elapsed_time = time.time() - start_time

        best_models[model_name] = None
        search_results[model_name] = {
            "error": str(e),
            "time_seconds": float(elapsed_time)
        }

        summary_rows.append({
            "Model": model_name,
            "Best_CV_AUC": np.nan,
            "Time_seconds": elapsed_time,
            "Best_params": f"Tuning failed: {str(e)}"
        })

        print(f"Model tuning failed for {model_name}: {str(e)}")
        continue


# =============================================================================
# 8. Save tuning results
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        by="Best_CV_AUC",
        ascending=False,
        na_position="last"
    )

summary_path = os.path.join(OUTPUT_DIR, "hyperparameter_tuning_summary.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("Hyperparameter tuning completed")
print("=" * 80)
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: {summary_path}")


# Save best parameters in JSON-friendly format
best_params_export = {}

for model_name, result in search_results.items():
    if "best_params" in result:
        best_params_export[model_name] = {
            "best_cv_auc": result["best_score"],
            "best_params": result["best_params"],
            "time_seconds": result["time_seconds"]
        }
    else:
        best_params_export[model_name] = {
            "error": result.get("error", "Unknown error"),
            "time_seconds": result["time_seconds"]
        }

best_params_path = os.path.join(OUTPUT_DIR, "best_hyperparameters.json")

with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(best_params_export, f, ensure_ascii=False, indent=4)

print(f"Best hyperparameters saved to: {best_params_path}")


# Save fitted best models
models_path = os.path.join(OUTPUT_DIR, "best_models.joblib")
joblib.dump(best_models, models_path)

print(f"Best fitted models saved to: {models_path}")


# Save train-test split objects for downstream evaluation
split_path = os.path.join(OUTPUT_DIR, "train_test_split.joblib")
joblib.dump(
    {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    },
    split_path
)

print(f"Train-test split saved to: {split_path}")


# =============================================================================
# 9. Optional: save cross-validation results for each model
# =============================================================================

cv_results_dir = os.path.join(OUTPUT_DIR, "cv_results")
os.makedirs(cv_results_dir, exist_ok=True)

for model_name, result in search_results.items():
    if "cv_results" in result:
        cv_df = pd.DataFrame(result["cv_results"])
        safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
        cv_path = os.path.join(cv_results_dir, f"{safe_model_name}_cv_results.csv")
        cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")

print(f"Cross-validation results saved to: {cv_results_dir}")
