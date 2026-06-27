"""
Feature selection, multicollinearity assessment, and nested cross-validation
for CatBoost-based gastric cancer outcome prediction.

This script performs:
1. Train/test split.
2. Standardization of predictors.
3. Spearman correlation analysis.
4. Variance inflation factor (VIF) assessment.
5. RFECV-based feature selection within nested cross-validation.
6. Hyperparameter tuning of CatBoost using randomized search.
7. Feature-selection stability summary.
8. Final CatBoost model fitting on the development training set.

Note:
- This script is provided for reproducibility of the analytical workflow.
- Individual-level patient data are not included in this repository.
- Users should replace the input path and outcome variable according to their own dataset.
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RANDOM_STATE = 42

CATBOOST_PARAM_GRID: Dict[str, List] = {
    "iterations": [100, 200, 300, 500],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.001, 0.01, 0.1, 0.3],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128],
    "random_strength": [0, 0.1, 1],
    "bagging_temperature": [0, 0.5, 1],
    "od_type": ["IncToDec", "Iter"],
    "od_wait": [10, 20, 50],
}


class CatBoostWrapper(CatBoostClassifier):
    """
    Wrapper for CatBoostClassifier to expose feature_importances_,
    which is required by RFECV.
    """

    @property
    def feature_importances_(self):
        if hasattr(self, "get_feature_importance"):
            return self.get_feature_importance()
        raise AttributeError("Model has not been fitted yet.")

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, verbose=False, **kwargs)


# =============================================================================
# Utility functions
# =============================================================================

def load_dataset(input_path: str, outcome_col: str, id_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load dataset and split predictors/outcome.

    Parameters
    ----------
    input_path : str
        Path to the input CSV or Excel file.
    outcome_col : str
        Name of the binary outcome variable.
    id_cols : list[str], optional
        Columns to exclude, such as patient ID.

    Returns
    -------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Binary outcome.
    feature_names : list[str]
        Predictor names.
    """
    input_path = Path(input_path)

    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        data = pd.read_excel(input_path)
    elif input_path.suffix.lower() == ".csv":
        data = pd.read_csv(input_path)
    else:
        raise ValueError("Input file must be .csv, .xlsx, or .xls")

    if outcome_col not in data.columns:
        raise ValueError(f"Outcome column '{outcome_col}' was not found in the dataset.")

    if id_cols is None:
        id_cols = []

    drop_cols = [outcome_col] + [col for col in id_cols if col in data.columns]

    X = data.drop(columns=drop_cols)
    y = data[outcome_col]

    # Convert categorical predictors to dummy variables if present.
    X = pd.get_dummies(X, drop_first=True)

    # Ensure numeric values.
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values were detected in predictors. "
            "Please run preprocessing/imputation before this script."
        )

    feature_names = X.columns.tolist()

    return X, y, feature_names


def compute_spearman_correlations(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> pd.DataFrame:
    """
    Compute Spearman correlation between each predictor and the outcome.

    Also saves the feature-feature Spearman correlation matrix.
    """
    target_correlations = []

    for col in X.columns:
        rho, p_value = spearmanr(X[col], y)
        target_correlations.append({
            "feature": col,
            "spearman_rho_with_outcome": rho,
            "p_value": p_value,
            "abs_rho": abs(rho)
        })

    target_corr_df = pd.DataFrame(target_correlations).sort_values(
        "abs_rho", ascending=False
    )

    corr_matrix = X.corr(method="spearman")

    target_corr_df.to_csv(output_dir / "spearman_target_correlations.csv", index=False)
    corr_matrix.to_csv(output_dir / "spearman_feature_correlation_matrix.csv")

    return target_corr_df


def compute_vif(X: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Compute variance inflation factors for predictors.
    """
    X_with_const = add_constant(X)

    vif_rows = []
    for i, feature in enumerate(X_with_const.columns):
        if feature == "const":
            continue

        try:
            vif_value = variance_inflation_factor(X_with_const.values, i)
        except Exception:
            vif_value = np.nan

        vif_rows.append({
            "feature": feature,
            "VIF": vif_value
        })

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(output_dir / "vif_results.csv", index=False)

    return vif_df


def summarize_collinearity(vif_df: pd.DataFrame) -> None:
    """
    Print a short summary of VIF results.
    """
    high_vif = vif_df[vif_df["VIF"] > 10]
    moderate_vif = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]

    print("\nTop 10 features by VIF:")
    print(vif_df.head(10).to_string(index=False))

    if not high_vif.empty:
        print("\nFeatures with VIF > 10, suggesting severe multivariable collinearity:")
        print(high_vif.to_string(index=False))
    else:
        print("\nNo feature showed VIF > 10.")

    if not moderate_vif.empty:
        print("\nFeatures with 5 < VIF <= 10, suggesting moderate multivariable collinearity:")
        print(moderate_vif.to_string(index=False))


def nested_rfecv_catboost(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    outer_splits: int = 5,
    inner_splits: int = 5,
    rfecv_min_features: int = 5,
    rfecv_step: int = 1,
    n_iter_search: int = 30,
    selection_frequency_threshold: float = 0.60,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict:
    """
    Perform nested CV with RFECV feature selection and CatBoost hyperparameter tuning.

    RFECV and hyperparameter tuning are performed only within the development
    training set. The held-out internal test set should not be used in this step.
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 1)

    y_train = np.asarray(y_train)

    outer_scores = []
    selected_feature_history = []
    fold_summary_rows = []

    print(f"\nStarting nested cross-validation: {outer_splits} outer folds")

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_scaled, y_train), start=1):
        print(f"\nOuter fold {fold}/{outer_splits}")

        X_outer_train = X_train_scaled[train_idx]
        X_outer_val = X_train_scaled[val_idx]
        y_outer_train = y_train[train_idx]
        y_outer_val = y_train[val_idx]

        base_model = CatBoostWrapper(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False
        )

        rfecv = RFECV(
            estimator=clone(base_model),
            step=rfecv_step,
            cv=inner_cv,
            scoring="roc_auc",
            min_features_to_select=rfecv_min_features,
            n_jobs=1
        )

        rfecv.fit(X_outer_train, y_outer_train)

        selected_mask = rfecv.support_
        selected_indices = np.where(selected_mask)[0].tolist()
        selected_feature_history.extend(selected_indices)

        X_outer_train_selected = X_outer_train[:, selected_mask]
        X_outer_val_selected = X_outer_val[:, selected_mask]

        print(f"Selected features in this fold: {len(selected_indices)}")

        random_search = RandomizedSearchCV(
            estimator=CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False
            ),
            param_distributions=CATBOOST_PARAM_GRID,
            n_iter=n_iter_search,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=1,
            random_state=random_state + fold,
            verbose=0
        )

        random_search.fit(X_outer_train_selected, y_outer_train)

        best_model = random_search.best_estimator_
        y_val_prob = best_model.predict_proba(X_outer_val_selected)[:, 1]
        fold_auc = roc_auc_score(y_outer_val, y_val_prob)

        outer_scores.append(fold_auc)

        fold_summary_rows.append({
            "fold": fold,
            "n_selected_features": len(selected_indices),
            "selected_features": "; ".join([feature_names[i] for i in selected_indices]),
            "best_inner_cv_auc": random_search.best_score_,
            "outer_validation_auc": fold_auc,
            "best_params": json.dumps(random_search.best_params_)
        })

        print(f"Best inner CV AUC: {random_search.best_score_:.4f}")
        print(f"Outer validation AUC: {fold_auc:.4f}")

    # Save fold-level summary
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    fold_summary_df.to_csv(output_dir / "nested_cv_fold_summary.csv", index=False)

    # Feature-selection stability
    feature_counter = Counter(selected_feature_history)
    stability_rows = []

    for idx, name in enumerate(feature_names):
        count = feature_counter.get(idx, 0)
        frequency = count / outer_splits
        stability_rows.append({
            "feature_index": idx,
            "feature": name,
            "selected_count": count,
            "selection_frequency": frequency
        })

    feature_stability_df = pd.DataFrame(stability_rows).sort_values(
        ["selection_frequency", "selected_count"], ascending=False
    )

    feature_stability_df.to_csv(output_dir / "rfecv_feature_selection_stability.csv", index=False)

    selected_final_indices = feature_stability_df.loc[
        feature_stability_df["selection_frequency"] >= selection_frequency_threshold,
        "feature_index"
    ].astype(int).tolist()

    if len(selected_final_indices) < rfecv_min_features:
        selected_final_indices = feature_stability_df.head(rfecv_min_features)["feature_index"].astype(int).tolist()

    selected_final_indices = sorted(selected_final_indices)
    selected_final_features = [feature_names[i] for i in selected_final_indices]

    print("\nNested CV summary")
    print(f"Mean outer AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    print(f"Final selected features ({len(selected_final_features)}):")
    print(selected_final_features)

    # Save final selected features
    final_features_df = pd.DataFrame({
        "feature_index": selected_final_indices,
        "feature": selected_final_features
    })
    final_features_df.to_csv(output_dir / "final_selected_features.csv", index=False)

    return {
        "outer_scores": outer_scores,
        "mean_outer_auc": float(np.mean(outer_scores)),
        "std_outer_auc": float(np.std(outer_scores)),
        "feature_stability": feature_stability_df,
        "final_feature_indices": selected_final_indices,
        "final_feature_names": selected_final_features
    }


def tune_final_catboost_model(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    final_feature_indices: List[int],
    output_dir: Path,
    inner_splits: int = 5,
    n_iter_search: int = 50,
    random_state: int = DEFAULT_RANDOM_STATE
) -> CatBoostClassifier:
    """
    Tune and fit the final CatBoost model on the full development training set
    using the final selected features.
    """
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    X_train_final = X_train_scaled[:, final_feature_indices]

    final_search = RandomizedSearchCV(
        estimator=CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False
        ),
        param_distributions=CATBOOST_PARAM_GRID,
        n_iter=n_iter_search,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=1,
        random_state=random_state,
        verbose=0
    )

    final_search.fit(X_train_final, y_train)

    final_model = final_search.best_estimator_

    best_params_path = output_dir / "final_catboost_best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(final_search.best_params_, f, indent=4)

    print("\nFinal CatBoost model tuning completed.")
    print(f"Best CV AUC: {final_search.best_score_:.4f}")
    print(f"Best parameters saved to: {best_params_path}")

    return final_model


def save_run_summary(results: Dict, output_dir: Path) -> None:
    """
    Save overall run summary as JSON.
    """
    summary = {
        "mean_outer_auc": results["mean_outer_auc"],
        "std_outer_auc": results["std_outer_auc"],
        "outer_scores": results["outer_scores"],
        "final_feature_names": results["final_feature_names"]
    }

    with open(output_dir / "feature_selection_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RFECV, VIF, correlation analysis, and CatBoost feature selection."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input_template.csv",
        help="Path to the imputed input dataset."
    )

    parser.add_argument(
        "--outcome",
        type=str,
        default="STATUS3Y",
        help="Name of the binary outcome column."
    )

    parser.add_argument(
        "--id-cols",
        type=str,
        nargs="*",
        default=[],
        help="Optional ID columns to exclude from predictors."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_selection",
        help="Directory to save output files."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of data used as the held-out internal test set."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed."
    )

    parser.add_argument(
        "--n-iter-search",
        type=int,
        default=30,
        help="Number of iterations for randomized hyperparameter search in nested CV."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Feature selection and multicollinearity analysis")
    print("=" * 80)

    X, y, feature_names = load_dataset(
        input_path=args.input,
        outcome_col=args.outcome,
        id_cols=args.id_cols
    )

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} predictors.")
    print(f"Outcome column: {args.outcome}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # Save train/test indices for reproducibility
    split_indices = pd.DataFrame({
        "index": X.index,
        "set": ["train" if idx in X_train.index else "test" for idx in X.index]
    })
    split_indices.to_csv(output_dir / "train_test_split_indices.csv", index=False)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Spearman correlation and VIF are assessed in the development training set.
    print("\nComputing Spearman correlations...")
    target_corr_df = compute_spearman_correlations(X_train, y_train, output_dir)
    print("Top 10 predictors by absolute Spearman correlation with outcome:")
    print(target_corr_df.head(10).to_string(index=False))

    print("\nComputing VIF...")
    vif_df = compute_vif(X_train, output_dir)
    summarize_collinearity(vif_df)

    # Nested RFECV + CatBoost tuning
    results = nested_rfecv_catboost(
        X_train_scaled=X_train_scaled,
        y_train=np.asarray(y_train),
        feature_names=feature_names,
        output_dir=output_dir,
        outer_splits=5,
        inner_splits=5,
        rfecv_min_features=5,
        rfecv_step=1,
        n_iter_search=args.n_iter_search,
        selection_frequency_threshold=0.60,
        random_state=args.random_state
    )

    # Final model tuning on full development training set
    final_model = tune_final_catboost_model(
        X_train_scaled=X_train_scaled,
        y_train=np.asarray(y_train),
        final_feature_indices=results["final_feature_indices"],
        output_dir=output_dir,
        inner_splits=5,
        n_iter_search=50,
        random_state=args.random_state
    )

    save_run_summary(results, output_dir)

    print("\n" + "=" * 80)
    print("Completed.")
    print("=" * 80)
    print("Main outputs:")
    print(f"- Spearman target correlations: {output_dir / 'spearman_target_correlations.csv'}")
    print(f"- Spearman feature correlation matrix: {output_dir / 'spearman_feature_correlation_matrix.csv'}")
    print(f"- VIF results: {output_dir / 'vif_results.csv'}")
    print(f"- Nested CV fold summary: {output_dir / 'nested_cv_fold_summary.csv'}")
    print(f"- RFECV feature stability: {output_dir / 'rfecv_feature_selection_stability.csv'}")
    print(f"- Final selected features: {output_dir / 'final_selected_features.csv'}")
    print(f"- Final CatBoost best parameters: {output_dir / 'final_catboost_best_params.json'}")


if __name__ == "__main__":
    main()
