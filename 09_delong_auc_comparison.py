"""
Paired DeLong test for comparing correlated ROC-AUCs.

This script provides utility functions for:
1. Comparing the final reference model, such as CatBoost, with alternative
   candidate models using paired DeLong tests.
2. Comparing a full model with a reduced model, for example to evaluate the
   incremental predictive contribution of body composition features.

The implementation is intended for binary classification tasks with labels
encoded as 0 and 1.

Author: [Your Name]
Project: Gastric cancer body composition machine-learning study
"""

import numpy as np
import pandas as pd
from math import sqrt, erfc
from sklearn.metrics import roc_auc_score


# =============================================================================
# Core DeLong test functions
# =============================================================================

def compute_midrank(x):
    """
    Compute midranks for a one-dimensional array.

    Parameters
    ----------
    x : array-like
        Prediction scores.

    Returns
    -------
    midranks : numpy.ndarray
        Midranks of the input values.
    """
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # 1-based midrank
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    output = np.empty(n, dtype=float)
    output[order] = midranks
    return output


def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong implementation for correlated ROC-AUCs.

    Parameters
    ----------
    predictions_sorted_transposed : numpy.ndarray
        Array with shape (n_classifiers, n_examples). Samples must be sorted
        so that positive cases appear first.
    label_1_count : int
        Number of positive samples.

    Returns
    -------
    aucs : numpy.ndarray
        AUCs of the classifiers.
    delong_cov : numpy.ndarray
        DeLong covariance matrix.
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m

    if m <= 0 or n <= 0:
        raise ValueError("Both positive and negative samples are required.")

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))

    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def two_sided_pvalue_from_z(z):
    """
    Compute a two-sided P value from a z statistic.

    Parameters
    ----------
    z : float
        Z statistic.

    Returns
    -------
    p_value : float
        Two-sided P value.
    """
    return erfc(abs(z) / sqrt(2))


def delong_roc_test(y_true, y_score_1, y_score_2):
    """
    Perform paired DeLong test for two correlated ROC curves.

    Parameters
    ----------
    y_true : array-like
        Binary labels encoded as 0 and 1.
    y_score_1 : array-like
        Prediction scores or probabilities from model 1.
    y_score_2 : array-like
        Prediction scores or probabilities from model 2.

    Returns
    -------
    result : dict
        A dictionary containing AUCs, AUC difference, z statistic, and P value.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score_1 = np.asarray(y_score_1, dtype=float)
    y_score_2 = np.asarray(y_score_2, dtype=float)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must contain binary labels encoded as 0 and 1.")

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both positive and negative samples.")

    if len(y_true) != len(y_score_1) or len(y_true) != len(y_score_2):
        raise ValueError("y_true and prediction scores must have the same length.")

    # Sort samples so that positive labels appear first
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]

    predictions_sorted = np.vstack([
        y_score_1[order],
        y_score_2[order]
    ])

    label_1_count = int(np.sum(y_true_sorted == 1))

    aucs, delong_cov = fast_delong(
        predictions_sorted_transposed=predictions_sorted,
        label_1_count=label_1_count
    )

    auc_diff = aucs[0] - aucs[1]
    var_diff = (
        delong_cov[0, 0]
        + delong_cov[1, 1]
        - 2 * delong_cov[0, 1]
    )

    if var_diff <= 0:
        z_stat = np.nan
        p_value = np.nan
    else:
        z_stat = auc_diff / np.sqrt(var_diff)
        p_value = two_sided_pvalue_from_z(z_stat)

    return {
        "AUC_Model_1": aucs[0],
        "AUC_Model_2": aucs[1],
        "AUC_Difference": auc_diff,
        "Z": z_stat,
        "P_value": p_value
    }


def holm_adjust(p_values):
    """
    Perform Holm-Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : array-like
        Raw P values.

    Returns
    -------
    adjusted : numpy.ndarray
        Holm-adjusted P values.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    previous_adjusted_p = 0.0

    for rank, idx in enumerate(order):
        adjusted_p = (m - rank) * p_values[idx]
        adjusted_p = max(adjusted_p, previous_adjusted_p)
        adjusted[idx] = min(adjusted_p, 1.0)
        previous_adjusted_p = adjusted[idx]

    return adjusted


# =============================================================================
# Helper functions for model prediction scores
# =============================================================================

def get_model_scores(model, X, pos_label=1):
    """
    Extract prediction scores for ROC-AUC calculation.

    This function prioritizes predict_proba() and falls back to
    decision_function() if probabilities are unavailable.

    Parameters
    ----------
    model : fitted estimator
        Trained binary classification model.
    X : array-like
        Feature matrix.
    pos_label : int or str, default=1
        Positive class label.

    Returns
    -------
    scores : numpy.ndarray
        Prediction scores for the positive class.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            pos_index = classes.index(pos_label) if pos_label in classes else 1
        else:
            pos_index = 1

        return probabilities[:, pos_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise ValueError("The model must have either predict_proba or decision_function.")


# =============================================================================
# Model comparison functions
# =============================================================================

def compare_reference_model_with_delong(
    models,
    X_test,
    y_test,
    reference_model_name="CatBoost",
    pos_label=1,
    apply_holm=True
):
    """
    Compare one reference model with all alternative models using paired DeLong tests.

    Parameters
    ----------
    models : dict
        Dictionary of fitted models, for example:
        {"CatBoost": catboost_model, "Random Forest": rf_model, ...}
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        Test labels encoded as 0 and 1.
    reference_model_name : str, default="CatBoost"
        Name or keyword of the reference model.
    pos_label : int or str, default=1
        Positive class label.
    apply_holm : bool, default=True
        Whether to apply Holm correction to multiple comparisons.

    Returns
    -------
    results_df : pandas.DataFrame
        DeLong comparison results.
    """
    y_test_array = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    y_test_array = y_test_array.astype(int)

    model_names = list(models.keys())
    matched_reference_names = [
        name for name in model_names
        if reference_model_name.lower() in str(name).lower()
    ]

    if len(matched_reference_names) == 0:
        raise ValueError(f"Reference model '{reference_model_name}' was not found.")

    reference_name = matched_reference_names[0]
    reference_model = models[reference_name]
    reference_scores = get_model_scores(reference_model, X_test, pos_label=pos_label)

    results = []

    for model_name, model in models.items():
        if model_name == reference_name:
            continue

        if model is None:
            continue

        comparison_scores = get_model_scores(model, X_test, pos_label=pos_label)

        delong_result = delong_roc_test(
            y_true=y_test_array,
            y_score_1=reference_scores,
            y_score_2=comparison_scores
        )

        results.append({
            "Reference_Model": reference_name,
            "Compared_Model": model_name,
            "AUC_Reference": delong_result["AUC_Model_1"],
            "AUC_Compared": delong_result["AUC_Model_2"],
            "AUC_Difference": delong_result["AUC_Difference"],
            "Z": delong_result["Z"],
            "P_value": delong_result["P_value"]
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    if apply_holm:
        results_df["P_Holm"] = holm_adjust(results_df["P_value"].values)

    results_df = results_df.sort_values("P_value", ascending=True).reset_index(drop=True)
    return results_df


def compare_full_vs_reduced_model_with_delong(
    full_model,
    reduced_model,
    X_test_full,
    X_test_reduced,
    y_test,
    full_model_name="Full model",
    reduced_model_name="Reduced model",
    pos_label=1
):
    """
    Compare a full model and a reduced model using paired DeLong test.

    This function is useful for evaluating the incremental predictive
    contribution of a set of variables, such as body composition features.

    Parameters
    ----------
    full_model : fitted estimator
        Full model fitted with all selected predictors.
    reduced_model : fitted estimator
        Reduced model fitted after removing selected predictors.
    X_test_full : array-like
        Test matrix for the full model.
    X_test_reduced : array-like
        Test matrix for the reduced model.
    y_test : array-like
        Test labels encoded as 0 and 1.
    full_model_name : str, default="Full model"
        Name of the full model.
    reduced_model_name : str, default="Reduced model"
        Name of the reduced model.
    pos_label : int or str, default=1
        Positive class label.

    Returns
    -------
    results_df : pandas.DataFrame
        Full-versus-reduced DeLong comparison result.
    """
    y_test_array = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    y_test_array = y_test_array.astype(int)

    full_scores = get_model_scores(full_model, X_test_full, pos_label=pos_label)
    reduced_scores = get_model_scores(reduced_model, X_test_reduced, pos_label=pos_label)

    delong_result = delong_roc_test(
        y_true=y_test_array,
        y_score_1=full_scores,
        y_score_2=reduced_scores
    )

    results_df = pd.DataFrame([{
        "Model_1": full_model_name,
        "Model_2": reduced_model_name,
        "AUC_Model_1": delong_result["AUC_Model_1"],
        "AUC_Model_2": delong_result["AUC_Model_2"],
        "AUC_Difference": delong_result["AUC_Difference"],
        "Z": delong_result["Z"],
        "P_value": delong_result["P_value"]
    }])

    return results_df


# =============================================================================
# Example usage
# =============================================================================
# The following examples assume that fitted models and test data have already
# been generated in previous scripts.
#
# Example 1: Compare CatBoost with alternative candidate models
#
# df_delong_models = compare_reference_model_with_delong(
#     models=best_models,
#     X_test=X_test_scaled,
#     y_test=y_test,
#     reference_model_name="CatBoost",
#     pos_label=1,
#     apply_holm=True
# )
#
# df_delong_models.to_csv(
#     "delong_catboost_vs_candidate_models.csv",
#     index=False
# )
#
#
# Example 2: Compare full CatBoost model with reduced CatBoost model
#
# df_delong_full_reduced = compare_full_vs_reduced_model_with_delong(
#     full_model=catboost_full_model,
#     reduced_model=catboost_reduced_model,
#     X_test_full=X_test_full_scaled,
#     X_test_reduced=X_test_reduced_scaled,
#     y_test=y_test,
#     full_model_name="Full CatBoost model",
#     reduced_model_name="Reduced CatBoost model",
#     pos_label=1
# )
#
# df_delong_full_reduced.to_csv(
#     "delong_full_vs_reduced_catboost.csv",
#     index=False
# )
