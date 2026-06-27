# ============================================================================
# Threshold Selection and Decision Curve Analysis
# ============================================================================
# This script evaluates candidate machine-learning models across decision
# thresholds and performs decision curve analysis (DCA).
#
# Expected inputs from the model-training workflow:
#   best_models: dict
#       Dictionary of trained candidate models, e.g.,
#       {"CatBoost": cat_model, "XGBoost": xgb_model, ...}
#
#   X_test_scaled: array-like
#       Internal test-set predictors after preprocessing/scaling.
#
#   y_test: array-like
#       Binary outcome labels in the internal test set.
#
# Notes:
#   - For recurrence prediction, the selected operating threshold was 0.35.
#   - For 3-year survival prediction, the selected operating threshold was 0.30.
#   - Thresholds are exploratory and require external validation before
#     clinical implementation.
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# User-defined settings
# ============================================================================

TARGET_MODEL_NAME = "CatBoost"

# Change this according to endpoint:
#   recurrence prediction: 0.35
#   3-year survival prediction: 0.30
SELECTED_THRESHOLD = 0.35

OUTPUT_DIR = "outputs/threshold_selection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds for classification metrics
METRIC_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Thresholds for decision curve analysis
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.99, 0.01), 2)


# ============================================================================
# Utility functions
# ============================================================================

def get_predicted_probability(model, X):
    """
    Return predicted probability for the positive class.

    If the model does not support predict_proba, decision_function values are
    rescaled to [0, 1]. This is used for threshold-based comparison only.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(X)
        scaler = MinMaxScaler()
        y_prob = scaler.fit_transform(decision_values.reshape(-1, 1)).ravel()
    else:
        raise ValueError("The model must support predict_proba or decision_function.")

    return np.asarray(y_prob)


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    """
    Calculate classification metrics at a given decision threshold.
    """
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }


def evaluate_model_across_thresholds(model, model_name, X_test, y_test, thresholds):
    """
    Evaluate one trained model across a set of decision thresholds.
    """
    y_prob = get_predicted_probability(model, X_test)

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    rows = []
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_test, y_prob, threshold)
        metrics["Model"] = model_name
        metrics["AUC"] = auc
        metrics["PR_AUC"] = pr_auc
        rows.append(metrics)

    return pd.DataFrame(rows)


def evaluate_all_models_across_thresholds(best_models, X_test, y_test, thresholds):
    """
    Evaluate all candidate models across decision thresholds.
    """
    all_results = []

    for model_name, model in best_models.items():
        if model is None:
            continue

        model_df = evaluate_model_across_thresholds(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            thresholds=thresholds
        )
        all_results.append(model_df)

    return pd.concat(all_results, axis=0, ignore_index=True)


def calculate_net_benefit(y_true, y_prob, thresholds):
    """
    Calculate model net benefit across decision thresholds.

    Net benefit = TP / n - FP / n * threshold / (1 - threshold)
    """
    y_true = np.asarray(y_true)
    n = len(y_true)

    net_benefits = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        if threshold <= 0 or threshold >= 1:
            net_benefit = np.nan
        else:
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))

        net_benefits.append(net_benefit)

    return np.asarray(net_benefits)


def calculate_treat_all_net_benefit(y_true, thresholds):
    """
    Calculate net benefit for treat-all strategy.
    """
    y_true = np.asarray(y_true)
    prevalence = np.mean(y_true)

    treat_all = []
    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            treat_all.append(np.nan)
        else:
            nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            treat_all.append(nb)

    return np.asarray(treat_all)


def calculate_decision_curve_results(best_models, X_test, y_test, thresholds):
    """
    Calculate DCA net-benefit curves for all candidate models.
    """
    dca_results = {}

    for model_name, model in best_models.items():
        if model is None:
            continue

        y_prob = get_predicted_probability(model, X_test)
        dca_results[model_name] = calculate_net_benefit(y_test, y_prob, thresholds)

    dca_results["Treat all"] = calculate_treat_all_net_benefit(y_test, thresholds)
    dca_results["Treat none"] = np.zeros_like(thresholds, dtype=float)

    return dca_results


def mask_to_threshold_ranges(thresholds, mask):
    """
    Convert a boolean mask over thresholds into continuous threshold ranges.
    """
    thresholds = np.asarray(thresholds)
    mask = np.asarray(mask, dtype=bool)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return "None"

    ranges = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((thresholds[start], thresholds[prev]))
            start = i
            prev = i

    ranges.append((thresholds[start], thresholds[prev]))

    formatted = []
    for low, high in ranges:
        if np.isclose(low, high):
            formatted.append(f"{low:.2f}")
        else:
            formatted.append(f"{low:.2f}-{high:.2f}")

    return "; ".join(formatted)


def summarize_net_benefit_ranges(dca_results, thresholds, target_model_name, clinical_range=(0.10, 0.60)):
    """
    Summarize threshold ranges where the target model has higher net benefit
    than both treat-all and treat-none strategies.
    """
    model_key = None
    for name in dca_results.keys():
        if target_model_name.lower() in name.lower():
            model_key = name
            break

    if model_key is None:
        raise ValueError(f"Target model '{target_model_name}' not found in DCA results.")

    model_nb = np.asarray(dca_results[model_key])
    treat_all_nb = np.asarray(dca_results["Treat all"])
    treat_none_nb = np.asarray(dca_results["Treat none"])

    better_than_none = model_nb > treat_none_nb
    better_than_all = model_nb > treat_all_nb
    better_than_both = better_than_none & better_than_all

    clinical_mask = (
        (thresholds >= clinical_range[0]) &
        (thresholds <= clinical_range[1])
    )

    better_than_both_clinical = better_than_both & clinical_mask

    summary = pd.DataFrame([{
        "Model": model_key,
        "Range_GT_TreatNone": mask_to_threshold_ranges(thresholds, better_than_none),
        "Range_GT_TreatAll": mask_to_threshold_ranges(thresholds, better_than_all),
        "Range_GT_Both_AllThresholds": mask_to_threshold_ranges(thresholds, better_than_both),
        f"Range_GT_Both_Clinical_{clinical_range[0]:.2f}_{clinical_range[1]:.2f}": mask_to_threshold_ranges(
            thresholds,
            better_than_both_clinical
        )
    }])

    return summary


# ============================================================================
# Plotting functions
# ============================================================================

def plot_threshold_metrics(metrics_df, target_model_name, selected_threshold, output_dir):
    """
    Plot threshold-dependent classification metrics for the target model.
    """
    model_df = metrics_df[metrics_df["Model"].str.lower() == target_model_name.lower()].copy()

    if model_df.empty:
        raise ValueError(f"Target model '{target_model_name}' not found in metrics table.")

    plt.figure(figsize=(10, 7), facecolor="white")

    for metric in ["Accuracy", "Precision", "Recall", "F1", "Specificity"]:
        plt.plot(
            model_df["Threshold"],
            model_df[metric],
            marker="o",
            linewidth=2,
            label=metric
        )

    plt.axvline(
        x=selected_threshold,
        linestyle="--",
        linewidth=2,
        label=f"Selected threshold = {selected_threshold:.2f}"
    )

    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title(f"Threshold-dependent metrics for {target_model_name}")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{target_model_name}_threshold_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def plot_decision_curve(dca_results, thresholds, target_model_name, selected_threshold, output_dir):
    """
    Plot decision curve analysis results.
    """
    plt.figure(figsize=(10, 7), facecolor="white")

    for model_name, net_benefits in dca_results.items():
        if model_name.lower() == target_model_name.lower():
            plt.plot(
                thresholds,
                net_benefits,
                linewidth=3,
                label=f"{model_name} (final model)"
            )
        elif model_name in ["Treat all", "Treat none"]:
            plt.plot(
                thresholds,
                net_benefits,
                linestyle="--",
                linewidth=2,
                label=model_name
            )
        else:
            plt.plot(
                thresholds,
                net_benefits,
                linewidth=1.5,
                alpha=0.6,
                label=model_name
            )

    plt.axvline(
        x=selected_threshold,
        linestyle=":",
        linewidth=2,
        label=f"Selected threshold = {selected_threshold:.2f}"
    )

    plt.xlabel("Decision threshold")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{target_model_name}_decision_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


# ============================================================================
# Main analysis
# ============================================================================

def run_threshold_selection_and_dca(
    best_models,
    X_test,
    y_test,
    target_model_name=TARGET_MODEL_NAME,
    selected_threshold=SELECTED_THRESHOLD,
    output_dir=OUTPUT_DIR
):
    """
    Run threshold-dependent performance evaluation and decision curve analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Threshold selection and decision curve analysis")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Threshold-dependent classification metrics
    # ------------------------------------------------------------------------
    threshold_metrics = evaluate_all_models_across_thresholds(
        best_models=best_models,
        X_test=X_test,
        y_test=y_test,
        thresholds=METRIC_THRESHOLDS
    )

    metrics_path = os.path.join(output_dir, "threshold_metrics_all_models.csv")
    threshold_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"Threshold metrics saved to: {metrics_path}")

    # Selected threshold performance for the target model
    target_metrics = threshold_metrics[
        (threshold_metrics["Model"].str.lower() == target_model_name.lower()) &
        (np.isclose(threshold_metrics["Threshold"], selected_threshold))
    ]

    if target_metrics.empty:
        print(
            f"Selected threshold {selected_threshold:.2f} not found in metric thresholds. "
            "Calculating directly."
        )
        target_model = best_models[target_model_name]
        y_prob_target = get_predicted_probability(target_model, X_test)
        selected_metrics = calculate_metrics_at_threshold(
            y_true=y_test,
            y_prob=y_prob_target,
            threshold=selected_threshold
        )
        selected_metrics["Model"] = target_model_name
        selected_metrics["AUC"] = roc_auc_score(y_test, y_prob_target)
        selected_metrics["PR_AUC"] = average_precision_score(y_test, y_prob_target)
        selected_metrics = pd.DataFrame([selected_metrics])
    else:
        selected_metrics = target_metrics.copy()

    selected_path = os.path.join(output_dir, f"{target_model_name}_selected_threshold_metrics.csv")
    selected_metrics.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print(f"Selected threshold metrics saved to: {selected_path}")
    print("\nSelected threshold performance:")
    print(selected_metrics.to_string(index=False, float_format="%.4f"))

    # Plot threshold metrics
    threshold_plot_path = plot_threshold_metrics(
        metrics_df=threshold_metrics,
        target_model_name=target_model_name,
        selected_threshold=selected_threshold,
        output_dir=output_dir
    )

    print(f"Threshold metric plot saved to: {threshold_plot_path}")

    # ------------------------------------------------------------------------
    # 2. Decision curve analysis
    # ------------------------------------------------------------------------
    dca_results = calculate_decision_curve_results(
        best_models=best_models,
        X_test=X_test,
        y_test=y_test,
        thresholds=DCA_THRESHOLDS
    )

    dca_df = pd.DataFrame({"Threshold": DCA_THRESHOLDS})
    for model_name, net_benefits in dca_results.items():
        dca_df[model_name] = net_benefits

    dca_path = os.path.join(output_dir, "decision_curve_net_benefit_all_models.csv")
    dca_df.to_csv(dca_path, index=False, encoding="utf-8-sig")

    print(f"DCA net-benefit values saved to: {dca_path}")

    # Plot DCA
    dca_plot_path = plot_decision_curve(
        dca_results=dca_results,
        thresholds=DCA_THRESHOLDS,
        target_model_name=target_model_name,
        selected_threshold=selected_threshold,
        output_dir=output_dir
    )

    print(f"DCA plot saved to: {dca_plot_path}")

    # Summarize net-benefit ranges for target model
    dca_range_summary = summarize_net_benefit_ranges(
        dca_results=dca_results,
        thresholds=DCA_THRESHOLDS,
        target_model_name=target_model_name,
        clinical_range=(0.10, 0.60)
    )

    dca_range_path = os.path.join(output_dir, f"{target_model_name}_dca_net_benefit_ranges.csv")
    dca_range_summary.to_csv(dca_range_path, index=False, encoding="utf-8-sig")

    print(f"DCA net-benefit range summary saved to: {dca_range_path}")
    print("\nDCA net-benefit range summary:")
    print(dca_range_summary.to_string(index=False))

    print("\nAnalysis complete.")

    return {
        "threshold_metrics": threshold_metrics,
        "selected_metrics": selected_metrics,
        "dca_results": dca_results,
        "dca_range_summary": dca_range_summary
    }


# ============================================================================
# Example usage
# ============================================================================
# Uncomment and run after model training:
#
# results = run_threshold_selection_and_dca(
#     best_models=best_models,
#     X_test=X_test_scaled,
#     y_test=y_test,
#     target_model_name="CatBoost",
#     selected_threshold=0.35,  # 0.35 for recurrence; 0.30 for 3-year survival
#     output_dir="outputs/threshold_selection_recurrence"
# )
