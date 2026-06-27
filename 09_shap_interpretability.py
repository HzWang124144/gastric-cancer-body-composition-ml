"""
SHAP interpretability analysis for the final CatBoost model.

This script computes SHAP values for the final trained CatBoost model and
generates SHAP summary, feature-importance, heatmap, and force plots.

The script assumes that:
1. A trained CatBoost model has been saved as a .pkl or .joblib file.
2. The feature matrix used for SHAP interpretation is provided as a CSV file.
3. The CSV file contains only model input features and does not contain patient identifiers.

Example usage:
    python 12_shap_interpretability.py \
        --model_path models/final_catboost_recurrence.pkl \
        --x_path data_templates/X_test_scaled_template.csv \
        --output_dir outputs/shap_recurrence \
        --max_display 20

Notes:
    - Individual-level patient data are not included in this repository.
    - This script is provided to document the analytical workflow.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save SHAP plots for the final CatBoost model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved trained CatBoost model (.pkl or .joblib).",
    )

    parser.add_argument(
        "--x_path",
        type=str,
        required=True,
        help="Path to the feature matrix CSV used for SHAP interpretation.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/shap",
        help="Directory where SHAP outputs will be saved.",
    )

    parser.add_argument(
        "--max_display",
        type=int,
        default=20,
        help="Maximum number of features displayed in SHAP plots.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for selecting samples if subsampling is used.",
    )

    parser.add_argument(
        "--n_explain",
        type=int,
        default=100,
        help="Maximum number of samples used for SHAP explanation.",
    )

    return parser.parse_args()


def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model


def load_feature_matrix(x_path):
    x_path = Path(x_path)

    if not x_path.exists():
        raise FileNotFoundError(f"Feature matrix file not found: {x_path}")

    x = pd.read_csv(x_path)

    if x.empty:
        raise ValueError("The input feature matrix is empty.")

    return x


def compute_shap_values(model, x_explain):
    """
    Compute SHAP values using TreeExplainer.

    For binary classification, SHAP may return:
    - a list of two arrays, one for each class; or
    - a single array depending on the SHAP and model versions.

    This function returns SHAP values for the positive class when applicable.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_explain)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1, :, :]
        else:
            raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")

    return explainer, shap_values


def get_expected_value(explainer):
    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    expected_value = np.asarray(expected_value)

    if expected_value.ndim > 0:
        expected_value = expected_value.flatten()[0]

    return expected_value


def save_shap_summary_plot(shap_values, x_explain, output_dir, max_display):
    plt.figure(figsize=(10, 8), facecolor="white")
    shap.summary_plot(
        shap_values,
        x_explain,
        feature_names=x_explain.columns.tolist(),
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    ax = plt.gca()
    ax.grid(False)
    plt.title("SHAP summary plot", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "shap_summary_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def save_shap_bar_plot(shap_values, feature_names, output_dir, top_n=15):
    shap_importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsSHAP": shap_importance,
        }
    ).sort_values("MeanAbsSHAP", ascending=False)

    top_features = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.barh(
        range(len(top_features)),
        top_features["MeanAbsSHAP"].values,
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"].values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP feature importance", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(False)
    plt.tight_layout()

    output_path = output_dir / "shap_feature_importance_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    table_path = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    return output_path, table_path


def save_shap_heatmap(shap_values, x_explain, output_dir, max_display):
    """
    Save SHAP heatmap. Uses shap.plots.heatmap when supported.
    Falls back to a manual heatmap if the installed SHAP version does not
    support the requested object format.
    """
    feature_names = x_explain.columns.tolist()

    try:
        explanation = shap.Explanation(
            values=shap_values,
            data=x_explain.values,
            feature_names=feature_names,
        )

        plt.figure(figsize=(9, 7), facecolor="white")
        shap.plots.heatmap(explanation, max_display=max_display, show=False)
        plt.title("SHAP heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = output_dir / "shap_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception:
        shap_importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(shap_importance)[::-1][:max_display]
        heatmap_data = shap_values[:, sorted_idx]
        heatmap_features = [feature_names[i] for i in sorted_idx]

        plt.figure(figsize=(9, 7), facecolor="white")
        vmax = np.abs(heatmap_data).max()
        im = plt.imshow(
            heatmap_data.T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.yticks(range(len(heatmap_features)), heatmap_features, fontsize=9)
        plt.xticks([])
        plt.xlabel("Samples")
        plt.ylabel("Features")
        plt.title("SHAP heatmap", fontsize=14, fontweight="bold")
        plt.colorbar(im, label="SHAP value", shrink=0.8)
        plt.tight_layout()

        output_path = output_dir / "shap_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return output_path


def save_shap_force_plot(model, explainer, shap_values, x_explain, output_dir):
    """
    Save a force plot for one representative sample.

    The representative sample is selected as the sample with predicted
    probability closest to 0.5, corresponding to a boundary-case prediction.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_prob = model.predict_proba(x_explain)[:, 1]
    target_idx = int(np.argmin(np.abs(y_prob - 0.5)))

    expected_value = get_expected_value(explainer)

    plt.figure(figsize=(16, 5), facecolor="white")
    shap.force_plot(
        expected_value,
        shap_values[target_idx, :],
        x_explain.iloc[target_idx, :],
        matplotlib=True,
        show=False,
    )
    plt.title(
        f"SHAP force plot | Predicted probability = {y_prob[target_idx]:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "shap_force_plot_representative_sample.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.random_state)

    model = load_model(args.model_path)
    x = load_feature_matrix(args.x_path)

    n_explain = min(args.n_explain, len(x))
    if n_explain < len(x):
        selected_idx = np.random.choice(len(x), n_explain, replace=False)
        x_explain = x.iloc[selected_idx, :].copy()
    else:
        x_explain = x.copy()

    feature_names = x_explain.columns.tolist()

    explainer, shap_values = compute_shap_values(model, x_explain)

    summary_path = save_shap_summary_plot(
        shap_values=shap_values,
        x_explain=x_explain,
        output_dir=output_dir,
        max_display=args.max_display,
    )

    bar_path, table_path = save_shap_bar_plot(
        shap_values=shap_values,
        feature_names=feature_names,
        output_dir=output_dir,
        top_n=min(15, len(feature_names)),
    )

    heatmap_path = save_shap_heatmap(
        shap_values=shap_values,
        x_explain=x_explain,
        output_dir=output_dir,
        max_display=args.max_display,
    )

    force_path = save_shap_force_plot(
        model=model,
        explainer=explainer,
        shap_values=shap_values,
        x_explain=x_explain,
        output_dir=output_dir,
    )

    print("SHAP analysis completed.")
    print(f"Summary plot saved to: {summary_path}")
    print(f"Feature-importance bar plot saved to: {bar_path}")
    print(f"Feature-importance table saved to: {table_path}")
    print(f"Heatmap saved to: {heatmap_path}")

    if force_path is not None:
        print(f"Force plot saved to: {force_path}")
    else:
        print("Force plot was not generated because the model lacks predict_proba().")


if __name__ == "__main__":
    main()
