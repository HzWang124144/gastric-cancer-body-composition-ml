
def get_predicted_probability(model: Any, X: Any) -> np.ndarray:
    """
    Return predicted probability for the positive class.

    If predict_proba is unavailable, decision_function scores are min-max
    scaled to [0, 1] for plotting and threshold-based evaluation.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        score_min, score_max = scores.min(), scores.max()
        if np.isclose(score_min, score_max):
            return np.full_like(scores, fill_value=0.5, dtype=float)
        return (scores - score_min) / (score_max - score_min)

    raise AttributeError("Model must implement predict_proba or decision_function.")


def specificity_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Calculate specificity for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_model_at_threshold(
    model: Any,
    model_name: str,
    X_test: Any,
    y_test: Iterable[int],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a trained model at a specified probability threshold."""
    y_true = np.asarray(y_test).astype(int)
    y_prob = get_predicted_probability(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

    metrics = {
        "Model": model_name,
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "PR_AUC": auc(recall_curve, precision_curve),
        "Threshold_Used": threshold,
    }

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_data": (fpr, tpr, metrics["AUC"]),
        "pr_data": (precision_curve, recall_curve, metrics["PR_AUC"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def evaluate_models(
    models: dict[str, Any],
    X_test: Any,
    y_test: Iterable[int],
    threshold: float,
    skip_models: Optional[list[str]] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate all trained models and return detailed results and a summary table."""
    skip_models = skip_models or []
    all_results: dict[str, Any] = {}

    for model_name, model in models.items():
        if model is None:
            continue
        if any(skip.lower() in str(model_name).lower() for skip in skip_models):
            continue

        all_results[model_name] = evaluate_model_at_threshold(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
        )

    summary = pd.DataFrame([res["metrics"] for res in all_results.values()])
    if not summary.empty:
        summary = summary.sort_values("AUC", ascending=False).reset_index(drop=True)

    return all_results, summary


def bootstrap_auc_ci(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    n_bootstrap: int = 1000,
    ci: float = 95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Calculate ROC-AUC and percentile bootstrap confidence interval."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    auc_original = roc_auc_score(y_true, y_prob)
    rng = np.random.default_rng(random_state)
    boot_values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        boot_values.append(roc_auc_score(y_true[indices], y_prob[indices]))

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_values, alpha)
    upper = np.percentile(boot_values, 100 - alpha)

    return auc_original, lower, upper


def add_auc_confidence_intervals(
    all_results: dict[str, Any],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a table of AUC values with bootstrap confidence intervals."""
    rows = []
    for model_name, result in all_results.items():
        auc_value, ci_lower, ci_upper = bootstrap_auc_ci(
            result["y_true"],
            result["y_prob"],
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        rows.append({
            "Model": model_name,
            "AUC": auc_value,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper,
        })

    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def plot_roc_curves(
    all_results: dict[str, Any],
    title: str = "ROC curves in the internal test set",
    output_path: Optional[Path] = None,
) -> None:
    """Plot ROC curves for all evaluated models."""
    plt.figure(figsize=(8, 7), facecolor="white")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="Chance")

    for model_name, result in all_results.items():
        fpr, tpr, roc_auc = result["roc_data"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    all_results: dict[str, Any],
    y_test: Iterable[int],
    title: str = "Precision-recall curves in the internal test set",
    output_path: Optional[Path] = None,
) -> None:
    """Plot precision-recall curves for all evaluated models."""
    y_true = np.asarray(y_test).astype(int)
    positive_ratio = y_true.mean()

    plt.figure(figsize=(8, 7), facecolor="white")
    plt.axhline(
        y=positive_ratio,
        linestyle="--",
        color="black",
        alpha=0.5,
        label=f"Baseline prevalence = {positive_ratio:.3f}",
    )

    for model_name, result in all_results.items():
        precision_curve, recall_curve, pr_auc = result["pr_data"]
        plt.plot(recall_curve, precision_curve, linewidth=2, label=f"{model_name} (AUC = {pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance_heatmap(
    summary: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    title: str = "Model performance heatmap",
    output_path: Optional[Path] = None,
) -> None:
    """Plot a simple heatmap of selected performance metrics."""
    if summary.empty:
        return

    metrics = metrics or ["AUC", "Accuracy", "F1", "Recall", "Precision", "Specificity"]
    data = summary.set_index("Model")[metrics].astype(float)

    plt.figure(figsize=(10, 6), facecolor="white")
    im = plt.imshow(data.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Score")

    plt.xticks(range(len(metrics)), metrics, rotation=45, ha="right")
    plt.yticks(range(len(data.index)), data.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            plt.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9)

    plt.title(title)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_evaluation_outputs(
    all_results: dict[str, Any],
    summary: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    y_test: Iterable[int],
) -> None:
    """Save performance tables and core plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / f"{prefix}_performance_summary.csv", index=False)

    auc_ci = add_auc_confidence_intervals(all_results)
    auc_ci.to_csv(output_dir / f"{prefix}_auc_bootstrap_ci.csv", index=False)

    plot_roc_curves(
        all_results,
        title=f"{prefix}: ROC curves",
        output_path=output_dir / f"{prefix}_roc_curves.png",
    )
    plot_pr_curves(
        all_results,
        y_test=y_test,
        title=f"{prefix}: Precision-recall curves",
        output_path=output_dir / f"{prefix}_pr_curves.png",
    )
    plot_performance_heatmap(
        summary,
        title=f"{prefix}: Performance heatmap",
        output_path=output_dir / f"{prefix}_performance_heatmap.png",
    )
