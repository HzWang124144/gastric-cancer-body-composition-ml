
try:
    import statsmodels.api as sm
except ImportError:
    sm = None


def _logit_clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to logits after clipping."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expected_calibration_error(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    n_bins: int = 10,
) -> float:
    """Calculate weighted expected calibration error."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    n = len(y_true)

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(observed - predicted)

    return float(ece)


def calibration_intercept_slope(
    y_true: Iterable[int],
    y_prob: Iterable[float],
) -> tuple[float, float]:
    """
    Estimate calibration intercept and slope by logistic recalibration.

    Returns NaN values if statsmodels is unavailable or fitting fails.
    """
    if sm is None:
        return np.nan, np.nan

    y_true = np.asarray(y_true).astype(int)
    logits = _logit_clip(np.asarray(y_prob, dtype=float))

    try:
        X = sm.add_constant(logits)
        model = sm.Logit(y_true, X).fit(disp=False)
        intercept, slope = model.params[0], model.params[1]
        return float(intercept), float(slope)
    except Exception:
        return np.nan, np.nan


def calculate_calibration_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, Any]:
    """Calculate calibration curve and quantitative calibration metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )

    intercept, slope = calibration_intercept_slope(y_true, y_prob)

    return {
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "Brier Score": brier_score_loss(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "Calibration Intercept": intercept,
        "Calibration Slope": slope,
    }


def summarize_calibration(
    all_results: dict[str, Any],
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Summarize calibration metrics for multiple models."""
    rows = []

    for model_name, result in all_results.items():
        metrics = calculate_calibration_metrics(
            y_true=result["y_true"],
            y_prob=result["y_prob"],
            n_bins=n_bins,
            strategy=strategy,
        )
        rows.append({
            "Model": model_name,
            "Brier Score": metrics["Brier Score"],
            "ECE": metrics["ECE"],
            "Calibration Intercept": metrics["Calibration Intercept"],
            "Calibration Slope": metrics["Calibration Slope"],
        })

    return pd.DataFrame(rows).sort_values("Brier Score").reset_index(drop=True)


def plot_calibration_curve_single(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    model_name: str,
    n_bins: int = 10,
    strategy: str = "quantile",
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Plot a calibration curve for one model."""
    metrics = calculate_calibration_metrics(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    plt.figure(figsize=(7, 7), facecolor="white")
    plt.plot([0, 1], [0, 1], linestyle=":", color="black", label="Ideal calibration")
    plt.plot(
        metrics["mean_predicted_value"],
        metrics["fraction_of_positives"],
        marker="o",
        linewidth=2,
        label=model_name,
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event proportion")
    plt.title(f"Calibration curve: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return metrics


def save_calibration_outputs(
    all_results: dict[str, Any],
    output_dir: Path,
    prefix: str,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Save calibration metrics and calibration plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_calibration(all_results, n_bins=n_bins, strategy=strategy)
    summary.to_csv(output_dir / f"{prefix}_calibration_metrics.csv", index=False)

    for model_name, result in all_results.items():
        safe_name = str(model_name).replace(" ", "_").replace("/", "_")
        plot_calibration_curve_single(
            y_true=result["y_true"],
            y_prob=result["y_prob"],
            model_name=model_name,
            n_bins=n_bins,
            strategy=strategy,
            output_path=output_dir / f"{prefix}_{safe_name}_calibration_curve.png",
        )

    return summary
