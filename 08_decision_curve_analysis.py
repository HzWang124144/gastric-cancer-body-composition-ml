

def calculate_net_benefit(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float,
) -> float:
    """Calculate model net benefit at one threshold probability."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if threshold <= 0 or threshold >= 1:
        return np.nan

    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)

    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def treat_all_net_benefit(
    y_true: Iterable[int],
    thresholds: Iterable[float],
) -> np.ndarray:
    """Calculate treat-all net benefit across thresholds."""
    y_true = np.asarray(y_true).astype(int)
    prevalence = y_true.mean()
    thresholds = np.asarray(thresholds, dtype=float)
    return prevalence - (1 - prevalence) * thresholds / (1 - thresholds)


def calculate_dca_curves(
    all_results: dict[str, dict],
    thresholds: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calculate net benefit curves for all models."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for model_name, result in all_results.items():
        y_true = result["y_true"]
        y_prob = result["y_prob"]
        for threshold in thresholds:
            rows.append({
                "Model": model_name,
                "Threshold": threshold,
                "Net Benefit": calculate_net_benefit(y_true, y_prob, threshold),
            })

    return pd.DataFrame(rows), thresholds


def mask_to_threshold_ranges(thresholds: Iterable[float], mask: Iterable[bool]) -> str:
    """Convert a Boolean mask to compact threshold ranges."""
    thresholds = np.asarray(thresholds, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return "None"

    ranges = []
    start = idx[0]
    prev = idx[0]

    for current in idx[1:]:
        if current == prev + 1:
            prev = current
        else:
            ranges.append((thresholds[start], thresholds[prev]))
            start = current
            prev = current

    ranges.append((thresholds[start], thresholds[prev]))

    parts = []
    for low, high in ranges:
        if np.isclose(low, high):
            parts.append(f"{low:.2f}")
        else:
            parts.append(f"{low:.2f}-{high:.2f}")

    return "; ".join(parts)


def summarize_dca_ranges(
    all_results: dict[str, dict],
    thresholds: Optional[np.ndarray] = None,
    clinical_range: tuple[float, float] = (0.10, 0.60),
    target_model_keyword: str = "CatBoost",
) -> pd.DataFrame:
    """
    Summarize threshold ranges where a target model has higher net benefit than
    both treat-all and treat-none strategies.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []

    for model_name, result in all_results.items():
        if target_model_keyword.lower() not in str(model_name).lower():
            continue

        y_true = np.asarray(result["y_true"]).astype(int)
        y_prob = np.asarray(result["y_prob"], dtype=float)

        model_nb = np.array([calculate_net_benefit(y_true, y_prob, th) for th in thresholds])
        all_nb = treat_all_net_benefit(y_true, thresholds)
        none_nb = np.zeros_like(thresholds)

        better_both = (model_nb > all_nb) & (model_nb > none_nb)
        clinical_mask = (thresholds >= clinical_range[0]) & (thresholds <= clinical_range[1])

        rows.append({
            "Model": model_name,
            "Range_GT_Both_AllThresholds": mask_to_threshold_ranges(thresholds, better_both),
            "Range_GT_Both_Clinical": mask_to_threshold_ranges(thresholds, better_both & clinical_mask),
            "Clinical_Range_Lower": clinical_range[0],
            "Clinical_Range_Upper": clinical_range[1],
        })

    return pd.DataFrame(rows)


def plot_dca(
    all_results: dict[str, dict],
    y_true: Iterable[int],
    thresholds: Optional[np.ndarray] = None,
    title: str = "Decision curve analysis",
    output_path: Optional[Path] = None,
) -> None:
    """Plot DCA curves for all models with treat-all and treat-none references."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = np.asarray(y_true).astype(int)
    treat_all = treat_all_net_benefit(y_true, thresholds)
    treat_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 7), facecolor="white")
    plt.plot(thresholds, treat_all, linestyle="--", color="black", label="Treat all")
    plt.plot(thresholds, treat_none, linestyle=":", color="black", label="Treat none")

    for model_name, result in all_results.items():
        model_nb = np.array(
            [calculate_net_benefit(result["y_true"], result["y_prob"], th) for th in thresholds]
        )
        plt.plot(thresholds, model_nb, linewidth=2, label=model_name)

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dca_outputs(
    all_results: dict[str, dict],
    y_true: Iterable[int],
    output_dir: Path,
    prefix: str,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save DCA curve data, DCA summary, and DCA plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dca_data, thresholds = calculate_dca_curves(all_results, thresholds=thresholds)
    dca_data.to_csv(output_dir / f"{prefix}_dca_curve_data.csv", index=False)

    dca_ranges = summarize_dca_ranges(all_results, thresholds=thresholds)
    dca_ranges.to_csv(output_dir / f"{prefix}_dca_net_benefit_ranges.csv", index=False)

    plot_dca(
        all_results,
        y_true=y_true,
        thresholds=thresholds,
        title=f"{prefix}: Decision curve analysis",
        output_path=output_dir / f"{prefix}_dca_curves.png",
    )

    return dca_data, dca_ranges
