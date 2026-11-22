import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import PROCESSED_DATA_DIR, PROJ_ROOT
from mibici.modeling.predict import load_model
from mibici.modeling.train import get_feature_groups



CUTOFF_DATE = pd.Timestamp("2026-08-01")
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000
UNSEEN_EXPOSURE = 0.20

PERFORMANCE_ALERT_THRESHOLD_PCT = 10.0

OUTPUT_DIR = PROJ_ROOT / "reports" / "drift"


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate regression performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = (
        PROCESSED_DATA_DIR / "mibici_features.parquet"
    )

    df = pd.read_parquet(features_path)
    df["fecha"] = pd.to_datetime(df["fecha"])

    train = df[df["fecha"] < CUTOFF_DATE].copy()
    post = df[df["fecha"] >= CUTOFF_DATE].copy()

    train_stations = set(
        train["origen_id"].unique()
    )

    post_stations = set(
        post["origen_id"].unique()
    )

    unseen_stations = sorted(
        post_stations - train_stations
    )

    print(
        f"Unseen stations after cutoff: "
        f"{len(unseen_stations)}"
    )

    if not unseen_stations:
        raise ValueError(
            "No unseen stations were found after the cutoff."
        )

    # Use only stations already known by the champion
    # as the reference population.
    known_post = post[
        post["origen_id"].isin(train_stations)
    ].copy()

    reference = known_post.sample(
        n=min(SAMPLE_SIZE, len(known_post)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    current = reference.copy()

    rng = np.random.default_rng(RANDOM_STATE)

    n_replace = int(
        len(current) * UNSEEN_EXPOSURE
    )

    replace_indices = rng.choice(
        len(current),
        size=n_replace,
        replace=False,
    )

    replacement_ids = rng.choice(
        unseen_stations,
        size=n_replace,
        replace=True,
    )

    # Synthetic network-expansion scenario:
    # selected observations are reassigned to stations
    # unseen during model training.
    current.loc[
        replace_indices,
        "origen_id",
    ] = replacement_ids

    # Cold-start condition:
    # new stations do not yet have sufficient
    # historical demand information.
    current.loc[
        replace_indices,
        "demanda_lag_1h",
    ] = 0.0

    current.loc[
        replace_indices,
        "demanda_lag_24h",
    ] = 0.0

    _, _, _, feature_cols = get_feature_groups()

    model = load_model()

    reference_predictions = model.predict(
        reference[feature_cols]
    )

    current_predictions = model.predict(
        current[feature_cols]
    )

    baseline_metrics = calculate_metrics(
        reference["demanda"],
        reference_predictions,
    )

    drift_metrics = calculate_metrics(
        current["demanda"],
        current_predictions,
    )

    rmse_degradation_pct = (
        (
            drift_metrics["rmse"]
            - baseline_metrics["rmse"]
        )
        / baseline_metrics["rmse"]
        * 100
    )

    performance_alert = (
        rmse_degradation_pct
        >= PERFORMANCE_ALERT_THRESHOLD_PCT
    )

    metrics = pd.DataFrame(
        [
            {
                "scenario": "baseline",
                **baseline_metrics,
                "rmse_degradation_pct": 0.0,
            },
            {
                "scenario": (
                    "network_expansion_full_cold_start"
                ),
                **drift_metrics,
                "rmse_degradation_pct":
                    rmse_degradation_pct,
            },
        ]
    )

    metrics_path = (
        OUTPUT_DIR / "drift_performance_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print("\n--- Performance comparison ---")
    print(metrics.to_string(index=False))

    print(
        "\nPerformance alert:",
        performance_alert,
    )

    print(
        "Alert threshold:",
        f"{PERFORMANCE_ALERT_THRESHOLD_PCT:.0f}%",
    )

    plot_path = OUTPUT_DIR / "rmse_comparison.png"

    labels = [
        "Baseline",
        "Expansion +\nCold Start",
    ]

    values = [
        baseline_metrics["rmse"],
        drift_metrics["rmse"],
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        labels,
        values,
    )

    plt.ylabel("RMSE")

    plt.title(
        "Champion Performance Under Synthetic Network Expansion"
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.text(
        0.5,
        max(values) + 0.12,
        f"RMSE degradation: +{rmse_degradation_pct:.2f}%",
        ha="center",
        va="bottom",
    )

    plt.ylim(
        0,
        max(values) + 0.30,
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    # Explicit mapping is important because several
    # integer columns are categorical by meaning.
    data_definition = DataDefinition(
        categorical_columns=[
            "origen_id",
            "hora",
            "dia_semana",
            "mes",
            "es_fin_de_semana",
        ],
        numerical_columns=[
            "demanda_lag_1h",
            "demanda_lag_24h",
        ],
    )

    reference_dataset = Dataset.from_pandas(
        reference[feature_cols],
        data_definition=data_definition,
    )

    current_dataset = Dataset.from_pandas(
        current[feature_cols],
        data_definition=data_definition,
    )

    report = Report(
        [
            DataDriftPreset(),
        ],
        include_tests=True,
    )

    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    html_path = OUTPUT_DIR / "data_drift_report.html"
    json_path = OUTPUT_DIR / "data_drift_report.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    print("\n--- Evidently outputs ---")
    print("HTML:", html_path)
    print("JSON:", json_path)

    print("\n--- Scenario ---")
    print(
        f"Synthetic unseen-station exposure: "
        f"{UNSEEN_EXPOSURE:.0%}"
    )
    print(
        "Cold start: demanda_lag_1h = 0 "
        "and demanda_lag_24h = 0"
    )


if __name__ == "__main__":
    main()