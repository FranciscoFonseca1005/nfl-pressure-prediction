from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
ANALYTICAL_TABLE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pass_rush_analytical_table.parquet"
)

OUTPUT_PATHS = [
    REPORTS_DIR / "report_weekly_pressure_rate.png",
    REPORTS_DIR / "report_model_average_precision.png",
    REPORTS_DIR / "report_test_confusion_matrix.png",
]

BLUE = "#3568A8"
DARK_BLUE = "#173F5F"
ORANGE = "#E47C3C"
GRAY = "#A7B0B8"
LIGHT_GRAY = "#D8DDE2"
GRID = "#D6D9DC"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#222222",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    }
)


def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def create_weekly_pressure_chart():
    data = pd.read_parquet(
        ANALYTICAL_TABLE,
        columns=["actual_week", "pressure"],
    )

    weekly = (
        data.groupby("actual_week", as_index=False)
        .agg(
            rows=("pressure", "size"),
            positives=("pressure", "sum"),
            pressure_rate=("pressure", "mean"),
        )
        .sort_values("actual_week")
    )

    assert weekly["actual_week"].tolist() == list(range(1, 9))
    assert int(weekly["rows"].sum()) == 36259
    assert int(weekly["positives"].sum()) == 4214

    fig, ax = plt.subplots(figsize=(9.2, 5.3))

    ax.plot(
        weekly["actual_week"],
        weekly["pressure_rate"],
        color=BLUE,
        linewidth=2.4,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
    )

    ax.scatter(
        [7],
        weekly.loc[
            weekly["actual_week"].eq(7),
            "pressure_rate",
        ],
        color=ORANGE,
        s=80,
        zorder=3,
        label="Validation week",
    )

    ax.scatter(
        [8],
        weekly.loc[
            weekly["actual_week"].eq(8),
            "pressure_rate",
        ],
        color=DARK_BLUE,
        s=80,
        zorder=3,
        label="Final test week",
    )

    for row in weekly.itertuples(index=False):
        ax.annotate(
            f"{row.pressure_rate:.1%}",
            (row.actual_week, row.pressure_rate),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#333333",
        )

    ax.axvline(
        6.5,
        color=GRAY,
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        7.5,
        color=GRAY,
        linestyle="--",
        linewidth=1,
    )

    ax.text(
        3.5,
        0.101,
        "Training",
        ha="center",
        color="#555555",
    )
    ax.text(
        7,
        0.101,
        "Validation",
        ha="center",
        color="#555555",
    )
    ax.text(
        8,
        0.101,
        "Test",
        ha="center",
        color="#555555",
    )

    ax.set_title(
        "Pressure prevalence remained stable across weeks\n"
        "Each row represents one pass rusher at the snap",
        pad=14,
    )
    ax.set_xlabel("Actual NFL week")
    ax.set_ylabel("Observed pressure rate")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0.10, 0.132)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        alpha=0.8,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper right",
    )

    fig.text(
        0.01,
        0.01,
        "Pressure means a PFF-recorded hurry, hit, or sack.",
        fontsize=8.5,
        color="#666666",
    )

    save_figure(
        fig,
        REPORTS_DIR / "report_weekly_pressure_rate.png",
    )


def create_model_comparison_chart():
    stage6 = pd.read_csv(
        REPORTS_DIR / "stage6_baseline_model_results.csv"
    )
    stage7 = pd.read_csv(
        REPORTS_DIR / "stage7_candidate_results.csv"
    )

    stage6_models = stage6[
        [
            "model",
            "average_precision",
        ]
    ].copy()

    advanced_models = stage7.loc[
        stage7["model"].isin(
            [
                "ExtraTrees",
                "HistGradientBoosting",
            ]
        ),
        [
            "model",
            "average_precision",
        ],
    ].copy()

    comparison = pd.concat(
        [stage6_models, advanced_models],
        ignore_index=True,
    ).drop_duplicates(
        subset=["model"],
        keep="last",
    )

    order = [
        "DummyClassifier_prior",
        "HistGradientBoosting",
        "LogisticRegression_unweighted",
        "ExtraTrees",
    ]

    labels = {
        "DummyClassifier_prior": "Dummy prior",
        "HistGradientBoosting": "Histogram gradient boosting",
        "LogisticRegression_unweighted": "Logistic regression",
        "ExtraTrees": "Extra Trees",
    }

    comparison = (
        comparison.set_index("model")
        .loc[order]
        .reset_index()
    )

    expected = {
        "DummyClassifier_prior": 0.110479,
        "HistGradientBoosting": 0.145960,
        "LogisticRegression_unweighted": 0.151899,
        "ExtraTrees": 0.153689,
    }

    for model, expected_value in expected.items():
        observed = float(
            comparison.loc[
                comparison["model"].eq(model),
                "average_precision",
            ].iloc[0]
        )
        assert abs(observed - expected_value) <= 5e-7

    colors = [
        GRAY,
        LIGHT_GRAY,
        DARK_BLUE,
        ORANGE,
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    positions = np.arange(len(comparison))
    bars = ax.barh(
        positions,
        comparison["average_precision"],
        color=colors,
        height=0.62,
    )

    ax.set_yticks(
        positions,
        [labels[model] for model in comparison["model"]],
    )
    ax.invert_yaxis()

    for bar, value in zip(
        bars,
        comparison["average_precision"],
    ):
        ax.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10,
            color="#333333",
        )

    dummy_ap = float(
        comparison.loc[
            comparison["model"].eq(
                "DummyClassifier_prior"
            ),
            "average_precision",
        ].iloc[0]
    )

    ax.axvline(
        dummy_ap,
        color="#777777",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Advanced models did not materially improve ranking\n"
        "Week-7 validation average precision",
        pad=14,
    )
    ax.set_xlabel("Average precision")
    ax.set_xlim(0, 0.175)
    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.8,
        alpha=0.8,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.text(
        0.01,
        0.01,
        "Extra Trees gained only 0.00179 AP over logistic regression; "
        "its bootstrap 95% interval included zero. "
        "Logistic regression was therefore retained.",
        fontsize=8.5,
        color="#666666",
    )

    save_figure(
        fig,
        REPORTS_DIR / "report_model_average_precision.png",
    )


def create_test_confusion_matrix_chart():
    confusion = pd.read_csv(
        REPORTS_DIR / "stage9_confusion_matrix.csv"
    ).sort_values("actual_class")

    matrix = confusion[
        [
            "predicted_0",
            "predicted_1",
        ]
    ].to_numpy(dtype=int)

    expected_matrix = np.array(
        [
            [2670, 1270],
            [264, 240],
        ]
    )

    assert np.array_equal(matrix, expected_matrix)

    fig, ax = plt.subplots(figsize=(7.3, 5.8))

    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=matrix.max(),
    )

    row_totals = matrix.sum(axis=1)

    for row_index in range(2):
        for column_index in range(2):
            count = matrix[row_index, column_index]
            row_rate = count / row_totals[row_index]

            text_color = (
                "white"
                if count > matrix.max() * 0.48
                else "#222222"
            )

            ax.text(
                column_index,
                row_index,
                f"{count:,}\n{row_rate:.1%} of actual class",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(
        [0, 1],
        [
            "Predicted no pressure",
            "Predicted pressure",
        ],
    )
    ax.set_yticks(
        [0, 1],
        [
            "Actual no pressure",
            "Actual pressure",
        ],
    )

    ax.set_xlabel("Model decision")
    ax.set_ylabel("Observed outcome")
    ax.set_title(
        "Final week-8 decisions at the locked threshold\n"
        "240 of 504 pressures detected; 1,270 false alerts",
        pad=16,
    )

    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Rusher-play decisions")

    fig.text(
        0.01,
        0.01,
        "Threshold = 0.124855. No model or threshold changes "
        "were made after accessing week 8.",
        fontsize=8.5,
        color="#666666",
    )

    save_figure(
        fig,
        REPORTS_DIR / "report_test_confusion_matrix.png",
    )


def validate_outputs():
    records = []

    for path in OUTPUT_PATHS:
        assert path.exists(), f"Missing output: {path}"
        size_bytes = path.stat().st_size
        assert size_bytes > 10000, (
            f"Output is unexpectedly small: {path}"
        )

        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        records.append(
            {
                "artifact": path.name,
                "size_bytes": size_bytes,
                "sha256": digest,
                "status": "PASS",
            }
        )

    return pd.DataFrame(records)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    create_weekly_pressure_chart()
    create_model_comparison_chart()
    create_test_confusion_matrix_chart()

    validation = validate_outputs()

    print(validation.to_string(index=False))
    print()
    print("STAGE 10 REPORT VISUALIZATIONS: PASS")


if __name__ == "__main__":
    main()