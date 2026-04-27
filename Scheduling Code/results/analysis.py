#%%
# analysis.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 0.) Paths and plotting setup

results_dir = Path(__file__).resolve().parent
fig_dir = results_dir / "figures"
fig_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
plt.rcParams["font.family"] = "serif"

# 1.) Helper functions

def nice_strategy_name(s):
    name_map = {
        "dedicated_printer_baseline": "Dedicated printer baseline",
        "two_phase_greedy": "Two-phase greedy",
        "satiation_aware": "Satiation-aware",
        "location_aware": "Location-aware",
    }
    return name_map.get(s, s.replace("_", " ").title())


def load_prefixed_csvs(prefix):
    files = sorted(results_dir.glob(f"{prefix}_*.csv"))
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def save_figure(fig, filename):
    fig.savefig(fig_dir / filename, bbox_inches="tight")
    plt.close(fig)

# 2.) Load datasets

metrics_path = results_dir / "metrics_all_runs.csv"

if not metrics_path.exists():
    raise FileNotFoundError("metrics_all_runs.csv not found in results folder")

metrics = pd.read_csv(metrics_path)
drones = load_prefixed_csvs("drones")

metrics["strategy_label"] = metrics["strategy"].apply(nice_strategy_name)

if not drones.empty:
    drones["strategy_label"] = drones["strategy"].apply(nice_strategy_name)

# 3.) Common definitions

strategy_order = [
    "Dedicated printer baseline",
    "Two-phase greedy",
    "Location-aware",
    "Satiation-aware",
]

strategy_colors = {
    "Dedicated printer baseline": "#2E2585",
    "Two-phase greedy": "#9F4A9C",
    "Location-aware": "#B2C9AB",
    "Satiation-aware": "#7E2954",
}

run_titles = {
    "run_001": "Run 001: 200 drones, 5 mesh",
    "run_002": "Run 002: 1000 drones, 25 mesh",
    "run_003": "Run 003: 10000 drones, 250 mesh",
    "run_004": "Run 004: 200 drones, 5 mesh",
    "run_005": "Run 005: 1000 drones, 25 mesh",
    "run_006": "Run 006: 10000 drones, 250 mesh",
}

main_runs = ["run_001", "run_002", "run_003"]
split_runs = ["run_004", "run_005", "run_006"]


# Figure 1 - Completion curves for Runs 001-003

if not drones.empty:
    fig = plt.figure(figsize=(12.5, 8.6))

    # [left, bottom, width, height]
    ax1 = fig.add_axes([0.08, 0.57, 0.38, 0.28])   # top left
    ax2 = fig.add_axes([0.54, 0.57, 0.38, 0.28])   # top right
    ax3 = fig.add_axes([0.31, 0.16, 0.38, 0.28])   # bottom centred

    axes_map = {
        "run_001": ax1,
        "run_002": ax2,
        "run_003": ax3,
    }

    legend_handles = []

    for run_id in main_runs:
        ax = axes_map[run_id]

        d_run = drones[drones["run_id"] == run_id].copy()
        m_run = metrics[metrics["run_id"] == run_id].copy()

        if d_run.empty or m_run.empty:
            ax.set_visible(False)
            continue

        mesh_value = int(m_run["mesh"].iloc[0])
        n_drones_total = int(m_run["n_drones"].iloc[0])

        ax.axhline(
            mesh_value,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.35
        )

        xmax = 0

        for label in strategy_order:
            d = d_run[d_run["strategy_label"] == label].copy()
            if d.empty:
                continue

            d = d.sort_values("drone_finish_time").reset_index(drop=True)
            x = d["drone_finish_time"].to_numpy()
            y = np.arange(1, len(d) + 1)

            color = strategy_colors[label]

            ax.step(
                x,
                y,
                where="post",
                linewidth=2.2,
                color=color
            )

            xmax = max(xmax, x.max())

            if run_id == "run_001":
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=2.2, label=label)
                )

        ax.set_title(run_titles[run_id], pad=10)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Cumulative drones completed")
        ax.set_ylim(0, n_drones_total)
        ax.set_xlim(0, xmax * 1.02)
        ax.grid(True, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        handles=legend_handles,
        title="Strategy",
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.97)
    )

    save_figure(fig, "fig01_completion_curves_runs_001_003.png")

# Figure 2 - Summary bars for T_mesh and makespan

plot_df = metrics[metrics["run_id"].isin(main_runs)].copy()

plot_df["strategy_label"] = pd.Categorical(
    plot_df["strategy_label"],
    categories=strategy_order,
    ordered=True
)
plot_df = plot_df.sort_values(["run_id", "strategy_label"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

metric_info = [
    ("T_mesh", r"$T_{\mathrm{mesh}}$ (min)"),
    ("makespan", "Makespan (min)")
]

run_positions = np.arange(len(main_runs))
bar_width = 0.18

for ax, (metric_col, ylabel) in zip(axes, metric_info):
    for i, label in enumerate(strategy_order):
        subset = plot_df[plot_df["strategy_label"] == label].copy()
        subset = subset.set_index("run_id").reindex(main_runs)

        values = subset[metric_col].to_numpy()
        xpos = run_positions + (i - 1.5) * bar_width

        bars = ax.bar(
            xpos,
            values,
            width=bar_width,
            color=strategy_colors[label],
            label=label
        )

        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin if ymax > ymin else 1.0

        for bar, val in zip(bars, values):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015 * yrange,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    ax.set_xticks(run_positions)
    ax.set_xticklabels([run_titles[r] for r in main_runs], rotation=12)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_title(r"Mesh completion time, $T_{\mathrm{mesh}}$")
axes[1].set_title("Final makespan")

handles = [
    Line2D([0], [0], color=strategy_colors[label], linewidth=8, label=label)
    for label in strategy_order
]
fig.legend(
    handles=handles,
    title="Strategy",
    loc="upper center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, 1.02)
)
save_figure(fig, "fig02_summary_bars_runs_001_003.png")



# Figure 3 - Effect of part splitting on T_mesh
# compare original (runs 001-003) vs split (runs 004-006)
# satiation-aware only

orig_runs = ["run_001", "run_002", "run_003"]
split_runs = ["run_004", "run_005", "run_006"]

orig_df = metrics[
    (metrics["run_id"].isin(orig_runs)) &
    (metrics["strategy"] == "satiation_aware")
].copy()

split_df = metrics[
    (metrics["run_id"].isin(split_runs)) &
    (metrics["strategy"] == "satiation_aware")
].copy()

if not orig_df.empty and not split_df.empty:
    scale_labels = [
        "200 drones,\n5 mesh",
        "1000 drones,\n25 mesh",
        "10000 drones,\n250 mesh",
    ]

    orig_df = orig_df.set_index("run_id").reindex(orig_runs).reset_index()
    split_df = split_df.set_index("run_id").reindex(split_runs).reset_index()

    orig_tmesh = orig_df["T_mesh"].to_numpy()
    split_tmesh = split_df["T_mesh"].to_numpy()

    x = np.arange(len(scale_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    bars1 = ax.bar(
        x - width / 2,
        orig_tmesh,
        width=width,
        color="#7E2954",
        label="Original part set"
    )

    bars2 = ax.bar(
        x + width / 2,
        split_tmesh,
        width=width,
        color="#B2C9AB",
        label="Split part set"
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if pd.notna(h):
                ax.annotate(
                    f"{h:.1f}",
                    (bar.get_x() + bar.get_width() / 2, h),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 4),
                    textcoords="offset points"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_ylabel(r"$T_{\mathrm{mesh}}$ (min)")
    ax.grid(True, axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True)

    save_figure(fig, "fig03_part_splitting_comparison.png")

# 4.)Summary tables

summary_a = metrics[
    metrics["run_id"].isin(main_runs)
][[
    "run_id", "strategy_label", "n_drones", "mesh", "T_mesh", "makespan",
    "throughput_avg_per_hour", "throughput_post_mesh_per_hour"
]].copy()

summary_a = summary_a.sort_values(["run_id", "strategy_label"]).reset_index(drop=True)
summary_a.to_csv(results_dir / "summary_experiment_A.csv", index=False)

summary_b = metrics[
    (metrics["run_id"].isin(split_runs)) &
    (metrics["strategy"] == "satiation_aware")
][[
    "run_id", "strategy_label", "n_drones", "mesh", "T_mesh", "makespan",
    "throughput_avg_per_hour", "throughput_post_mesh_per_hour"
]].copy()

summary_b = summary_b.sort_values("run_id").reset_index(drop=True)
summary_b.to_csv(results_dir / "summary_experiment_B.csv", index=False)

print("\nDone.")
print(f"Figures saved to: {fig_dir.resolve()}")
print(f"Experiment A summary saved to: {(results_dir / 'summary_experiment_A.csv').resolve()}")
print(f"Experiment B summary saved to: {(results_dir / 'summary_experiment_B.csv').resolve()}")