#%%
#0.) - Imports:
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies import strategy_map

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

#%%
#1.) - Input Parameters, Printers List, Parts List, Job List:
#1.1.) - Input Parameters
# STRATEGY DICTIONARY:
# satiation_aware
# two_phase_greedy
# location_aware
# dedicated_printer_baseline

strategy = "satiation_aware"
run_id = "run_001"
n_drones = 200
mesh = 5
glut = 1.1

#1.2) - Printers List
printers = pd.read_excel(DATA_DIR / "printers.xlsx")
total_printers = len(printers)

# data cleaning
number_columns = ["Bed_X_mm", "Bed_Y_mm", "Bed_Z_mm", "Print_Time"]
printers[number_columns] = printers[number_columns].apply(pd.to_numeric, errors="coerce")

# speed factor and loads set to 0
t_reference = printers["Print_Time"].min()
printers["speed_factor"] = printers["Print_Time"] / t_reference
printers["load"] = 0.0

# rename columns
printers = printers.rename(columns={
    "Printer_ID": "id",
    "Model": "model",
    "Location": "location",
    "Bed_X_mm": "bed_x",
    "Bed_Y_mm": "bed_y",
    "Bed_Z_mm": "bed_z",
    "Print_Time": "time_metric"
})

#1.3) - Parts List:
parts = pd.DataFrame({
    "part": ["arm", "base_plate", "top_plate"],
    "qty":  [4, 1, 1],
    "t":    [28, 53, 44]
})

#1.4) - Drone Job Generation:
drone_total_workload = (parts["qty"] * parts["t"]).sum()

jobs = (
    parts.loc[parts.index.repeat(parts["qty"])]
    .drop(columns="qty")
    .reset_index(drop=True)
)

jobs = pd.concat([jobs.assign(drone_id=d) for d in range(n_drones)], ignore_index=True)
jobs = jobs.sort_values(["drone_id", "t"], ascending=[True, False]).reset_index(drop=True)

jobs["location"] = None
jobs["printer_id"] = None
jobs["t_speed"] = np.nan
jobs["finish_time"] = np.nan

drone_finish = np.zeros(n_drones)
drone_location = np.empty(n_drones, dtype=object)

#1.5) - Locations:
location_names = printers["location"].dropna().unique()
location_work = pd.Series(0.0, index=location_names)

#1.6) - WPL and Phi Pandas Series:
WPL_location = pd.Series(index=location_names, dtype=float)
phi_location = pd.Series(index=location_names, dtype=float)

#%%
#2.) - Strategy Call:
jobs, printers, drone_finish, drone_location, location_work = strategy_map[strategy](
    printers=printers,
    jobs=jobs,
    n_drones=n_drones,
    mesh=mesh,
    glut=glut,
    total_printers=total_printers,
    drone_total_workload=drone_total_workload,
    drone_finish=drone_finish,
    drone_location=drone_location,
    location_names=location_names,
    location_work=location_work,
    WPL_location=WPL_location,
    phi_location=phi_location,
)

#%%
#3.) - Metrics
def compute_metrics(printers, drone_finish, mesh, location_names, location_work, drone_location=None):
    WPL_location = pd.Series(index=location_names, dtype=float)

    for loc in location_names:
        n_printers_location = (printers["location"] == loc).sum()
        WPL_location.loc[loc] = location_work.loc[loc] / n_printers_location

    phi_location = pd.Series(index=location_names, dtype=float)
    mean_load = printers["load"].mean()

    if mean_load > 0:
        for loc in location_names:
            phi_location.loc[loc] = WPL_location.loc[loc] / mean_load
    else:
        phi_location[:] = np.nan

    phi_summary = (
        pd.DataFrame({
            "location": location_names,
            "WPL": WPL_location.values,
            "satiation_index": phi_location.values
        })
        .sort_values("satiation_index")
        .reset_index(drop=True)
    )

    sorted_finish = np.sort(np.array(drone_finish, dtype=float))
    n_total = len(sorted_finish)

    T_mesh = sorted_finish[mesh - 1] if n_total >= mesh else np.nan
    makespan = printers["load"].max()

    throughput_avg_per_hour = 60 * n_total / makespan if makespan > 0 else np.nan

    if pd.notna(T_mesh) and makespan > T_mesh and n_total > mesh:
        throughput_post_mesh_per_hour = 60 * (n_total - mesh) / (makespan - T_mesh)
    else:
        throughput_post_mesh_per_hour = np.nan

    load_min = printers["load"].min()
    load_mean = printers["load"].mean()
    load_max = printers["load"].max()
    load_std = printers["load"].std()

    phi_mean = phi_location.mean()
    phi_std = phi_location.std()
    phi_min = phi_location.min()
    phi_max = phi_location.max()

    metrics = {
        "makespan": makespan,
        "T_mesh": T_mesh,
        "throughput_avg_per_hour": throughput_avg_per_hour,
        "throughput_post_mesh_per_hour": throughput_post_mesh_per_hour,
        "load_min": load_min,
        "load_mean": load_mean,
        "load_max": load_max,
        "load_std": load_std,
        "phi_min": phi_min,
        "phi_mean": phi_mean,
        "phi_max": phi_max,
        "phi_std": phi_std,
        "n_drones_total": n_total,
    }

    if drone_location is not None:
        drone_counts = pd.Series(drone_location).value_counts(dropna=False)
    else:
        drone_counts = None

    return metrics, phi_summary, phi_location, WPL_location, sorted_finish, drone_counts


#%%
#4.) - Results
results = []

metrics, phi_summary, phi_location, WPL_location, sorted_finish, drone_counts = compute_metrics(
    printers=printers,
    drone_finish=drone_finish,
    mesh=mesh,
    location_names=location_names,
    location_work=location_work,
    drone_location=drone_location,
)

results.append({
    "method": strategy,
    **metrics
})

print("\nSatiation index (φ) by location:")
print(phi_summary.to_string(index=False))

print("\nScalar metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nDrones per location:")
print(drone_counts)

results_df = pd.DataFrame(results)
print("\nComparison table:")
print(results_df.to_string(index=False))


#%%
#5.) - Drone completion curve
sorted_finish = np.sort(np.array(sorted_finish, dtype=float))
drones_completed = np.arange(1, len(sorted_finish) + 1)

plt.figure(figsize=(10, 6))
plt.step(sorted_finish, drones_completed, where="post", linewidth=2, label="Drones completed")

T_mesh = sorted_finish[mesh - 1]
plt.axhline(mesh, linestyle="--", linewidth=1, label=f"Mesh threshold = {mesh}")
plt.axvline(T_mesh, linestyle="--", linewidth=1, label=f"T_mesh = {T_mesh:.1f} min")

plt.xlabel("Time (min)")
plt.ylabel("Cumulative drones completed")
plt.title(f"Drone completion curve - {strategy}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
#6.) - Export
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
output_dir = RESULTS_DIR
file_tag = f"{strategy}_{run_id}"

# jobs
jobs_export = jobs.copy()
jobs_export["strategy"] = strategy
jobs_export["run_id"] = run_id
jobs_export["n_drones"] = n_drones
jobs_export["mesh"] = mesh
jobs_export["glut"] = glut

jobs_export = jobs_export.rename(columns={"t": "t_nominal"})
jobs_export["job_id"] = np.arange(len(jobs_export))
jobs_export["start_time"] = jobs_export["finish_time"] - jobs_export["t_speed"]

drone_finish_map = {d: drone_finish[d] for d in range(n_drones)}
jobs_export["drone_finish_time"] = jobs_export["drone_id"].map(drone_finish_map)

drone_location_map = {d: drone_location[d] for d in range(n_drones)}
jobs_export["drone_location"] = jobs_export["drone_id"].map(drone_location_map)

jobs_export["is_mesh_candidate"] = jobs_export["drone_id"] < mesh
jobs_export["assignment_step"] = jobs_export["drone_id"]

printer_lookup = printers[["id", "model", "location", "speed_factor"]].rename(columns={
    "id": "printer_id_lookup",
    "model": "printer_model",
    "location": "printer_location_lookup",
    "speed_factor": "printer_speed_factor_lookup"
})

jobs_export = jobs_export.merge(
    printer_lookup,
    how="left",
    left_on="printer_id",
    right_on="printer_id_lookup"
)

if "speed_factor" not in jobs_export.columns:
    jobs_export["speed_factor"] = jobs_export["printer_speed_factor_lookup"]
else:
    jobs_export["speed_factor"] = jobs_export["speed_factor"].fillna(jobs_export["printer_speed_factor_lookup"])

jobs_export.drop(
    columns=["printer_id_lookup", "printer_location_lookup", "printer_speed_factor_lookup"],
    inplace=True
)

job_cols = [
    "strategy", "run_id", "n_drones", "mesh", "glut",
    "job_id", "drone_id", "part",
    "t_nominal", "speed_factor", "t_speed",
    "location", "printer_id", "printer_model",
    "start_time", "finish_time", "drone_finish_time",
    "drone_location", "is_mesh_candidate", "assignment_step"
]
jobs_export = jobs_export[[c for c in job_cols if c in jobs_export.columns] +
                          [c for c in jobs_export.columns if c not in job_cols]]

jobs_export.to_csv(output_dir / f"jobs_{file_tag}.csv", index=False)

# drones
drones_export = (
    jobs_export.groupby("drone_id", as_index=False)
    .agg(
        strategy=("strategy", "first"),
        run_id=("run_id", "first"),
        n_drones=("n_drones", "first"),
        mesh=("mesh", "first"),
        glut=("glut", "first"),
        location=("drone_location", "first"),
        n_parts=("part", "count"),
        first_part_start=("start_time", "min"),
        drone_finish_time=("drone_finish_time", "max"),
    )
)

drones_export["drone_flow_time"] = (
    drones_export["drone_finish_time"] - drones_export["first_part_start"]
)

drones_export.to_csv(output_dir / f"drones_{file_tag}.csv", index=False)

# printers
printer_job_counts = (
    jobs_export.groupby("printer_id", as_index=False)
    .agg(
        n_jobs=("job_id", "count"),
        n_parts_arm=("part", lambda s: (s == "arm").sum()),
        n_parts_base_plate=("part", lambda s: (s == "base_plate").sum()),
        n_parts_top_plate=("part", lambda s: (s == "top_plate").sum()),
    )
)

printers_export = printers.copy()
printers_export["strategy"] = strategy
printers_export["run_id"] = run_id

printers_export = printers_export.merge(
    printer_job_counts,
    how="left",
    left_on="id",
    right_on="printer_id"
)

for col in ["n_jobs", "n_parts_arm", "n_parts_base_plate", "n_parts_top_plate"]:
    if col in printers_export.columns:
        printers_export[col] = printers_export[col].fillna(0).astype(int)

printers_export["final_load"] = printers_export["load"]
printers_export["utilisation"] = (
    printers_export["final_load"] / metrics["makespan"]
    if metrics["makespan"] > 0 else np.nan
)

printers_export = printers_export.rename(columns={
    "id": "printer_id",
    "model": "printer_model"
})

printer_cols = [
    "strategy", "run_id", "printer_id", "printer_model", "location",
    "speed_factor", "final_load", "n_jobs",
    "n_parts_arm", "n_parts_base_plate", "n_parts_top_plate",
    "utilisation"
]
printers_export = printers_export[[c for c in printer_cols if c in printers_export.columns] +
                                  [c for c in printers_export.columns if c not in printer_cols]]

printers_export.to_csv(output_dir / f"printers_{file_tag}.csv", index=False)

# locations
location_job_counts = (
    jobs_export.groupby("location", as_index=False)
    .agg(
        drones_assigned=("drone_id", "nunique"),
        jobs_assigned=("job_id", "count")
    )
)

location_printer_summary = (
    printers.groupby("location", as_index=False)
    .agg(
        n_printers=("id", "count"),
        mean_printer_load=("load", "mean"),
        max_printer_load=("load", "max")
    )
)

location_capacity = (
    printers.groupby("location")["speed_factor"]
    .apply(lambda s: (1.0 / s).sum())
    .reset_index(name="effective_capacity")
)

locations_export = pd.DataFrame({"location": location_names})
locations_export["strategy"] = strategy
locations_export["run_id"] = run_id
locations_export["total_location_work"] = locations_export["location"].map(location_work)
locations_export["WPL"] = locations_export["location"].map(WPL_location)
locations_export["phi"] = locations_export["location"].map(phi_location)

locations_export = locations_export.merge(location_printer_summary, on="location", how="left")
locations_export = locations_export.merge(location_capacity, on="location", how="left")
locations_export = locations_export.merge(location_job_counts, on="location", how="left")

locations_export["drones_assigned"] = locations_export["drones_assigned"].fillna(0).astype(int)
locations_export["jobs_assigned"] = locations_export["jobs_assigned"].fillna(0).astype(int)

location_cols = [
    "strategy", "run_id", "location", "n_printers", "effective_capacity",
    "total_location_work", "WPL", "phi",
    "drones_assigned", "jobs_assigned",
    "mean_printer_load", "max_printer_load"
]
locations_export = locations_export[[c for c in location_cols if c in locations_export.columns] +
                                    [c for c in locations_export.columns if c not in location_cols]]

locations_export.to_csv(output_dir / f"locations_{file_tag}.csv", index=False)

# metrics
metrics_row = pd.DataFrame([{
    "strategy": strategy,
    "run_id": run_id,
    "n_drones": n_drones,
    "mesh": mesh,
    "glut": glut,
    "makespan": metrics["makespan"],
    "T_mesh": metrics["T_mesh"],
    "throughput_avg_per_hour": metrics["throughput_avg_per_hour"],
    "throughput_post_mesh_per_hour": metrics["throughput_post_mesh_per_hour"],
    "load_min": metrics["load_min"],
    "load_mean": metrics["load_mean"],
    "load_max": metrics["load_max"],
    "load_std": metrics["load_std"],
    "phi_min": metrics["phi_min"],
    "phi_mean": metrics["phi_mean"],
    "phi_max": metrics["phi_max"],
    "phi_std": metrics["phi_std"],
    "n_drones_total": metrics["n_drones_total"],
}])

metrics_path = output_dir / "metrics_all_runs.csv"

if metrics_path.exists():
    metrics_all = pd.read_csv(metrics_path)
    metrics_all = metrics_all[~(
        (metrics_all["strategy"] == strategy) &
        (metrics_all["run_id"] == run_id)
    )]
    metrics_all = pd.concat([metrics_all, metrics_row], ignore_index=True)
else:
    metrics_all = metrics_row

metrics_all.to_csv(metrics_path, index=False)

print(f"\nExported files for {strategy}:")
print(output_dir / f"jobs_{file_tag}.csv")
print(output_dir / f"drones_{file_tag}.csv")
print(output_dir / f"printers_{file_tag}.csv")
print(output_dir / f"locations_{file_tag}.csv")
print(metrics_path)

# %%