import pandas as pd
import numpy as np


def run_dedicated_printer_baseline(
    printers,
    jobs,
    n_drones,
    mesh,
    glut,
    total_printers,
    drone_total_workload,
    drone_finish,
    drone_location,
    location_names,
    location_work,
    WPL_location,
    phi_location,
):
    drone_printer = np.empty(n_drones, dtype=object)

    for drone_id in range(n_drones):

        index = drone_id % total_printers
        chosen_printer_id = printers.at[index, "id"]
        chosen_location = printers.at[index, "location"]
        chosen_printer_speed_factor = printers.at[index, "speed_factor"]

        drone_location[drone_id] = chosen_location
        drone_printer[drone_id] = chosen_printer_id

        # all jobs belonging to this drone
        drone_job_rows = jobs.index[jobs["drone_id"] == drone_id]

        for j in drone_job_rows:
            t = jobs.at[j, "t"]
            t_speed = t * chosen_printer_speed_factor

            # add this job consecutively onto the same printer
            printers.at[index, "load"] += t_speed
            finish_time = printers.at[index, "load"]

            # record into jobs table
            jobs.at[j, "location"] = chosen_location
            jobs.at[j, "printer_id"] = chosen_printer_id
            jobs.at[j, "t_speed"] = t_speed
            jobs.at[j, "finish_time"] = finish_time

            # location workload
            if pd.notna(chosen_location):
                location_work[chosen_location] += t_speed

        # drone finishes when its last job on that printer finishes
        drone_finish[drone_id] = printers.at[index, "load"]

    return jobs, printers, drone_finish, drone_location, location_work


def run_two_phase_greedy(
    printers,
    jobs,
    n_drones,
    mesh,
    glut,
    total_printers,
    drone_total_workload,
    drone_finish,
    drone_location,
    location_names,
    location_work,
    WPL_location,
    phi_location,
):
    drone_printer = np.empty(n_drones, dtype=object)

    phaseA_jobs = jobs[jobs["drone_id"] < mesh].sort_values("t", ascending=False).copy()
    phaseB_jobs = jobs[jobs["drone_id"] >= mesh].sort_values("t", ascending=False).copy()

    #2.) - Sequential Drone Assignment Loop Phase A
    for j in phaseA_jobs.index:
        drone_id = jobs.at[j, "drone_id"]
        t = jobs.at[j, "t"]

        # choose least-loaded printer
        index = printers["load"].idxmin()

        chosen_printer_id = printers.at[index, "id"]
        chosen_location = printers.at[index, "location"]
        chosen_printer_speed_factor = printers.at[index, "speed_factor"]

        t_speed = t * chosen_printer_speed_factor

        printers.at[index, "load"] += t_speed
        finish_time = printers.at[index, "load"]

        jobs.at[j, "location"] = chosen_location
        jobs.at[j, "printer_id"] = chosen_printer_id
        jobs.at[j, "t_speed"] = t_speed
        jobs.at[j, "finish_time"] = finish_time

        phaseA_jobs.at[j, "location"] = chosen_location
        phaseA_jobs.at[j, "printer_id"] = chosen_printer_id
        phaseA_jobs.at[j, "t_speed"] = t_speed
        phaseA_jobs.at[j, "finish_time"] = finish_time
        
        if pd.notna(chosen_location):
            location_work[chosen_location] += t_speed

        if finish_time > drone_finish[drone_id]:
            drone_finish[drone_id] = finish_time

    #2.1) - Sequential Drone Assignment Loop Phase B
    for j in phaseB_jobs.index:
        drone_id = jobs.at[j, "drone_id"]
        t = jobs.at[j, "t"]

        # choose least-loaded printer
        index = printers["load"].idxmin()

        chosen_printer_id = printers.at[index, "id"]
        chosen_location = printers.at[index, "location"]
        chosen_printer_speed_factor = printers.at[index, "speed_factor"]

        t_speed = t * chosen_printer_speed_factor

        printers.at[index, "load"] += t_speed
        finish_time = printers.at[index, "load"]

        jobs.at[j, "location"] = chosen_location
        jobs.at[j, "printer_id"] = chosen_printer_id
        jobs.at[j, "t_speed"] = t_speed
        jobs.at[j, "finish_time"] = finish_time

        phaseB_jobs.at[j, "location"] = chosen_location
        phaseB_jobs.at[j, "printer_id"] = chosen_printer_id
        phaseB_jobs.at[j, "t_speed"] = t_speed
        phaseB_jobs.at[j, "finish_time"] = finish_time

        if pd.notna(chosen_location):
            location_work[chosen_location] += t_speed

        if finish_time > drone_finish[drone_id]:
            drone_finish[drone_id] = finish_time

    #2.2.) - Drone summary fields
    for drone_id in range(n_drones):
        drone_rows = jobs.index[jobs["drone_id"] == drone_id]

        if len(drone_rows) == 0:
            continue

        last_row = jobs.loc[drone_rows, "finish_time"].idxmax()
        drone_location[drone_id] = jobs.at[last_row, "location"]
        drone_printer[drone_id] = jobs.at[last_row, "printer_id"]

    return jobs, printers, drone_finish, drone_location, location_work


def run_location_aware(
    printers,
    jobs,
    n_drones,
    mesh,
    glut,
    total_printers,
    drone_total_workload,
    drone_finish,
    drone_location,
    location_names,
    location_work,
    WPL_location,
    phi_location,
):
    location_printer_count = printers.groupby("location")["id"].count().reindex(location_names)
    location_drone_count = pd.Series(0.0, index=location_names)

    phaseA_drones = list(range(mesh))
    phaseB_drones = list(range(mesh, n_drones))

    #2.) - Sequential Drone Assignment Loop Phase A
    for drone_id in phaseA_drones:

        DPL = location_drone_count / location_printer_count
        chosen_location = DPL.idxmin()

        location_drone_count[chosen_location] += 1
        drone_location[drone_id] = chosen_location

        drone_job_rows = jobs.index[jobs["drone_id"] == drone_id]

        for j in drone_job_rows:
            t = jobs.at[j, "t"]

            in_loc = printers["location"] == chosen_location
            index = printers.loc[in_loc, "load"].idxmin()

            chosen_printer_id = printers.at[index, "id"]
            chosen_printer_speed_factor = printers.at[index, "speed_factor"]

            t_speed = t * chosen_printer_speed_factor
            printers.at[index, "load"] += t_speed
            finish_time = printers.at[index, "load"]

            jobs.at[j, "location"] = chosen_location
            jobs.at[j, "printer_id"] = chosen_printer_id
            jobs.at[j, "t_speed"] = t_speed
            jobs.at[j, "finish_time"] = finish_time

            if pd.notna(chosen_location):
                location_work[chosen_location] += t_speed

            if finish_time > drone_finish[drone_id]:
                drone_finish[drone_id] = finish_time

    #2.1) - Sequential Drone Assignment Loop Phase B
    for drone_id in phaseB_drones:

        DPL = location_drone_count / location_printer_count
        chosen_location = DPL.idxmin()

        location_drone_count[chosen_location] += 1
        drone_location[drone_id] = chosen_location

        drone_job_rows = jobs.index[jobs["drone_id"] == drone_id]

        for j in drone_job_rows:
            t = jobs.at[j, "t"]

            in_loc = printers["location"] == chosen_location
            index = printers.loc[in_loc, "load"].idxmin()

            chosen_printer_id = printers.at[index, "id"]
            chosen_printer_speed_factor = printers.at[index, "speed_factor"]

            t_speed = t * chosen_printer_speed_factor
            printers.at[index, "load"] += t_speed
            finish_time = printers.at[index, "load"]

            jobs.at[j, "location"] = chosen_location
            jobs.at[j, "printer_id"] = chosen_printer_id
            jobs.at[j, "t_speed"] = t_speed
            jobs.at[j, "finish_time"] = finish_time

            if pd.notna(chosen_location):
                location_work[chosen_location] += t_speed

            if finish_time > drone_finish[drone_id]:
                drone_finish[drone_id] = finish_time

    return jobs, printers, drone_finish, drone_location, location_work


def run_satiation_aware(
    printers,
    jobs,
    n_drones,
    mesh,
    glut,
    total_printers,
    drone_total_workload,
    drone_finish,
    drone_location,
    location_names,
    location_work,
    WPL_location,
    phi_location,
):
    #2) - Drone Assignment Loop:
    for drone_id in range(n_drones):

        # calculates the total workload of the factory at each drone loop for phi assignment:
        total_workload = location_work.sum()
        if total_workload == 0:
            mean_load = 0 
        else:
            mean_load = total_workload / total_printers
        
        #creates a list of shortlisted locations to print drone in (geographical restriction)
        candidate_locations = []

        # iterates through all locations in location_names and calculates each locations phi score and WPL:
        for location_name in location_names:
            n_printers_location = (printers["location"] == location_name).sum()
            WPL_score = location_work[location_name] / n_printers_location

            if mean_load == 0:
                phi_L = 0 
            else:
                phi_L = WPL_score / mean_load

            WPL_location[location_name] = WPL_score
            phi_location[location_name] = phi_L

        # iterates through all locations in location_names and adds a location to the candidate list if
        # it has no load, or if its predicted workload doesnt become "gluttenous", that is phi > glutteny value
        for location_name in location_names:
            n_printers_location = (printers["location"] == location_name).sum()

            if mean_load == 0:
                candidate_locations.append(location_name)
            else:
                predicted_WPL = (location_work[location_name] + drone_total_workload) / n_printers_location
                predicted_phi = predicted_WPL / mean_load

                if predicted_phi <= glut:
                    candidate_locations.append(location_name)

        # if there is no candidate locations, the drone will be assigned to the location with the lowest phi score,
        # but otherwise, it will be assigned to the candidate location list lowest phi
        if len(candidate_locations) == 0:
            best_location = phi_location.idxmin()
        else:
            best_location = phi_location.loc[candidate_locations].idxmin()
        
        drone_location[drone_id] = best_location

        #assigns all jobs for the current drone in the loop to a sub-joblist called drone_job_rows
        drone_job_rows = jobs.index[jobs["drone_id"] == drone_id]

        for j in drone_job_rows:
            t = jobs.at[j, "t"]

            # only consider printers in the chosen location
            in_loc = printers["location"] == best_location

            # choose the printer with the smallest predicted load and add job multiplied by that printers speed factor
            speed_aware_finish = printers.loc[in_loc, ["load", "speed_factor"]].copy()
            speed_aware_finish["predicted_finish"] = speed_aware_finish["load"] + t * speed_aware_finish["speed_factor"]
            chosen_idx = speed_aware_finish["predicted_finish"].idxmin()
            printers.at[chosen_idx, "load"] += t * printers.at[chosen_idx, "speed_factor"]
            location_work[best_location] += t * printers.at[chosen_idx, "speed_factor"]

            # update printer finish time
            finish_time = printers.at[chosen_idx, "load"]

            # record into the jobs table
            jobs.at[j, "location"] = best_location  # which location job j was assigned to
            jobs.at[j, "printer_id"] = printers.at[chosen_idx, "id"]    # which printer took job j
            jobs.at[j, "t_speed"] = t * printers.at[chosen_idx, "speed_factor"] # time estimate of job j
            jobs.at[j, "finish_time"] = finish_time # printer load after job added

            # update drone finish time + chosen printer list
            if finish_time > drone_finish[drone_id]:
                drone_finish[drone_id] = finish_time

    return jobs, printers, drone_finish, drone_location, location_work


strategy_map = {
    "dedicated_printer_baseline": run_dedicated_printer_baseline,
    "two_phase_greedy": run_two_phase_greedy,
    "location_aware": run_location_aware,
    "satiation_aware": run_satiation_aware,
}