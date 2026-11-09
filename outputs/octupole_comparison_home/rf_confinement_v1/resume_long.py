"""Continue two matched high-energy rotating runs to 10,000 RF periods."""
import math
import run_v3 as v


def main():
    rows = list(v.read_journal().values())
    jobs = []
    for phase, direction in ((0.0, 1.0), (math.pi / 2, -1.0)):
        row = next(
            x for x in rows
            if x["job"]["model"] == "rotating"
            and x["job"]["field"] == "h35"
            and x["job"]["voltage_V"] == 250.0
            and x["job"]["frequency_Hz"] == 100000.0
            and x["job"]["mz_Th"] == 100000.0
            and x["job"]["r0_mm"] == 1.0
            and x["job"]["energy_meV"] == 1000.0
            and x["job"]["azimuth_rad"] == math.pi / 4
            and x["job"]["phase_rad"] == phase
            and x["job"]["direction"] == direction
            and x["job"]["horizon_RF"] == 1000
            and x["job"].get("stage") == "near_aperture_boundary"
        )
        job = dict(row["job"])
        job.update(horizon_RF=10000, stage="long_horizon_high_energy")
        jobs.append(job)
    print(v.run_jobs(jobs, workers=4), flush=True)


if __name__ == "__main__":
    main()
