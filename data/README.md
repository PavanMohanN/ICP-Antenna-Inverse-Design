# Data

`hfss/` holds the raw full-wave data used by the paper: 500 designs of the air-gap patch antenna simulated in
Ansys HFSS 2024.2 (project `WP_AI_V1_sweep`).

| File | Contents |
|---|---|
| `run_0000.s1p` … `run_0499.s1p` | S11 of each design, Touchstone, 31 points from 4 to 7 GHz (`# GHz S MA R 50`); the design variables are in the file header |
| `samples.csv` | the 9 free variables of every design (`run_id, Air_gap, W_Patch, L_Patch, Sx, Sy, W_Patch1, WS, Curve_Amp1, Extra`) |
| `runs.log` | HFSS simulation log: time stamp, run id, status, simulation time, variables |

* Variables and ranges: see the notation table of the main README. `Curve_pos = 0.5·L_Patch + 2` and `L_Patch1 = W_Patch1 + Extra` are dependent; `H_sub = 0.79 mm` and `Curve_len1 = 1` are fixed.
* Mean simulation time 160.6 s per design (22.3 h in total).
* Human-readable derived tables (design table, S11 in dB, equivalent-circuit parameters, descriptors) are in `runs/paper/csv/`.
* The Touchstone headers contain the path of the HFSS project on the computer that ran the simulations.
