# Results, and where every number of the paper comes from

All numbers below are in `runs/paper/` (the run used for the paper). A full rerun (`python run.py reproduce`) writes the same files into `runs/full/`; `python run.py compare full` checks them against the paper.

Step 14 writes two machine-readable summaries:

* `runs/paper/tables/paper_numbers.json`: every metric in full precision.
* `runs/paper/tables/paper_text_checks.json`: every number quoted in the paper text, formatted as in the paper, with the sentence fragment that contains it. `tests/test_07_paper_consistency.py` checks that each fragment appears verbatim in `paper/Paper.tex`.

## Super-resolution of the coarse sweep (Section IV-A, Fig. 3)

Fit the 16 even-indexed HFSS samples of every design, predict the 15 held-out samples.

| Metric | Equivalent circuit | Interpolants |
|---|---|---|
| MAE | 0.11 dB | 0.35–1.24 dB |
| MAE near resonance (\|S11\| ≤ −6 dB) | 0.31 dB | 7.2 times higher at best |
| Wrong −10 dB decisions (500 designs) | 20 | at least 165 |
| Passivity violations | 0 | both cubic splines violate |

* 293 designs meet −10 dB on the 1 MHz grid, against 265 on the raw 31-point grid; 91 designs have a true minimum more than 3 dB below their best sample.
* Fig. 3 (design run_0101, held-out 5.9 GHz sample): HFSS −35.1 dB, linear interpolation −11.5 dB, equivalent circuit −33.0 dB.

Files: `interp_benchmark_summary.csv`, `interp_benchmark_per_design.csv`, `design_descriptors.csv`, `figure_data/fig_extraction.npz`.

## Forward surrogates on unseen geometries (Table II, Fig. 4)

| Model | MAE (dB) | f_r error (MHz) | F1 | P@5 (%) | \|S11\| > 1 (%) |
|---|---|---|---|---|---|
| Extra Trees | 1.48 ± 0.08 | 231 | 0.02 | 19 | 0.0 |
| Kriging | 1.25 ± 0.05 | 258 | 0.10 | 27 | 0.7 |
| MLP, dB | 1.52 ± 0.09 | 264 | 0.06 | 25 | 3.7 |
| MLP, Re/Im | 2.59 ± 0.25 | 495 | 0.28 | 21 | 9.3 |
| **PC-Net** | **0.84 ± 0.12** | **118** | **0.62** | **55** | **0.0** |

* 62 % of the resonances are predicted within 50 MHz.
* Precision@5 is checked against the real HFSS response of unseen designs; the base rate is 4.7 %.
* κ = 1.22; the calibrated rule μ + κσ ≤ −10 dB is correct in 87.5 % of its decisions.

Files: `fwd_cv_mean.csv`, `fwd_cv_std.csv`, `fwd_cv_per_fold.csv` (every metric per fold), `fwd_oof_*.pkl` (out-of-fold predictions), `calibration.json`, `tables/table2_forward_models.tex`.

## Inverse design on held-out specifications (Table III, Fig. 5a)

100 specifications (f_c, B) achieved by unseen test-fold designs plus 70 single-frequency targets, scored by a reference PC-Net trained on all 500 designs with other seeds.

| Method | Spec. (%) | Calib. (%) | Optimism (dB) | Time (s) |
|---|---|---|---|---|
| Catalogue retrieval | 80 | 49 | – | < 0.01 |
| Direct inverse MLP | 6 | 1 | 0.0 | < 0.01 |
| Conditional VAE | 8 | 1 | 0.2 | < 0.01 |
| Tandem (unconstrained) | 90 | 33 | 2.9 | < 0.01 |
| Constrained optimisation, random init. | 98 | 55 | 3.3 | 1.77 |
| ICP generator only | 96 | 60 | 3.0 | 0.07 |
| ICP without constraints | 94 | 58 | 6.2 | 1.62 |
| **ICP (proposed)** | **98** | **65** | 3.8 | 1.76 |

* Initialised from the generator, the search reaches 74 % calibrated success without refinement, against 46 % from random starts (Fig. 5a).
* Above 6.5 GHz, at most seven simulated designs meet −10 dB at the target.
* Times depend on the computer (two-core CPU here).

Files: `inv_eval_rows_*.csv` (every proposed design), `tables/inverse_top1_with_optimism.csv`, `tables/table3_inverse_all_metrics.csv`, `catalog_coverage.csv`, `figure_data/fig_inverse_amortisation.csv`.

## Physical consistency and sensitivity (Sections II-B and IV-C, Fig. 5b)

* Designs with C < 2 mm follow f·L = 97.3 ± 2.9 GHz·mm, against 96.8 ± 5.1 GHz·mm for the simulated designs (`physics_check.json`).
* From about 6 GHz, the method switches to a second family with large C and small G (`final_design_grid.csv`).
* C and L dominate the resonance: about 0.29 and 0.20 GHz per millimetre (`parameter_sensitivity.csv`).

## Final designs (Table IV, Fig. 5c)

| f* (GHz) | 4.5 | 4.9† | 5.0 | 5.5 | 5.8 | 6.2 | 6.6 |
|---|---|---|---|---|---|---|---|
| G | 2.87 | 1.78 | 3.24 | 4.46 | 3.64 | 1.73 | 1.82 |
| W | 17.89 | 11.93 | 14.95 | 17.27 | 12.15 | 14.83 | 13.02 |
| L | 21.09 | 18.61 | 19.51 | 17.88 | 17.32 | 18.81 | 18.12 |
| W_T | 1.22 | 1.36 | 1.05 | 1.18 | 0.78 | 1.04 | 1.08 |
| C | 1.06 | 2.83 | 0.90 | 0.96 | 0.97 | 3.31 | 3.12 |
| W1 | 15.47 | 15.35 | 13.47 | 11.20 | 10.88 | 13.39 | 13.17 |
| L1 | 17.05 | 20.42 | 17.94 | 15.30 | 14.23 | 20.62 | 19.91 |
| Sx | 1.09 | 0.34 | 0.96 | 2.02 | 1.34 | 0.47 | 0.38 |
| Sy | 1.29 | 4.70 | 3.81 | 3.77 | 3.48 | 4.28 | 4.56 |
| \|S11(f*)\| (dB) | −22 ± 2 | −13 ± 2 | −24 ± 3 | −23 ± 2 | −27 ± 9 | −22 ± 2 | −25 ± 2 |
| f_L (GHz) | 4.32 | 4.35 | 4.70 | 5.27 | 5.54 | 6.05 | 6.46 |
| f_H (GHz) | 4.75 | 5.51 | 5.54 | 5.92 | 6.10 | 6.37 | 6.76 |

† Requested band B = 1 GHz (4.4–5.4 GHz). The calibrated bound narrows the predicted band to 4.42–5.22 GHz.

Files: `final_designs_top3.csv` (three designs per target), `design_runs/<target>/` (the designer output, including `.s1p` files), `tables/table4_final_designs.tex`.

## Data set (Sections II-B and III)

* 500 designs, 31 frequencies from 4 to 7 GHz; HFSS time 161 s per design on average, 22.3 h in total (`audit.json`, from `data/hfss/runs.log`).
* Variable ranges (Table I): `csv/design_table.csv`.
