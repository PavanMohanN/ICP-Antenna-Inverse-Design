# Pipeline: every step, its inputs, outputs and settings

The pipeline turns the 500 raw HFSS files into the trained models, the paper tables and the paper figures.

```bash
python run.py reproduce --quick     # small budgets, about 15 min on 2 cores -> runs/quick/
python run.py reproduce             # paper settings, about 2 h on 2 cores  -> runs/full/
python run.py steps                 # list of steps with estimated times
```

## How it works

* `run.py` creates the run folder `runs/<name>/` and calls the scripts of `pipeline/steps/` in order, inside that folder.
* Every script reads its inputs from the run folder and writes its outputs there. The raw data are read from `data/hfss/` (or `--data`).
* Shared code lives in `pipeline/lib/`: `common.py` (design space, folds, decoder), `circuit.py` (Foster model and fit), `models.py` (PC-Net and MLP), `inverse.py` (ensemble, support constraint, generator, refinement, ranking) and `settings.py` (budgets of both modes).
* The designer engine `designer/icp_engine.py` is used by steps 07c, 10, 11 and 15, so the paper results come from the same code as the tool.
* A finished step leaves a marker in `runs/<name>/.done/`. Rerunning the same command skips finished steps, so an interrupted run resumes.
* Output of each step goes to the console and to `runs/<name>/logs/<step>.log`.

## Steps

| Step | Script | Inputs | Outputs | Full, 2 cores |
|---|---|---|---|---|
| 01 | `01_audit.py` | `data/hfss/*.s1p`, `samples.csv`, `runs.log` | `raw.npz`, `audit.json` | < 1 min |
| 02 | `02_fit_circuits.py` | `raw.npz` | `circuit_fits.pkl` | 45 min |
| 03 | `03_process.py` | `raw.npz`, `circuit_fits.pkl` | `processed.npz`, `design_descriptors.csv`, `interp_benchmark_*.csv`, `frequency_coverage.csv`, `fit_quality.json` | 1 min |
| 04a | `04_forward_cv.py ET GP MLPdB MLPri` | `processed.npz` | `fwd_oof_ET_GP_MLPdB_MLPri_*.pkl` | 7 min |
| 04b | `04_forward_cv.py PCraw PC` (folds split over `--jobs`) | `processed.npz` | `fwd_oof_PCraw_PC_*.pkl`, `pcnet_fold0..4.pt` | 33 min |
| 05 | `05_forward_metrics.py` | `processed.npz`, `design_descriptors.csv`, `fwd_oof_*.pkl` | `fwd_cv_per_fold.csv`, `fwd_cv_mean.csv`, `fwd_cv_std.csv` | < 1 min |
| 06 | `06_calibrate.py` | `processed.npz`, `fwd_oof_*.pkl` | `calibration.json` | < 1 min |
| 07a | `07_train_final.py oracle` | `processed.npz` | `pcnet_oracle.pt` (copied to `pcnet_final.pt`) | 14 min |
| 07b | `07_train_final.py generator` | `pcnet_final.pt`, `design_descriptors.csv`, `calibration.json` | `generator_final.pt` | 1.5 min |
| 07c | `07_train_final.py export deploy/model` | `pcnet_final.pt`, `generator_final.pt` | `deploy/model/weights.npz`, `config.json` (+ NumPy = PyTorch check) | < 1 min |
| 08 | `08_kriging_reference.py` | `processed.npz` | `gp_oracle.pkl` | 4 min |
| 09 | `09_inverse_benchmark.py <folds>` (split over `--jobs`) | fold models, `pcnet_oracle.pt`, `gp_oracle.pkl`, `calibration.json` | `inv_eval_rows_*.csv`, `support.json` | 12 min |
| 10 | `10_final_designs.py` | `deploy/model` | `final_designs_top3.csv`, `design_runs/` | 1 min |
| 11 | `11_design_grid.py` | `deploy/model`, `processed.npz` | `final_design_grid.csv`, `catalog_coverage.csv` | 3 min |
| 12 | `12_physics_check.py` | `design_descriptors.csv`, `final_design_grid.csv` | `physics_check.json` | < 1 min |
| 13 | `13_sensitivity.py` | `gp_oracle.pkl`, `pcnet_oracle.pt` | `parameter_sensitivity.csv` | 1 min |
| 14 | `14_tables.py` | results of steps 01-13 | `tables/` (Tables II-IV as LaTeX and CSV, `paper_numbers.json`, `paper_text_checks.json`) | < 1 min |
| 15 | `15_export_figure_data.py` | results of steps 02-11 | `figure_data/` | < 1 min |
| 16 | `16_export_csv.py` | `processed.npz` | `csv/` (design table, raw S11 in dB, Foster parameters, descriptors) | < 1 min |
| 17 | `figures/make_figures.py` | `figure_data/` | `figures/` (PDF and PNG) | < 1 min |

## Settings of the two modes (`pipeline/lib/settings.py`)

| Setting | Full (paper) | Quick |
|---|---|---|
| Foster fit: orders K | 2, 3, 4, 5 | 3, 4 |
| Foster fit: starts per fit | 9 | 3 |
| Foster fit: evaluations per start / polish | 300 / 3000 | 60 / 200 |
| Ensemble members (CV and deployed) | 5 | 2 |
| PC-Net epochs | 4000 | 1500 |
| MLP epochs | 3000 | 300 |
| Extra Trees trees | 500 | 100 |
| Kriging restarts (CV / reference) | 2 / 1 | 0 / 0 |
| Kriging reference grid | 10 MHz | 100 MHz |
| Generator iterations (deployed / benchmark) | 2500 / 1500 | 500 / 400 |
| Inverse benchmark folds | 0-4 | 0 |
| Held-out specifications per fold | 20 | 4 |
| Frequency targets | 4.2-6.8 GHz, step 0.2 | 4.6, 5.4, 6.2 GHz |
| Candidates / refinement steps per query | 128 / 100 | 32 / 20 |
| Design grid (Fig. 5b) | 4.1-6.9 GHz, step 0.1 | 4.3, 4.9, 5.5, 6.1, 6.7 GHz |

The full mode gives the paper numbers: a complete rerun (2.0 h on a 2-core VM) reproduced every model file, table, figure and quoted number of `runs/paper/` exactly; only the measured run times differ. Fixed seeds everywhere: folds 2027, ensemble members 0-4 (cross-validation) and 100-104 (deployed), generator 0, designer 2027.
Bit-for-bit equality needs the same library versions (`requirements-lock.txt`); other versions can change the last digit of some numbers.

## The reference run `runs/paper/`

It holds every output of the paper run, except `processed.npz` (12 MB), which is rebuilt bit for bit from `raw.npz` and `circuit_fits.pkl` by step 03 in a few seconds the first time it is needed (`run.py test`, the test suite, or `reproduce --name paper --force`).

## Parallelism

* `--jobs N` (default: number of cores, at most 5) sets the number of processes for step 02 and splits the folds of steps 04b and 09 over N processes.
* Every fold and every design is computed independently with its own seed, so the results do not depend on `--jobs`.
* PyTorch uses one thread per process (set in the scripts) for reproducibility.

## Running a single step by hand

`run.py` is the easiest way (`python run.py reproduce --only 14`). To call a script directly, run it inside the run folder:

```bash
cd runs/quick
python ../../pipeline/steps/14_tables.py
```

Set `ICP_MODE=quick` in the environment for a quick-mode run folder, because some steps read their budgets from it.
