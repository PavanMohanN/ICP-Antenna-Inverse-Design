# Designer manual

The designer is the part of the repository you use to design antennas. It needs only NumPy; SciPy is needed for the RF circuit and matplotlib for the plots. It does not need PyTorch.

| Tool | Direction | Command |
|---|---|---|
| `designer/design_antenna.py` | S-parameter target → dimensions (inverse) | `python run.py design ...` |
| `designer/predict_s11.py` | dimensions → S11 (forward) | `python run.py predict ...` |

Both accept `--model FOLDER` to use a model you trained yourself (`runs/<name>/deploy/model`).

## 1. Inverse: `design_antenna.py`

```bash
python run.py design --band 5 6 --rl 10 --circuit           # |S11| <= -10 dB over 5-6 GHz, with RF circuit
python run.py design --band 5 6 --rl 15                     # |S11| <= -15 dB over 5-6 GHz
python run.py design --freq 5.5                             # widest calibrated -10 dB band centred on 5.5 GHz
python run.py design --freq 4.9 --bw 1.0                    # same as --band 4.4 5.4 (Table IV)
python run.py design --freq 4.8 5.2 6.0 --top 3 --out my    # several targets
python run.py design                                        # interactive: asks for band, return loss and circuit
```

### What happens inside

1. **Candidates.** 160 starting designs: half from the learned inverse generator g(f*, B, z), half uniform in the design box. Without a band, they are spread over five band levels (0 to 0.6 GHz).
2. **Constrained refinement.** 60 Adam steps on the ICP objective: the calibrated bound μ + κσ must be at least 1 dB below −RL at 9 frequencies across the band; a penalty keeps the design inside the data support.
3. **Hard constraints.** Box bounds by construction (sigmoid); snapping to the 0.01 mm HFSS grid; dependent variables computed exactly (`Curve_pos = 0.5·L_Patch + 2`, `L_Patch1 = W_Patch1 + Extra`); fixed `H_sub = 0.79 mm`, `Curve_len1 = 1`.
4. **Ranking.** (1) calibrated bound meets the target over the band, inside the support; (2) calibrated band centred on f* at least max(0.2 GHz, B) wide; (3) largest calibrated margin.
5. **Diversity.** The top K designs differ by at least 8 % of the range on average.

### Output columns of `designs.csv`

| Column | Meaning |
|---|---|
| `design_id` | `ICP_<target>_<rank>` |
| `target_GHz`, `requested_band_GHz`, `required_S11_dB`, `required_return_loss_dB` | the request |
| `Air_gap` … `Extra` | the 9 free dimensions (mm) |
| `Curve_pos`, `L_Patch1`, `H_sub`, `Curve_len1` | derived and fixed variables |
| `pred_S11_at_centre_dB`, `pred_S11_std_dB` | ensemble mean and σ at the centre frequency |
| `worst_S11_in_band_mean_dB`, `worst_S11_in_band_calibrated_dB` | highest \|S11\| inside the requested band, mean and calibrated bound |
| `band_coverage_mean_pct`, `band_coverage_calibrated_pct` | share of the requested band that meets the target |
| `pred_resonance_GHz` | deepest predicted minimum within ±0.5 GHz of the centre |
| `pred_band_lo/hi_GHz`, `pred_BW_MHz` | contiguous band around the centre where the mean meets the target |
| `calibrated_band_lo/hi_GHz`, `calibrated_BW_MHz` | the same for the calibrated bound |
| `support_distance` | mean distance to the 3 nearest simulated designs (unit cube); the support limit is 0.65 |
| `confidence` | HIGH, MEDIUM, MEDIUM-LOW or LOW, plus an EXTRAPOLATION note when relevant |

### Confidence levels

| Level | Condition |
|---|---|
| HIGH | calibrated bound meets the target over the whole band, σ at the centre < 2 dB |
| MEDIUM | calibrated bound meets the target, σ ≥ 2 dB |
| MEDIUM-LOW | only the mean prediction meets the target |
| LOW | the mean misses the target (closest design returned), or the design is outside the simulated design space |
| + EXTRAPOLATION | the predicted band is wider than any band among the 500 simulated designs: 0.975 GHz at −10 dB, 0.833 at −12, 0.629 at −15, 0.299 at −20 |

## 2. Forward: `predict_s11.py`

```bash
python run.py predict --dims 1.78 11.93 18.61 0.34 4.70 15.35 1.36 2.83 5.07          # HFSS order, mm
python run.py predict --dims 1.78 11.93 18.61 0.34 4.70 15.35 1.36 2.83 5.07 --circuit
python run.py predict --csv designer/examples/5-6GHz_RL10/hfss_samples.csv
python run.py predict --csv data/hfss/samples.csv --rows 101 291 72                       # with HFSS overlay
python run.py predict --dims ... --rl 15                                                  # band defined at -15 dB
```

The order of `--dims` is `Air_gap W_Patch L_Patch Sx Sy W_Patch1 WS Curve_Amp1 Extra` (paper: G W L Sx Sy W1 W_T C E).
For a simulated design (`run_id` of `data/hfss/samples.csv`), the HFSS response is overlaid and the error reported; such designs were in the training set, so this is an in-sample check.

## 3. RF equivalent circuit (`--circuit`, `rf_circuit.py`)

The predicted complex S11 (ensemble mean, 10 MHz grid) is fitted with the passive, causal Foster circuit used inside PC-Net:

```
S11(f) = exp(-j 2π f τ) · (Z - 50) / (Z + 50)
Z(f)   = R0 + j2πf·L0 + 1/(j2πf·C0) + Σ_k (R_k ∥ L_k ∥ C_k)
```

| Element | Meaning |
|---|---|
| τ | round-trip delay of the feed up to the reference plane (a lossless 50 Ω line of one-way delay τ/2) |
| R0, L0, C0 | series loss, feed (probe) inductance and coupling (gap) capacitance |
| f_k, Q_k | resonance frequency and loaded Q of resonator k: f_k = 1/(2π√(L_k C_k)), Q_k = R_k √(C_k/L_k) |
| R_k, L_k, C_k | element values of resonator k (Ω, nH, pF) |

* The order K (2–5) is the smallest one whose complex RMSE is within 10 % of the best order.
* All elements are non-negative, so the circuit is passive and causal by construction.
* Resonators outside 4–7 GHz only shape the band edges; do not read them as physical modes.
* The file reports the fit quality (complex RMSE and the largest \|S11\| error in dB, which is largest at deep nulls).
* Use the circuit in ADS, AWR or HFSS Circuit, for example to design a matching network.

## 4. Worked examples (in `designer/examples/`)

| Folder | Command | Result |
|---|---|---|
| `4.4-5.4GHz_RL10` | `--band 4.4 5.4 --rl 10` | Table IV, 4.9 GHz column: mean band 4.35–5.51 GHz, calibrated 4.42–5.22 GHz |
| `5.50GHz_max_band` | `--freq 5.5` | Table IV, 5.5 GHz column: −10 dB band 5.27–5.92 GHz |
| `5-6GHz_RL10` | `--band 5 6 --rl 10 --circuit` | design 1 meets −10 dB over the whole band even with the calibrated bound (−10.9 dB worst), but the band is wider than any simulated design: one full-wave check is advised |
| `5-6GHz_RL15` | `--band 5 6 --rl 15 --circuit` | not achievable: no design reaches −15 dB over all of 5–6 GHz; design 2 meets −15 dB over 5.14–5.955 GHz (82 % of the band), worst −10.0 dB |

For a 15 dB return loss, a narrower band is realistic: `python run.py design --freq 5.5 --rl 15` gives about 5.3–5.9 GHz.

## 5. A standalone Windows program

`designer/build_exe.bat` builds `dist\design_antenna.exe` with PyInstaller (run it once on a Windows PC with Python). The `.exe` runs without Python: `design_antenna.exe --band 5 6 --rl 10 --circuit`.
`run_design.bat` (Windows) and `run_design.sh` (macOS/Linux) are double-click launchers of the interactive mode.
