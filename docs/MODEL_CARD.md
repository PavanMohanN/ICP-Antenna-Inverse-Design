# Model card: ICP antenna designer (PC-Net ensemble + ICP generator)

**Task.** An S-parameter target (a band in 4–7 GHz and a return loss) maps to the 9 free HFSS variables of the `WP_AI_V1` antenna. Two variables follow from them, and two are fixed. The forward direction (dimensions → S11) is also available.

**Files.** `designer/model/weights.npz` and `config.json` (NumPy, used by the designer); `runs/paper/pcnet_final.pt` and `generator_final.pt` (PyTorch); `runs/paper/pcnet_fold0..4.pt` (cross-validation models).

**Training data.** 500 HFSS 2024.2 designs, 31-point S11 sweep from 4 to 7 GHz, sampled uniformly inside these bounds:

| Variable | Range (mm) | | Variable | Range (mm) |
|---|---|---|---|---|
| Air_gap | 1.5–5 | | W_Patch1 | 8–17 |
| W_Patch | 10–23 | | WS | 0.2–1.5 |
| L_Patch | 15–23 | | Curve_Amp1 | 0.5–4 |
| Sx | 0.2–5 | | Extra | 0.5–8 |
| Sy | 0.2–5 | | | |

Fixed: H_sub = 0.79 mm, Curve_len1 = 1. Derived: Curve_pos = 0.5·L_Patch + 2, L_Patch1 = W_Patch1 + Extra.

**Forward model (PC-Net).**
* A 9 → 128 → 128 → 128 SiLU MLP outputs the elements of a Foster equivalent circuit: delay τ, R0, L0, C0 and 6 parallel-RLC resonators (f_k, Q_k, R_k).
* S11 = e^(−j2πfτ)(Z − 1)/(Z + 1). Every element is non-negative, so the model is passive and causal for any input.
* Ensemble of 5 members trained on all 500 designs, each with its own 50-design early-stopping split. The same ensemble is the reference that scores the inverse benchmark, where the designs come from fold models that never saw the test fold.
* Uncertainty: ensemble σ. Calibrated bound μ + κσ with κ = 1.22 (empirical one-sided 84 % bound on out-of-fold data).

**Inverse model (ICP generator).**
* g(f*, B, z), z ~ N(0, I₄), sigmoid output (box constraints are hard).
* Trained through the frozen ensemble with a robust band constraint (μ + κσ ≤ −11 dB over the band), a k-NN data-support constraint (k = 3, d_max = 95th percentile) and a mode-seeking diversity term.
* Test time: constrained refinement, snapping to the 0.01 mm grid, ranking, diversity filter.

**Validated performance** (five-fold cross-validation by geometry, unseen HFSS designs; `runs/paper/fwd_cv_mean.csv`):
* |S11| MAE 0.84 dB; resonance error 118 MHz (62 % within 50 MHz).
* −10 dB detection: F1 0.62, AUC 0.977. Screening precision@5: 55 % (base rate 4.7 %).
* Passivity violations: 0.
* Inverse: 98 % of held-out specifications met; calibrated success 65 % (Table III).

**Intended use.** Fast first-pass synthesis of HFSS-ready dimensions, and fast S11 prediction of candidate geometries, with calibrated uncertainty and a confidence flag.

**Known limits.**
* Above about 6.3 GHz few simulated designs are matched; screening precision drops and σ grows.
* S11 only: no gain, pattern or efficiency.
* Valid only inside the bounds above, with H_sub = 0.79 mm and the same materials and feed.
* Bands wider than any simulated band (0.975 GHz at −10 dB, 0.629 GHz at −15 dB) are extrapolations and are flagged.
* The confidence flag is LOW outside the data support.
