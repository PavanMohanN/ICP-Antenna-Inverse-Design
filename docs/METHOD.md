# Method: the paper in one page, and the implementation in detail

## 1. The antenna (paper, Section II and Fig. 1)

* Three-layer patch antenna for secure C-band links. Two 60 × 60 mm PCBs with 0.79 mm substrates, separated by an air gap G set by a 3-D printed PLA fixture.
* Bottom PCB: driven patch W × L above a solid ground plane, fed from a 50 Ω microstrip line through a folded high-impedance transformer (width W_T, meander width C, length L_T). No plated through holes.
* Top PCB: four tightly coupled parasitic patches W1 × L1, separated by the gaps Sx and Sy. Its substrate also acts as a radome.

| Quantity | Value |
|---|---|
| Impedance bandwidth | 4.4–5.4 GHz (20.4 %) |
| Gain | 8–10 dBi, stable boresight |
| E- / H-plane beamwidth | 58°–46° / 70°–62° |
| Radiation efficiency (simulated) | 93 % |
| Size (without fixture) | 60 × 60 × 5.2 mm = 0.98λ₀ × 0.98λ₀ × 0.085λ₀ at 4.9 GHz |

## 2. The problem

* Nine dimensions interact strongly. One HFSS run takes 161 s on average; the 500-design data set took 22.3 h.
* Generic surrogates have three weaknesses on such resonant structures:
  1. the 100 MHz sweep aliases the high-Q resonances;
  2. black-box regressors can predict |S11| > 1 (non-passive);
  3. unconstrained inverse networks exploit surrogate errors, because the inverse mapping is one-to-many.

## 3. ICP in three stages (paper, Section III and Fig. 2)

| Stage | What it does | Guarantee |
|---|---|---|
| 1. Physically consistent data | every HFSS response is fitted with a Foster equivalent circuit; the order K = 2…5 is chosen by frequency hold-out; the result is a 1 MHz response per design | all elements ≥ 0 ⇒ positive-real impedance ⇒ \|S11\| ≤ 1 and causal |
| 2. PC-Net forward ensemble | an MLP predicts the circuit elements from the nine dimensions; a differentiable decoder gives S11(f); five members give μ and σ; κ = 1.22 makes μ + κσ an 84 % one-sided bound | passive and causal for any input geometry |
| 3. ICP inverse | a conditional generator g(f*, B, z) is trained through the frozen ensemble with a robust band constraint, a data-support constraint and a diversity term; at test time: refinement, snapping to the 0.01 mm grid, ranking | box limits, grid and dependent variables hold exactly |

### Equivalent circuit (paper, eq. 1)

```
S11(f) = exp(-j 2π f τ) · (Z(f) - 1) / (Z(f) + 1)
Z(f)   = R0 + j a f - j b / f + Σ_k R_k / (1 + j Q_k (f/f_k - f_k/f))        (Z normalised to 50 Ω, f in GHz, τ in ns)
```

In physical units (Z0 = 50 Ω, f_k in GHz): R0 [Ω] = 50·R0, L0 [nH] = 50·a/(2π), C0 [pF] = 1000/(2π·50·b); tank k: R [Ω] = 50·R_k, L [nH] = R/(2π f_k Q_k), C [pF] = 1000·Q_k/(2π f_k R).

### Inverse loss (paper, eq. 2)

```
L = ⟨ [μ + κσ + 11]_+^2 ⟩_B  +  λ_s [d_kNN(u) - d_max]_+^2  -  λ_d ‖u1 - u2‖ / ‖z1 - z2‖
```

* First term: |S11| ≤ −10 dB with a 1 dB margin on the calibrated bound, averaged over 9 frequencies of the requested band.
* Second term: stay inside the sampled domain; d_max is the 95th percentile of the 3-nearest-neighbour distance between simulated designs.
* Third term: distinct latent vectors give distinct designs (multiple solutions of the one-to-many problem).

## 4. Key results (paper, Section IV)

All learning results use one fixed five-fold split by geometry, so every error refers to designs the model never saw.
The full tables and the file behind every number are in [RESULTS.md](RESULTS.md).

* **Super-resolution.** Fitting 16 samples and predicting the 15 held out, the circuit has an MAE of 0.11 dB, against 0.35–1.24 dB for interpolants; near resonance it is 7.2 times better; it makes 20 wrong −10 dB decisions against at least 165, and never violates passivity.
* **Forward model (Table II).** PC-Net: MAE 0.84 dB, resonance error 118 MHz, F1 0.62, screening precision 55 % (best baseline 27 %, base rate 4.7 %), no passivity violation.
* **Inverse design (Table III).** ICP meets 98 % of the held-out specifications and has the highest calibrated success, 65 %; the constraints reduce the optimism from 6.2 to 3.8 dB.
* **Physics.** Designs with C < 2 mm follow f·L = 97.3 ± 2.9 GHz·mm, close to 96.8 ± 5.1 GHz·mm of the simulated designs, although the law was never imposed.

### What the paper claims, and what it does not

* It claims passivity and causality by construction; better accuracy and screening on unseen full-wave designs; calibrated uncertainty; diverse, physically scaled inverse designs in seconds.
* It does not claim that the inverse designs were re-simulated (a reference surrogate scores them), gain or pattern prediction (the data contain S11 only), or that the reported antenna's dimensions came from the tool.

## 5. Implementation details

### 5.1 Data and protocol (`pipeline/steps/01_audit.py`, `pipeline/lib/common.py`)

* 500 HFSS 2024.2 designs sampled uniformly inside the box of the README notation table; S11 sweep 4–7 GHz, 31 points, format MA.
* Folds: random permutation with seed 2027, five folds of 100 designs. The same folds are used for every model.

### 5.2 Physically consistent representation (`circuit.py`, steps 02 and 03)

* Bounds: τ ∈ [0, 2.5] ns, R0 ∈ [0, 5], a ∈ [0, 5], b ∈ [0, 200]; f_k ∈ [3, 8.5] GHz, Q_k ∈ [0.5, 400], R_k ∈ [0, 60].
* Fit: bounded trust-region least squares on the complex residual; 9 starts (τ ∈ {0, 0.1, 0.2, 0.3} ns with seeds at the peaks of Re Z, plus 5 random starts); the best start is polished.
* Order selection: fit the 16 even-indexed samples, score the 15 odd ones, take the smallest K within 10 % of the best hold-out error, refit on all 31 samples.
* Outputs: 1 MHz responses; resonance f_r, minimum |S11|, −10 dB band, number of resonances; fit quality (median complex RMSE 0.011).

### 5.3 PC-Net (`models.py`, steps 04 and 07)

| Item | Setting |
|---|---|
| Inputs | 9 dimensions, min–max scaled to [0, 1] |
| Network | 3 × 128 SiLU layers; 22 outputs: τ, R0, a, b and (f_k, Q_k, R_k) for 6 resonators |
| Output heads | τ = 0.6·sigmoid; R0, a, b, R_k = softplus × (0.1, 0.2, 2, 5); f_k = 2.8 + 5.4·sigmoid GHz; Q_k = exp(1 + 5·sigmoid) |
| Loss | complex MSE + 0.02 × dB-MAE (floor −40 dB), on the 31 raw samples and on the 10 MHz responses, equal weights |
| Optimiser | AdamW, lr 3e-3, weight decay 1e-4, cosine schedule, 4000 full-batch epochs, gradient clip 5, early stopping on a 10 % validation split |
| Ensemble | 5 members with different validation splits; μ and σ of |S11| in dB |
| Calibration | κ = 84th percentile of (y − μ)/σ on out-of-fold points with y or μ ≤ −5 dB: κ = 1.22 |

### 5.4 ICP inverse (`inverse.py`, steps 07 and 09; `designer/icp_engine.py`)

* Generator: input (f*, B, z), z ∈ ℝ⁴; 3 × 128 SiLU MLP with a sigmoid output; Adam, lr 2e-3, cosine schedule, batch 256, 2500 iterations (1500 inside the cross-validation).
* Training conditions: 75 % specifications (f_c, B) achieved by training designs, with jitter; 25 % uniform f* ∈ [4.05, 6.95] GHz with B ≤ 80 MHz.
* Constraint weights: λ_s = 50 (support, k = 3), λ_d = 0.05 (diversity, ratio clamped at 0.5).
* Test time (benchmark): 128 candidates, half from the generator and half uniform; 100 Adam steps in logit space (lr 0.05); snap to 0.01 mm; re-evaluate on a 5 MHz grid; rank; diversity filter (mean |Δu| ≥ 0.08); top five.
* Ranking: (1) calibrated feasibility inside the data support; (2) calibrated band centred on f* at least max(0.2 GHz, B) wide, about twice the resonance error of the surrogate; (3) calibrated margin.
* Designer (NumPy, `icp_engine.py`): the same objective and ranking; 160 candidates spread over five band levels when no band is given, 60 Adam steps with finite-difference gradients; about 6 s per query on two cores. The required level is −RL dB (`--rl`).

### 5.5 Baselines and evaluation (steps 04, 05, 09)

* Forward baselines: Extra Trees (500 trees, min leaf 2); kriging (Matérn-5/2 ARD + white noise); MLP ensembles on dB and on Re/Im.
* Inverse baselines: catalogue retrieval from the training designs; direct inverse MLP; conditional VAE; unconstrained tandem network (deterministic, κ = 0, no support term); constrained optimisation from random starts; two ablations of ICP.
* Screening test (non-circular): in each test fold, rank the 100 unseen designs at each frequency; precision@5 is the fraction of the top five whose real HFSS response meets −10 dB.
* Inverse scoring: a reference 5-member PC-Net trained on all 500 designs with other seeds scores designs proposed by fold models that never saw the test fold. Kriging was rejected as a reference, because it detects fewer than 7 % of the matched samples.
* Optimism = reference minus self-predicted |S11(f*)|; it measures how much a method exploits the errors of its own surrogate.
