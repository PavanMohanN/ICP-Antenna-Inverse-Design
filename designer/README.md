# Designer

Stand-alone antenna designer (NumPy only). From the repository root:

```bash
python run.py design --band 5 6 --rl 10 --circuit      # S-parameter target -> dimensions (+ RF circuit)
python run.py predict --dims 1.78 11.93 18.61 0.34 4.70 15.35 1.36 2.83 5.07   # dimensions -> S11
```

or from this folder: `python design_antenna.py --band 5 6 --rl 10 --circuit`, `python predict_s11.py --dims ...`.

| File | Role |
|---|---|
| `design_antenna.py` | inverse tool: band and return loss in, dimensions out |
| `predict_s11.py` | forward tool: dimensions in, S11 out |
| `icp_engine.py` | NumPy implementation of the PC-Net ensemble, the generator and the constrained search |
| `rf_circuit.py` | RF equivalent circuit (R, L, C, Q of each resonance) of a predicted S11 |
| `model/` | deployed model: `weights.npz`, `config.json` |
| `examples/` | 4.4–5.4 GHz (Table IV), 5.5 GHz, 5–6 GHz at 10 dB and at 15 dB return loss |
| `run_design.bat`, `run_design.sh` | double-click launchers (interactive mode) |
| `build_exe.bat` | builds a standalone `design_antenna.exe` on Windows (PyInstaller) |

Full manual: [../docs/DESIGNER.md](../docs/DESIGNER.md).
