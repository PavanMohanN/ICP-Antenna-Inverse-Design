"""Run-time settings of the research pipeline.

Two modes, selected with the environment variable ICP_MODE (run.py sets it for you):

    full   (default)  the exact settings of the paper. Reproduces every number in the paper.
                      About 2 h on a 2-core laptop; faster with more cores (see run.py --jobs).
    quick             the same code with small budgets: fewer optimiser starts, epochs, ensemble members,
                      folds and targets. Every step runs and every output file is written, in about
                      15 min on 2 cores. The numbers are NOT the paper's; use it to check the installation
                      and to try the whole algorithm end to end.

Other environment variables
    ICP_DATA     folder with the HFSS export (run_0000.s1p ... run_0499.s1p, samples.csv, runs.log).
                 Default: <repository>/data/hfss
    ICP_WORKERS  processes for the equivalent-circuit fits of step 02 (default: number of CPU cores).
                 The result does not depend on it.
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODE = os.environ.get('ICP_MODE', 'full').strip().lower()
if MODE not in ('full', 'quick'):
    raise ValueError(f"ICP_MODE must be 'full' or 'quick', not {MODE!r}")
QUICK = MODE == 'quick'

DATA_DIR = os.environ.get('ICP_DATA', os.path.join(REPO, 'data', 'hfss'))
DESIGNER_DIR = os.path.join(REPO, 'designer')
WORKERS = int(os.environ.get('ICP_WORKERS', str(os.cpu_count() or 2)))


def pick(full, quick):
    return quick if QUICK else full


# ---- step 02: passive Foster fits of every HFSS response --------------------------------------------
FIT_ORDERS = pick([2, 3, 4, 5], [3, 4])           # K resonators tried for every design
FIT_TAU_STARTS = pick([0.0, 0.1, 0.2, 0.3], [0.0, 0.2])   # delay values used to seed the multistart
FIT_RANDOM_STARTS = pick(5, 1)                     # additional random starts
FIT_NFEV = pick(300, 60)                           # function evaluations per start
FIT_POLISH_NFEV = pick(3000, 200)                  # final polish of the best start

# ---- step 04: forward models, five-fold cross-validation ---------------------------------------------
CV_MEMBERS = pick(5, 2)                            # ensemble members (MLPs and PC-Net)
PCNET_EPOCHS = pick(4000, 1500)
MLP_EPOCHS = pick(3000, 300)
ET_TREES = pick(500, 100)
GP_RESTARTS = pick(2, 0)

# ---- step 07: deployed models (all 500 designs) ------------------------------------------------------
FINAL_MEMBERS = pick(5, 2)
GEN_ITERS = pick(2500, 500)

# ---- step 08: kriging reference (diagnostic only) ----------------------------------------------------
KRIG_STEP_GHZ = pick(0.01, 0.1)
KRIG_RESTARTS = pick(1, 0)

# ---- step 09: inverse benchmark ----------------------------------------------------------------------
INV_FOLDS = pick([0, 1, 2, 3, 4], [0])
INV_SPECS_PER_FOLD = pick(20, 4)
INV_GRID_GHZ = pick([round(4.2 + 0.2 * i, 2) for i in range(14)], [4.6, 5.4, 6.2])
INV_NPROP = pick(128, 32)                          # candidates per query
INV_STEPS = pick(100, 20)                          # refinement steps
INV_GEN_ITERS = pick(1500, 400)
INV_DIRECT_EPOCHS = pick(3000, 300)
INV_CVAE_EPOCHS = pick(4000, 400)
INV_CURVE_STEPS = pick((0, 10, 30, 100), (0, 10, 20))

# ---- step 11: design grid of Fig. 5(b) ---------------------------------------------------------------
GRID_TARGETS_GHZ = pick([round(4.1 + 0.1 * i, 2) for i in range(29)], [4.3, 4.9, 5.5, 6.1, 6.7])


def describe():
    return (f"mode={MODE} | data={DATA_DIR} | workers={WORKERS} | fit starts={len(FIT_TAU_STARTS) + FIT_RANDOM_STARTS} | "
            f"members={CV_MEMBERS} | PC-Net epochs={PCNET_EPOCHS} | inverse folds={INV_FOLDS}")
