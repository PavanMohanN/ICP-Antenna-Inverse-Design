"""End to end: the whole pipeline from the 500 raw HFSS files, in quick mode (opt-in, about 15 min).

    python run.py test --pipeline
The quick mode uses small budgets, so its numbers differ from the paper; this test checks that every step runs,
writes its outputs and produces a working designer model.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT

pytestmark = pytest.mark.pipeline
NAME = 'pytest_quick'


@pytest.fixture(scope='module')
def run_dir():
    d = os.path.join(ROOT, 'runs', NAME)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'run.py'), 'reproduce', '--quick', '--name', NAME],
                       capture_output=True, text=True, timeout=3 * 3600)
    assert r.returncode == 0, r.stdout[-5000:] + r.stderr[-5000:]
    return d


def test_every_step_wrote_its_outputs(run_dir):
    need = ['raw.npz', 'audit.json', 'circuit_fits.pkl', 'processed.npz', 'design_descriptors.csv', 'fwd_cv_mean.csv',
            'calibration.json', 'pcnet_final.pt', 'generator_final.pt', 'deploy/model/weights.npz', 'gp_oracle.pkl',
            'final_designs_top3.csv', 'final_design_grid.csv', 'physics_check.json', 'parameter_sensitivity.csv',
            'tables/paper_numbers.json', 'tables/table4_final_designs.tex', 'figure_data/fig_forward.npz',
            'csv/design_table.csv', 'figures/fig_inverse.pdf'] + [f'pcnet_fold{k}.pt' for k in range(5)]
    missing = [n for n in need if not os.path.exists(os.path.join(run_dir, n))]
    assert not missing, missing


def test_quick_model_designs_antennas(run_dir):
    sys.path.insert(0, os.path.join(ROOT, 'designer'))
    from icp_engine import ICPModel
    m = ICPModel(os.path.join(run_dir, 'deploy', 'model'))
    res = m.design(5.5, K=2)
    X = m.from_unit(res['U'])
    assert np.isfinite(res['ev']['S11_at_f']).all() and (X >= m.LB).all() and (X <= m.UB).all()
    assert json.load(open(os.path.join(run_dir, 'deploy', 'model', 'config.json')))['pipeline_mode'] == 'quick'


def test_compare_runs(run_dir):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'run.py'), 'compare', NAME], capture_output=True, text=True)
    assert r.returncode == 0 and 'quantities identical' in r.stdout
