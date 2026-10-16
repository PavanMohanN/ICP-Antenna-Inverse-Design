"""Shared fixtures of the test suite.

Run everything with   python run.py test          (or: python -m pytest tests -v)
Opt-in end-to-end run python run.py test --pipeline   (adds the whole pipeline in quick mode, ~15 min)
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REF = os.path.join(ROOT, 'runs', 'paper')            # reference outputs of the paper run
DESIGNER = os.path.join(ROOT, 'designer')
STEPS = os.path.join(ROOT, 'pipeline', 'steps')
DATA = os.path.join(ROOT, 'data', 'hfss')
PAPER = os.path.join(ROOT, 'paper', 'Paper.tex')
for p in (DESIGNER, os.path.join(ROOT, 'pipeline', 'lib')):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_addoption(parser):
    parser.addoption('--run-pipeline', action='store_true', default=False,
                     help='also run the whole pipeline in quick mode (about 15 min)')


def pytest_configure(config):
    config.addinivalue_line('markers', 'pipeline: runs the whole pipeline in quick mode (opt-in with --run-pipeline)')
    config.addinivalue_line('markers', 'torch: needs PyTorch')


def pytest_collection_modifyitems(config, items):
    skip_pipe = pytest.mark.skip(reason='opt-in: python run.py test --pipeline')
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    for it in items:
        if 'pipeline' in it.keywords and not config.getoption('--run-pipeline'):
            it.add_marker(skip_pipe)
        if 'torch' in it.keywords and not has_torch:
            it.add_marker(pytest.mark.skip(reason='PyTorch not installed'))


@pytest.fixture(scope='session', autouse=True)
def reference_ready():
    """runs/paper/processed.npz is rebuilt from the stored circuit fits (step 03) if it is missing."""
    sys.path.insert(0, ROOT)
    from run import ensure_reference
    ensure_reference(verbose=False)


@pytest.fixture(scope='session')
def model():
    from icp_engine import ICPModel
    return ICPModel()


@pytest.fixture(scope='session')
def processed():
    return np.load(os.path.join(REF, 'processed.npz'), allow_pickle=True)


@pytest.fixture(scope='session')
def paper_text():
    return ' '.join(open(PAPER, encoding='utf-8').read().split())


def run_step(script, workdir, inputs, args=(), env=None, timeout=1800):
    """Copy `inputs` from the reference run into `workdir`, run pipeline step `script` there and return its output."""
    os.makedirs(workdir, exist_ok=True)
    for name in inputs:
        src = os.path.join(REF, name)
        dst = os.path.join(workdir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    e = dict(os.environ, ICP_MODE='full', **(env or {}))
    r = subprocess.run([sys.executable, os.path.join(STEPS, script), *args], cwd=workdir, env=e,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f'{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}'
    return r.stdout


def run_tool(script, args, cwd, timeout=900):
    r = subprocess.run([sys.executable, os.path.join(DESIGNER, script), *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f'{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}'
    return r.stdout


def latex_table_rows(tex, label):
    """Data rows (between \\midrule and \\bottomrule) of the table with \\label{label}, as lists of cells."""
    i = tex.index('\\label{' + label + '}')
    body = tex[tex.index('\\midrule', i) + len('\\midrule'):tex.index('\\bottomrule', i)]
    rows = []
    for ln in body.split('\\\\'):
        ln = ln.replace('\\midrule', '').strip()
        if ln:
            rows.append([c.strip() for c in ln.split('&')])
    return rows
