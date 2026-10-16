"""Figures: figures/make_figures.py redraws every paper figure from the exported figure data."""
import os
import subprocess
import sys

from conftest import REF, ROOT

NAMES = ['fig_framework', 'fig_extraction', 'fig_forward', 'fig_inverse']


def test_figure_data_is_complete():
    need = ['fig_extraction.npz', 'fig_forward.npz', 'fig_inverse_amortisation.csv', 'fig_inverse_designs.npz',
            'fig_inverse_hfss_matched.csv', 'fig_inverse_icp_grid.csv', 'fig_inverse_meta.json']
    assert all(os.path.exists(os.path.join(REF, 'figure_data', n)) for n in need)


def test_all_figures_are_drawn(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'figures', 'make_figures.py'), '--out', str(tmp_path), '--formats', 'pdf', 'png'],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    for n in NAMES:
        for ext in ('pdf', 'png'):
            p = tmp_path / f'{n}.{ext}'
            assert p.exists() and p.stat().st_size > 10_000, p


def test_paper_uses_every_figure():
    tex = open(os.path.join(ROOT, 'paper', 'Paper.tex'), encoding='utf-8').read()
    for n in NAMES:
        assert f'figures/{n}.pdf' in tex and os.path.exists(os.path.join(ROOT, 'paper', 'figures', f'{n}.pdf'))
