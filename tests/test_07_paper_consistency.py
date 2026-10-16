"""Paper consistency: every table and every number quoted in paper/Paper.tex comes out of the pipeline (step 14)."""
import json
import os

import pandas as pd
import pytest

from conftest import REF, latex_table_rows, run_step

CHECKS = json.load(open(os.path.join(REF, 'tables', 'paper_text_checks.json')))


@pytest.mark.parametrize('key', sorted(CHECKS))
def test_number_in_paper_text(key, paper_text):
    assert ' '.join(CHECKS[key]['text'].split()) in paper_text, CHECKS[key]['text']


def test_table3_in_paper_matches_results():
    tex = open(os.path.join(REF, '..', '..', 'paper', 'Paper.tex'), encoding='utf-8').read()
    gen = pd.read_csv(os.path.join(REF, 'tables', 'table3_inverse_benchmark.csv'), dtype=str).values.tolist()
    assert gen == latex_table_rows(tex, 'tab:inv')


def test_step14_regenerates_the_tables(tmp_path):
    """Step 14 run from the stored intermediate results gives the same tables and text numbers."""
    inputs = ['audit.json', 'interp_benchmark_summary.csv', 'fit_quality.json', 'processed.npz', 'design_descriptors.csv',
              'circuit_fits.pkl', 'fwd_cv_mean.csv', 'fwd_cv_std.csv', 'calibration.json', 'inv_eval_rows_0_1_2.csv',
              'inv_eval_rows_3_4.csv', 'support.json', 'catalog_coverage.csv', 'physics_check.json', 'parameter_sensitivity.csv',
              'final_designs_top3.csv'] + [f'pcnet_fold{k}.pt' for k in range(5)]
    run_step('14_tables.py', tmp_path, inputs)
    for t in ('table2_forward_models.csv', 'table3_inverse_benchmark.csv', 'table4_final_designs.csv'):
        assert open(tmp_path / 'tables' / t).read() == open(os.path.join(REF, 'tables', t)).read(), t
    assert json.load(open(tmp_path / 'tables' / 'paper_text_checks.json')) == CHECKS
