#!/usr/bin/env python3
"""
run.py - one entry point for everything in this repository.

    python run.py check                              is my installation complete?
    python run.py demo                               two-minute tour: design, predict, RF circuit, figures
    python run.py test                               run the test suite (about 2 min)
    python run.py design --band 5 6 --rl 10 --circuit
                                                     S-parameter target -> antenna dimensions (+ RF circuit)
    python run.py predict --dims 1.78 11.93 18.61 0.34 4.70 15.35 1.36 2.83 5.07
                                                     antenna dimensions -> predicted S11 (+ RF circuit)
    python run.py examples                           regenerate the designer examples in designer/examples
    python run.py figures                            redraw every paper figure from runs/paper/figure_data
    python run.py reproduce --quick                  whole pipeline from the raw HFSS files, small budgets (~15 min)
    python run.py reproduce                          whole pipeline with the paper settings (~2 h on 2 cores)
    python run.py compare quick                      compare a run with the paper run, number by number
    python run.py steps                              list the pipeline steps
    python run.py paper                              compile paper/Paper.pdf (needs a LaTeX installation)

Every command prints what it does and where it writes. `python run.py <command> -h` shows its options.
Only the Python standard library is needed to start this script; each command checks its own requirements.
"""
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DESIGNER = os.path.join(ROOT, 'designer')
STEPS_DIR = os.path.join(ROOT, 'pipeline', 'steps')
RUNS = os.path.join(ROOT, 'runs')
PY = sys.executable


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def say(msg=''):
    print(msg, flush=True)


def banner(title):
    say('\n' + '=' * 78)
    say(title)
    say('=' * 78)


def call(cmd, cwd=ROOT, env=None):
    """Run a command in the foreground; return its exit code."""
    say('$ ' + ' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c for c in cmd))
    return subprocess.call(cmd, cwd=cwd, env=env)


def need(*modules):
    missing = []
    for m in modules:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    if missing:
        pip = {'sklearn': 'scikit-learn'}
        raise SystemExit('Missing Python packages: ' + ', '.join(missing) + '\nInstall them with:\n    pip install '
                         + ' '.join(pip.get(m, m) for m in missing) + '\n(or: pip install -r requirements.txt)')


def ensure_reference(verbose=True):
    """runs/paper/processed.npz (12 MB) is not stored in the repository: it is rebuilt from the stored equivalent-circuit
    fits by pipeline step 03 in a few seconds, bit for bit, the first time it is needed."""
    ref = os.path.join(RUNS, 'paper')
    target = os.path.join(ref, 'processed.npz')
    if os.path.exists(target):
        return True
    import tempfile
    if verbose:
        say('First use: rebuilding runs/paper/processed.npz from the stored circuit fits (pipeline step 03, a few seconds) ...')
    with tempfile.TemporaryDirectory() as tmp:
        for f in ('raw.npz', 'circuit_fits.pkl'):
            shutil.copy2(os.path.join(ref, f), os.path.join(tmp, f))
        r = subprocess.run([PY, os.path.join(STEPS_DIR, '03_process.py')], cwd=tmp, capture_output=True, text=True,
                           env=dict(os.environ, ICP_MODE='full'))
        if r.returncode:
            say(r.stdout[-2000:] + r.stderr[-2000:])
            raise SystemExit('Could not rebuild runs/paper/processed.npz (needs numpy, scipy and pandas).')
        shutil.copy2(os.path.join(tmp, 'processed.npz'), target)
    if verbose:
        say('  done: runs/paper/processed.npz')
    return True


def fmt_time(s):
    return f'{s:.0f} s' if s < 90 else (f'{s / 60:.1f} min' if s < 5400 else f'{s / 3600:.2f} h')


# ----------------------------------------------------------------------------------------------------------------
# check
# ----------------------------------------------------------------------------------------------------------------
def cmd_check(a):
    banner('Installation check')
    ok = True
    say(f'Python {sys.version.split()[0]}  ({PY})')
    if sys.version_info < (3, 10):
        say('  PROBLEM  Python 3.10 or newer is required')
        ok = False
    groups = [('designer (design, predict, examples)', ['numpy', 'scipy', 'matplotlib']),
              ('figures', ['numpy', 'matplotlib']),
              ('pipeline (reproduce, compare)', ['numpy', 'scipy', 'pandas', 'sklearn', 'torch', 'matplotlib']),
              ('tests', ['pytest'])]
    seen = {}
    for m in sorted({m for _, ms in groups for m in ms}):
        try:
            mod = importlib.import_module(m)
            seen[m] = getattr(mod, '__version__', '?')
        except ImportError:
            seen[m] = None
    say('\nPackages')
    for m, v in seen.items():
        say(f"  {'ok     ' if v else 'MISSING'}  {m:12s} {v or ''}")
    say('\nCommands')
    ready = {}
    for name, ms in groups:
        miss = [m for m in ms if seen[m] is None]
        ready[name] = not miss
        say(f"  {'ready  ' if not miss else 'blocked'}  {name}" + (f"   (install: {' '.join('scikit-learn' if m == 'sklearn' else m for m in miss)})" if miss else ''))
    say('\nFiles')
    n = len([f for f in os.listdir(os.path.join(ROOT, 'data', 'hfss')) if f.endswith('.s1p')]) if os.path.isdir(os.path.join(ROOT, 'data', 'hfss')) else 0
    items = [(n == 500, f'data/hfss: {n} HFSS Touchstone files (expected 500)'),
             (os.path.exists(os.path.join(DESIGNER, 'model', 'weights.npz')), 'designer/model: deployed model'),
             (os.path.exists(os.path.join(RUNS, 'paper', 'tables', 'paper_numbers.json')), 'runs/paper: reference outputs of the paper run'),
             (os.path.exists(os.path.join(RUNS, 'paper', 'figure_data', 'fig_forward.npz')), 'runs/paper/figure_data: data of the paper figures')]
    if not os.path.exists(os.path.join(RUNS, 'paper', 'processed.npz')):
        say('  note     runs/paper/processed.npz is rebuilt automatically (a few seconds) the first time the tests need it')
    for good, txt in items:
        say(f"  {'ok     ' if good else 'MISSING'}  {txt}")
        ok &= good
    say(f'\nCPU cores: {os.cpu_count()}')
    if ok and all(ready.values()):
        say('\nEverything is in place. Next: python run.py test')
    elif ok and ready[groups[0][0]]:
        say('\nThe designer and the figures are ready. For the tests and the pipeline: pip install -r requirements.txt')
    else:
        say('\nSome parts are missing; see above.')
    return 0 if ok and ready[groups[0][0]] else 1


# ----------------------------------------------------------------------------------------------------------------
# designer
# ----------------------------------------------------------------------------------------------------------------
def cmd_design(a, rest):
    need('numpy')
    return call([PY, os.path.join(DESIGNER, 'design_antenna.py')] + rest, cwd=os.getcwd())


def cmd_predict(a, rest):
    need('numpy')
    return call([PY, os.path.join(DESIGNER, 'predict_s11.py')] + rest, cwd=os.getcwd())


EXAMPLES = [
    ('4.4-5.4GHz_RL10', ['--band', '4.4', '5.4', '--rl', '10', '--top', '3']),
    ('5.50GHz_max_band', ['--freq', '5.5', '--top', '3']),
    ('5-6GHz_RL10', ['--band', '5', '6', '--rl', '10', '--top', '3', '--circuit']),
    ('5-6GHz_RL15', ['--band', '5', '6', '--rl', '15', '--top', '3', '--circuit']),
]


def cmd_examples(a):
    need('numpy', 'scipy', 'matplotlib')
    banner('Designer examples -> designer/examples/')
    ex = os.path.join(DESIGNER, 'examples')
    for name, args in EXAMPLES:
        out = os.path.join(ex, name)
        shutil.rmtree(out, ignore_errors=True)
        rc = call([PY, os.path.join(DESIGNER, 'design_antenna.py')] + args + ['--out', out])
        if rc:
            return rc
    return 0


def cmd_demo(a):
    """A two-minute tour: inverse design, forward prediction, RF circuit and figures, into demo_output/."""
    need('numpy', 'scipy', 'matplotlib')
    out = os.path.join(ROOT, 'demo_output')
    shutil.rmtree(out, ignore_errors=True)
    tool = lambda name: os.path.join(DESIGNER, name)
    steps = [
        ('1/4  Inverse design: |S11| <= -10 dB over 5-6 GHz, three designs, with RF circuit',
         [PY, tool('design_antenna.py'), '--band', '5', '6', '--rl', '10', '--top', '3', '--circuit', '--out', os.path.join(out, '1_design_5-6GHz_RL10')]),
        ('2/4  Inverse design: |S11| <= -15 dB over 5-6 GHz (harder target)',
         [PY, tool('design_antenna.py'), '--band', '5', '6', '--rl', '15', '--top', '3', '--out', os.path.join(out, '2_design_5-6GHz_RL15')]),
        ('3/4  Forward prediction of the first design of step 1, and of two simulated designs with their HFSS curves',
         None),
        ('4/4  The four paper figures',
         [PY, os.path.join(ROOT, 'figures', 'make_figures.py'), '--out', os.path.join(out, '4_figures'), '--formats', 'png']),
    ]
    for title, cmd in steps:
        banner(title)
        if cmd is None:
            src = os.path.join(out, '1_design_5-6GHz_RL10', 'hfss_samples.csv')
            rc = call([PY, tool('predict_s11.py'), '--csv', src, '--rows', '0', '--out', os.path.join(out, '3_predict')])
            rc = rc or call([PY, tool('predict_s11.py'), '--csv', os.path.join(ROOT, 'data', 'hfss', 'samples.csv'), '--rows', '291', '72',
                             '--out', os.path.join(out, '3_predict_simulated')])
        else:
            rc = call(cmd)
        if rc:
            return rc
    banner('Demo finished. Open the folder demo_output/:')
    for d in sorted(os.listdir(out)):
        say(f'  demo_output/{d}/  ' + ', '.join(sorted(os.listdir(os.path.join(out, d)))[:6]) + ' ...')
    say('Start with demo_output/1_design_5-6GHz_RL10/predicted_S11.png and designs.csv.')
    return 0


# ----------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------
def cmd_figures(a):
    need('numpy', 'matplotlib')
    data = os.path.join(RUNS, a.run, 'figure_data')
    if not os.path.isdir(data):
        raise SystemExit(f'{os.path.relpath(data, ROOT)} not found. Run the pipeline first (python run.py reproduce ...).')
    out = a.out or (os.path.join(ROOT, 'paper', 'figures') if a.to_paper else os.path.join(ROOT, 'figures', 'output', a.run))
    formats = ['pdf'] if a.to_paper and not a.formats else (a.formats or ['pdf', 'png'])
    banner(f'Paper figures from runs/{a.run}/figure_data -> {os.path.relpath(out, ROOT)}')
    cmd = [PY, os.path.join(ROOT, 'figures', 'make_figures.py'), '--data', data, '--out', out, '--formats'] + formats
    if a.only:
        cmd += ['--only'] + a.only
    return call(cmd)


# ----------------------------------------------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------------------------------------------
# (id, title, outputs, minutes on 2 cores in full / quick mode)
STEPS = [
    ('01', 'Data audit of the 500 HFSS files', 'raw.npz, audit.json', (0.3, 0.3)),
    ('02', 'Passive equivalent circuit of every design (K = 2..5 resonators)', 'circuit_fits.pkl', (45, 3)),
    ('03', 'Physically consistent 1 MHz responses, descriptors, super-resolution test', 'processed.npz, design_descriptors.csv', (0.1, 0.1)),
    ('04a', 'Forward CV: Extra Trees, kriging, MLP baselines (5 folds)', 'fwd_oof_ET_GP_MLPdB_MLPri_*.pkl', (7, 1.5)),
    ('04b', 'Forward CV: PC-Net ensembles (5 folds, parallel over --jobs)', 'fwd_oof_PCraw_PC_*.pkl, pcnet_fold*.pt', (33, 6)),
    ('05', 'Forward metrics (Table II)', 'fwd_cv_mean.csv, fwd_cv_std.csv', (0.1, 0.1)),
    ('06', 'Calibration of the ensemble spread (kappa)', 'calibration.json', (0.2, 0.2)),
    ('07a', 'Deployed PC-Net ensemble on all 500 designs', 'pcnet_oracle.pt, pcnet_final.pt', (14, 2.5)),
    ('07b', 'Constrained inverse generator', 'generator_final.pt', (1.5, 0.3)),
    ('07c', 'Export the NumPy designer model', 'deploy/model/', (0.3, 0.3)),
    ('08', 'Kriging reference (diagnostic)', 'gp_oracle.pkl', (4, 0.3)),
    ('09', 'Inverse benchmark (Table III, Fig. 5a; parallel over --jobs)', 'inv_eval_rows_*.csv', (12, 0.5)),
    ('10', 'Final designs of Table IV', 'final_designs_top3.csv, design_runs/', (0.7, 0.6)),
    ('11', 'Design grid 4.1-6.9 GHz (Fig. 5b)', 'final_design_grid.csv, catalog_coverage.csv', (3, 0.5)),
    ('12', 'Physics check f*L = const', 'physics_check.json', (0.1, 0.1)),
    ('13', 'Parameter sensitivity', 'parameter_sensitivity.csv', (0.2, 0.2)),
    ('14', 'Tables and every number of the paper text', 'tables/', (0.2, 0.2)),
    ('15', 'Figure data', 'figure_data/', (0.1, 0.1)),
    ('16', 'CSV exports of the processed data', 'csv/', (0.2, 0.2)),
    ('17', 'Paper figures', 'figures/', (0.2, 0.2)),
]
STEP_IDS = [s[0] for s in STEPS]


def cmd_steps(a):
    banner('Pipeline steps (python run.py reproduce runs them in this order)')
    say(f"{'step':5s} {'full':>7s} {'quick':>7s}  description -> outputs")
    for sid, title, outs, (mf, mq) in STEPS:
        say(f"{sid:5s} {('<1' if mf < 1 else f'{mf:.0f}'):>5s} m {('<1' if mq < 1 else f'{mq:.0f}'):>5s} m  {title} -> {outs}")
    say(f"      {sum(s[3][0] for s in STEPS) / 60:5.1f} h {sum(s[3][1] for s in STEPS):5.0f} m  total on a 2-core laptop")
    say('\nScripts: pipeline/steps/<id>_*.py. Shared code: pipeline/lib/. Settings of both modes: pipeline/lib/settings.py')
    return 0


def _stream(proc, prefix, logf):
    enc = sys.stdout.encoding or 'utf-8'
    for line in iter(proc.stdout.readline, ''):
        logf.write(line)
        logf.flush()
        sys.stdout.write((prefix + line).encode(enc, 'replace').decode(enc, 'replace'))   # safe on any console
        sys.stdout.flush()


def run_parallel(cmds, run_dir, env, log_name):
    """cmds: list of (prefix, argv, extra_env). Streams every process to the console and to logs/."""
    procs, threads, logs = [], [], []
    for i, (prefix, argv, extra) in enumerate(cmds):
        e = dict(env, **extra)
        lp = os.path.join(run_dir, 'logs', f'{log_name}' + (f'_part{i + 1}' if len(cmds) > 1 else '') + '.log')
        lf = open(lp, 'w', encoding='utf-8')
        logs.append(lf)
        p = subprocess.Popen(argv, cwd=run_dir, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding='utf-8', errors='replace')
        procs.append(p)
        t = threading.Thread(target=_stream, args=(p, prefix, lf), daemon=True)
        t.start()
        threads.append(t)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    for lf in logs:
        lf.close()
    return max(codes)


def split(items, n):
    n = max(1, min(n, len(items)))
    return [items[i::n] for i in range(n)]


def step_commands(sid, jobs, settings):
    s = lambda name: os.path.join(STEPS_DIR, name)
    if sid == '01':
        return [('', [PY, s('01_audit.py')], {})]
    if sid == '02':
        return [('', [PY, s('02_fit_circuits.py')], {'ICP_WORKERS': str(jobs)})]
    if sid == '03':
        return [('', [PY, s('03_process.py')], {})]
    if sid == '04a':
        return [('', [PY, s('04_forward_cv.py'), 'ET', 'GP', 'MLPdB', 'MLPri'], {})]
    if sid == '04b':
        groups = split([0, 1, 2, 3, 4], jobs)
        return [(f'[folds {",".join(map(str, g))}] ' if len(groups) > 1 else '', [PY, s('04_forward_cv.py'), 'PCraw', 'PC'],
                 {'FOLDS': ','.join(map(str, g))}) for g in groups]
    if sid in ('05', '06'):
        return [('', [PY, s({'05': '05_forward_metrics.py', '06': '06_calibrate.py'}[sid])], {})]
    if sid == '07a':
        return [('', [PY, s('07_train_final.py'), 'oracle'], {})]
    if sid == '07b':
        return [('', [PY, s('07_train_final.py'), 'generator'], {})]
    if sid == '07c':
        return [('', [PY, s('07_train_final.py'), 'export', os.path.join('deploy', 'model')], {})]
    if sid == '08':
        return [('', [PY, s('08_kriging_reference.py')], {})]
    if sid == '09':
        groups = split(list(settings.INV_FOLDS), jobs)
        return [(f'[folds {",".join(map(str, g))}] ' if len(groups) > 1 else '', [PY, s('09_inverse_benchmark.py')] + [str(k) for k in g], {})
                for g in groups]
    names = {'10': '10_final_designs.py', '11': '11_design_grid.py', '12': '12_physics_check.py', '13': '13_sensitivity.py',
             '14': '14_tables.py', '15': '15_export_figure_data.py', '16': '16_export_csv.py'}
    if sid in names:
        return [('', [PY, s(names[sid])], {})]
    if sid == '17':
        return [('', [PY, os.path.join(ROOT, 'figures', 'make_figures.py'), '--data', 'figure_data', '--out', 'figures',
                      '--formats', 'pdf', 'png'], {})]
    raise ValueError(sid)


def cmd_reproduce(a):
    need('numpy', 'scipy', 'pandas', 'sklearn', 'torch', 'matplotlib')
    mode = 'quick' if a.quick else 'full'
    name = a.name or mode
    if name == 'paper' and not a.force:
        raise SystemExit("runs/paper holds the reference outputs of the paper; choose another --name (or add --force).")
    run_dir = os.path.join(RUNS, name)
    if name == 'paper':
        ensure_reference()
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    done_dir = os.path.join(run_dir, '.done')
    os.makedirs(done_dir, exist_ok=True)
    env = dict(os.environ, ICP_MODE=mode, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    if a.data:
        env['ICP_DATA'] = os.path.abspath(a.data)
    os.environ.update({k: env[k] for k in ('ICP_MODE',) + (('ICP_DATA',) if a.data else ())})
    sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'lib'))
    import settings  # noqa: E402  (reads ICP_MODE)
    jobs = a.jobs or max(1, min(os.cpu_count() or 1, 5))

    meta_p = os.path.join(run_dir, 'run_info.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if meta.get('mode') and meta['mode'] != mode:
        raise SystemExit(f"runs/{name} was started in {meta['mode']} mode; use another --name for a {mode} run.")
    meta.update(mode=mode, data=settings.DATA_DIR, python=sys.version.split()[0], started=meta.get('started', time.strftime('%Y-%m-%d %H:%M:%S')))
    json.dump(meta, open(meta_p, 'w'), indent=1)

    todo = list(STEP_IDS)
    if a.only:
        todo = [s for s in STEP_IDS if s in a.only or s[:2] in a.only]
    elif a.from_step:
        start = next((i for i, s in enumerate(STEP_IDS) if s == a.from_step or s[:2] == a.from_step), None)
        if start is None:
            raise SystemExit(f'unknown step {a.from_step}; see python run.py steps')
        todo = STEP_IDS[start:]
    if a.from_step or a.only:
        for s in todo:                                          # requested steps are re-run
            p = os.path.join(done_dir, s)
            if os.path.exists(p):
                os.remove(p)

    total_est = sum(st[3][0 if mode == 'full' else 1] for st in STEPS if st[0] in todo)
    banner(f'Reproducing the pipeline in {mode.upper()} mode -> runs/{name}/')
    say(settings.describe())
    say(f'parallel jobs: {jobs} | steps: {", ".join(todo)} | estimated {fmt_time(60 * total_est)} on 2 cores (less on more cores)')
    say('Completed steps are skipped, so an interrupted run continues where it stopped (rerun the same command).')
    t_all = time.time()
    for sid, title, outs, mins in STEPS:
        if sid not in todo:
            continue
        if os.path.exists(os.path.join(done_dir, sid)):
            say(f'\n[{sid}] {title}: already done, skipped')
            continue
        say(f'\n[{sid}] {title}  (about {fmt_time(60 * mins[0 if mode == "full" else 1])} on 2 cores)  -> {outs}')
        t0 = time.time()
        rc = run_parallel(step_commands(sid, jobs, settings), run_dir, env, f'{sid}')
        if rc:
            say(f'\nStep {sid} FAILED (exit code {rc}). Log: runs/{name}/logs/{sid}*.log')
            say(f'Fix the problem and rerun the same command; it continues from step {sid}.')
            return rc
        if sid == '07a':
            shutil.copy2(os.path.join(run_dir, 'pcnet_oracle.pt'), os.path.join(run_dir, 'pcnet_final.pt'))
        open(os.path.join(done_dir, sid), 'w').write(f'{time.time() - t0:.1f}\n')
        say(f'[{sid}] done in {fmt_time(time.time() - t0)}  (total {fmt_time(time.time() - t_all)})')
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(meta, open(meta_p, 'w'), indent=1)
    banner(f'Finished in {fmt_time(time.time() - t_all)}. Outputs in runs/{name}/')
    say(f'  tables/      paper tables (LaTeX + CSV), paper_numbers.json, paper_text_checks.json')
    say(f'  figures/     the four paper figures (PDF + PNG)')
    say(f'  deploy/model the newly trained designer model; use it with: python run.py design --freq 5.5 --model runs/{name}/deploy/model')
    say(f'Next: python run.py compare {name}')
    return 0


# ----------------------------------------------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------------------------------------------
def _load_numbers(run):
    p = os.path.join(RUNS, run, 'tables', 'paper_numbers.json')
    if not os.path.exists(p):
        raise SystemExit(f'runs/{run}/tables/paper_numbers.json not found (run the pipeline up to step 14 first).')
    return json.load(open(p)), json.load(open(os.path.join(RUNS, run, 'tables', 'paper_text_checks.json')))


def cmd_compare(a):
    new, new_chk = _load_numbers(a.run)
    ref, ref_chk = _load_numbers(a.ref)
    banner(f'runs/{a.run} versus runs/{a.ref}')
    rows = []
    fm, fs = 'forward_cv_mean', 'forward_cv_std'
    for m, short in [('ExtraTrees (dB)', 'Extra Trees'), ('Kriging/GP (dB)', 'Kriging'), ('MLP ensemble (dB)', 'MLP dB'),
                     ('MLP ensemble (Re/Im)', 'MLP Re/Im'), ('PC-Net ensemble (proposed)', 'PC-Net')]:
        for key, lab, d in [('dB MAE', 'MAE (dB)', 2), ('f_res MAE (MHz)', 'f_r error (MHz)', 0), ('F1', 'F1', 2),
                            ('screen precision@5 (%)', 'P@5 (%)', 0), ('passivity violations (%)', '|S11|>1 (%)', 1)]:
            if m in new.get(fm, {}) and m in ref.get(fm, {}):
                rows.append((f'Table II  {short:11s} {lab}', ref[fm][m][key], new[fm][m][key], d))
    for m, short in [('Catalogue retrieval (training HFSS designs)', 'Catalogue'), ('Tandem network (unconstrained)', 'Tandem'),
                     ('Constrained surrogate optimisation (random starts)', 'Constr. opt.'),
                     ('Ablation: ICP generator only (no refinement)', 'Generator only'),
                     ('Ablation: no robust/support constraints', 'ICP w/o constr.'), ('Proposed ICP inverse', 'ICP')]:
        for key, lab, d in [('spec_success', 'Spec. (%)', 0), ('calib_success', 'Calib. (%)', 0), ('optimism_median', 'Opt. (dB)', 1)]:
            if m in new.get('inverse', {}) and m in ref.get('inverse', {}):
                rows.append((f'Table III {short:15s} {lab}', ref['inverse'][m][key], new['inverse'][m][key], d))
    rows.append(('kappa', ref['calibration']['kappa'], new['calibration']['kappa'], 3))
    if 'physics_check' in new and 'physics_check' in ref:
        rows.append(('f*L of ICP designs (GHz mm)', ref['physics_check']['icp_fL_mean'], new['physics_check']['icp_fL_mean'], 1))
    say(f"{'quantity':45s} {a.ref:>10s} {a.run:>10s}")
    same = 0
    for lab, r, n, d in rows:
        rs, ns = f'{r:.{d}f}', f'{n:.{d}f}'
        same += rs == ns
        say(f"{lab:45s} {rs:>10s} {ns:>10s}" + ('' if rs == ns else '   *'))
    say(f'\n{same}/{len(rows)} quantities identical at the precision printed in the paper (* = differs).')
    paper = _paper_text()
    hit = [k for k, v in new_chk.items() if _norm(v['text']) in paper]
    det = [k for k, v in new_chk.items() if not v['machine_dependent']]
    say(f"Paper text: {len([k for k in hit if k in det])}/{len(det)} machine-independent sentences of Paper.tex are reproduced exactly by runs/{a.run}"
        + (" (a quick run is not expected to match)." if json.load(open(os.path.join(RUNS, a.run, 'run_info.json'))).get('mode') == 'quick' else '.'))
    for k in det:
        if k not in hit:
            say(f'  differs: {k}: "{new_chk[k]["text"]}"')
    return 0


def _norm(t):
    return ' '.join(t.split())


def _paper_text():
    return _norm(open(os.path.join(ROOT, 'paper', 'Paper.tex'), encoding='utf-8').read())


# ----------------------------------------------------------------------------------------------------------------
# tests and paper
# ----------------------------------------------------------------------------------------------------------------
def cmd_test(a, rest):
    need('pytest', 'numpy', 'scipy', 'pandas')
    ensure_reference()
    args = [PY, '-m', 'pytest', os.path.join(ROOT, 'tests'), '-v']
    if a.pipeline:
        args += ['--run-pipeline']
    banner('Test suite' + (' including the quick end-to-end pipeline (~15 min)' if a.pipeline else ' (about 2 min)'))
    return call(args + rest)


def cmd_paper(a):
    if shutil.which('pdflatex') is None or shutil.which('bibtex') is None:
        raise SystemExit('pdflatex/bibtex not found. Install TeX Live or MiKTeX (with the IEEEtran class), or upload paper/ to Overleaf.')
    banner('Compiling paper/Paper.tex')
    d = os.path.join(ROOT, 'paper')
    for cmd in (['pdflatex', '-interaction=nonstopmode', 'Paper.tex'], ['bibtex', 'Paper'],
                ['pdflatex', '-interaction=nonstopmode', 'Paper.tex'], ['pdflatex', '-interaction=nonstopmode', 'Paper.tex']):
        rc = subprocess.call(cmd, cwd=d, stdout=subprocess.DEVNULL)
        if rc and cmd[0] == 'bibtex':
            return rc
    for ext in ('aux', 'bbl', 'blg', 'log', 'out'):
        p = os.path.join(d, f'Paper.{ext}')
        if os.path.exists(p) and ext != 'log':
            os.remove(p)
    say('written paper/Paper.pdf' if os.path.exists(os.path.join(d, 'Paper.pdf')) else 'compilation failed; see paper/Paper.log')
    return 0


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', metavar='command')
    sub.add_parser('check', help='check Python packages and files')
    pt = sub.add_parser('test', help='run the test suite (extra arguments go to pytest)')
    pt.add_argument('--pipeline', action='store_true', help='also run the whole pipeline in quick mode inside the tests (~15 min)')
    sub.add_parser('design', help='S-parameter target -> antenna dimensions (arguments of designer/design_antenna.py)', add_help=False)
    sub.add_parser('predict', help='antenna dimensions -> predicted S11 (arguments of designer/predict_s11.py)', add_help=False)
    sub.add_parser('demo', help='two-minute tour: design, predict, RF circuit, figures -> demo_output/')
    sub.add_parser('examples', help='regenerate designer/examples')
    pf = sub.add_parser('figures', help='redraw the paper figures')
    pf.add_argument('--run', default='paper', help='run folder under runs/ whose figure_data to use (default: paper)')
    pf.add_argument('--out', help='output folder (default: figures/output/<run>)')
    pf.add_argument('--formats', nargs='*', help='e.g. pdf png svg (default: pdf png)')
    pf.add_argument('--only', nargs='*', choices=['framework', 'extraction', 'forward', 'inverse'])
    pf.add_argument('--to-paper', action='store_true', help='write PDFs straight into paper/figures (overwrites them)')
    pr = sub.add_parser('reproduce', help='run the whole pipeline from the raw HFSS files')
    pr.add_argument('--quick', action='store_true', help='small budgets, about 15 min (numbers differ from the paper)')
    pr.add_argument('--name', help='run folder name under runs/ (default: quick or full)')
    pr.add_argument('--jobs', type=int, help='parallel processes (default: number of CPU cores, at most 5)')
    pr.add_argument('--from', dest='from_step', help='re-run from this step on (e.g. 09)')
    pr.add_argument('--only', nargs='*', help='re-run only these steps (e.g. 14 15 17)')
    pr.add_argument('--data', help='folder with your own HFSS export (default: data/hfss)')
    pr.add_argument('--force', action='store_true', help=argparse.SUPPRESS)
    pc = sub.add_parser('compare', help='compare a run with the paper run')
    pc.add_argument('run', help='run folder name under runs/, e.g. quick or full')
    pc.add_argument('--ref', default='paper', help='reference run (default: paper)')
    sub.add_parser('steps', help='list the pipeline steps')
    sub.add_parser('paper', help='compile paper/Paper.pdf with pdflatex and bibtex')

    argv = sys.argv[1:]
    if argv and argv[0] in ('design', 'predict', 'test'):        # everything after the command is passed through
        rest = [x for x in argv[1:] if x != '--pipeline']
        a = argparse.Namespace(cmd=argv[0], pipeline='--pipeline' in argv[1:])
        return {'design': cmd_design, 'predict': cmd_predict, 'test': cmd_test}[argv[0]](a, rest)
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 0
    return {'check': cmd_check, 'demo': cmd_demo, 'examples': cmd_examples, 'figures': cmd_figures, 'reproduce': cmd_reproduce,
            'compare': cmd_compare, 'steps': cmd_steps, 'paper': cmd_paper}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main() or 0)
