"""Makes the shared library (pipeline/lib) and the designer engine (designer/) importable from every step.

Every step reads and writes files in the CURRENT folder, which is the run folder (runs/<name>/).
Run the steps through run.py, which creates the run folder and calls them in order:
    python run.py reproduce --quick
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', 'lib'), os.path.join(_HERE, '..', '..', 'designer')):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
