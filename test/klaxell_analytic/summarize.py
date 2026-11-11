#!/usr/bin/env python3
"""Collect result.json of every case into a table (stdout and summary.md).

With --mutants, rerun every comparison against perturbed reference constants
(no solver runs) and list which cases detect each perturbation.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = sorted(glob.glob(os.path.join(HERE, "t?_*"))) + sorted(glob.glob(os.path.join(HERE, "c?_*")))
MUTANTS = ("cmu", "cb", "lambda")


def read(case, name="result.json"):
    path = os.path.join(case, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def worst(res):
    """Largest error / tolerance over the checks of a case."""
    return max((c["error"] / c["tolerance"] for c in res["checks"]), default=0.0)


lines = ["| Case | Checks | Passed | Largest error / tolerance | Status |", "|---|---|---|---|---|"]
all_ok = True
for case in CASES:
    res = read(case)
    name = os.path.basename(case)
    if res is None:
        lines.append("| `%s` | - | - | - | not run |" % name)
        all_ok = False
        continue
    n = len(res["checks"])
    npass = sum(c["pass"] for c in res["checks"])
    all_ok = all_ok and res["pass"]
    lines.append("| `%s` | %d | %d | %.2f | %s |" % (name, n, npass, worst(res), "pass" if res["pass"] else "FAIL"))

if "--mutants" in sys.argv:
    for mutant in MUTANTS:
        env = dict(os.environ, KLAXELL_MUTANT=mutant)
        for case in CASES:
            subprocess.run([sys.executable, os.path.join(case, "compare.py")], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    lines += ["", "| Case | " + " | ".join("mutant `%s`" % m for m in MUTANTS) + " |", "|---|" + "---|" * len(MUTANTS)]
    for case in CASES:
        cells = []
        for mutant in MUTANTS:
            res = read(case, "result_mutant_%s.json" % mutant)
            if res is None:
                cells.append("error")
            else:
                nfail = sum(not c["pass"] for c in res["checks"])
                cells.append("%d of %d fail" % (nfail, len(res["checks"])))
        lines.append("| `%s` | %s |" % (os.path.basename(case), " | ".join(cells)))

text = "\n".join(lines) + "\n"
print(text)
with open(os.path.join(HERE, "summary.md"), "w") as fh:
    fh.write(text)
sys.exit(0 if all_ok else 1)
