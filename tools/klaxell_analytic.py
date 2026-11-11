#!/usr/bin/env python3
"""Shared helpers for the KLAxell analytical tests in test/klaxell_analytic.

Holds the model constants, the closed-form and ODE reference solutions, a
reader for the ASCII line-sampler output, and the result/figure writers that
every case script uses.
"""
import glob
import json
import os

import numpy as np

# Model constants (defaults in src/turbulence/RANS/KLAxell.H)
CMU = 0.556
CB_STABLE = 0.25
CB_UNSTABLE = 0.35
KAPPA = 0.41
LAMBDA = 30.0
GRAV = 9.81
T_REF = 300.0
RT_CRIT = -1.0
RT_MIN = -3.0

# Mutant switch used to prove that the checks can fail: the reference constants
# are perturbed, nothing in the solver changes.
#   KLAXELL_MUTANT=cmu     C_mu + 2 %
#   KLAXELL_MUTANT=cb      C_b (stable and unstable) + 5 %
#   KLAXELL_MUTANT=lambda  lambda + 5 %
MUTANT = os.environ.get("KLAXELL_MUTANT", "")
if MUTANT == "cmu":
    CMU *= 1.02
elif MUTANT == "cb":
    CB_STABLE *= 1.05
    CB_UNSTABLE *= 1.05
elif MUTANT == "lambda":
    LAMBDA *= 1.05
elif MUTANT:
    raise RuntimeError("unknown KLAXELL_MUTANT " + MUTANT)


# ----------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------
def lscale_neutral(z):
    """Neutral length scale l_s = lambda kappa z / (lambda + kappa z)."""
    z = np.asarray(z, dtype=float)
    return LAMBDA * KAPPA * z / (LAMBDA + KAPPA * z)


def brunt_vaisala_sq(dtdz, t_ref=T_REF):
    """N^2 = (g / T0) dT/dz."""
    return GRAV / t_ref * dtdz


def limit_rt(rt):
    """Limiter applied to the turbulent Richardson number below Rt = -1."""
    rt = np.asarray(rt, dtype=float)
    lim = rt - (rt - RT_CRIT) ** 2 / (rt + RT_MIN - 2.0 * RT_CRIT)
    return np.where(rt > RT_CRIT, rt, np.maximum(rt, lim))


def cmu_rt(rt):
    """Momentum stability function."""
    return (CMU + 0.108 * rt) / (1.0 + 0.308 * rt + 0.00837 * rt**2)


def cmu_prime_rt(rt):
    """Heat stability function used in the buoyancy term."""
    return CMU / (1.0 + 0.277 * rt)


def stable_length(k, n2, l_s):
    """Stable length scale and turbulent Richardson number."""
    l_b = CB_STABLE * np.sqrt(k / n2)
    length = l_s * l_b / np.sqrt(l_s**2 + l_b**2)
    length = np.minimum(length, np.sqrt(CMU * k / n2))
    rt = length**2 * n2 / (CMU**6 * k)
    return length, rt


# ----------------------------------------------------------------------------
# Closed forms. f is the weight on the production terms: 1 for the model
# equations, rho for the code as written (kynema-sgf issue 2104).
# ----------------------------------------------------------------------------
def neutral_decay(t, k0, length):
    """dk/dt = -Cmu^3 k^1.5 / l."""
    tau = 2.0 * length / (CMU**3 * np.sqrt(k0))
    return k0 / (1.0 + np.asarray(t, dtype=float) / tau) ** 2


def stratified_decay_flux_off(t, k0, length, n2, f=1.0):
    """dk/dt = -Cmu^3 k^1.5/l - f Cmu l sqrt(k) N^2 (N^2 > 0). Returns k, t*."""
    a = CMU**3 / (2.0 * length)
    b = f * CMU * length * n2 / 2.0
    theta0 = np.arctan(np.sqrt(k0 * a / b))
    arg = theta0 - np.sqrt(a * b) * np.asarray(t, dtype=float)
    k = np.where(arg > 0.0, b / a * np.tan(np.maximum(arg, 0.0)) ** 2, 0.0)
    return k, theta0 / np.sqrt(a * b)


def source_relaxation(t, k0, length, src2, f=1.0):
    """dk/dt = f Cmu l sqrt(k) src2 - Cmu^3 k^1.5/l with src2 = S^2 or |N^2|.

    Returns k(t) and the equilibrium k_eq = f l^2 src2 / Cmu^2.
    """
    q_eq = np.sqrt(f * src2) * length / CMU
    rate = CMU**2 * np.sqrt(f * src2) / 2.0
    th = np.tanh(rate * np.asarray(t, dtype=float))
    q0 = np.sqrt(k0)
    q = q_eq * (q0 + q_eq * th) / (q_eq + q0 * th)
    return q**2, q_eq**2


def rk4(rhs, k0, tout, h=0.02):
    """Integrate dk/dt = rhs(k) with classical RK4 and return k at tout."""
    k = np.array(k0, dtype=float)
    t = 0.0
    out = []
    for target in tout:
        while t < target - 1.0e-9:
            step = min(h, target - t)
            a1 = rhs(k)
            a2 = rhs(k + 0.5 * step * a1)
            a3 = rhs(k + 0.5 * step * a2)
            a4 = rhs(k + step * a3)
            k = k + step / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            t += step
        out.append(np.array(k, copy=True))
    return np.array(out)


def stratified_rhs(l_s, n2, shear2=0.0, f=1.0):
    """Right-hand side for the stratified branch (ABL.surface_temp_flux on)."""

    def rhs(k):
        k = np.maximum(k, 1.0e-30)
        length, rt = stable_length(k, n2, l_s)
        prod = f * cmu_rt(rt) * length * np.sqrt(k) * shear2
        buoy = -f * cmu_prime_rt(rt) * length * np.sqrt(k) * n2
        return prod + buoy - CMU**3 * k**1.5 / length

    return rhs


def stratified_limit():
    """Rt and stability functions in the limit l_b << l_s."""
    rt_inf = CB_STABLE**2 / CMU**6
    return rt_inf, cmu_rt(rt_inf), cmu_prime_rt(rt_inf)


def stratified_growth_rate(n, ri=None, f=1.0):
    """Late-time rate d ln k / dt in the limit l_b << l_s.

    ri is the gradient Richardson number N^2/S^2; None means no shear.
    """
    _, cm, cmp_ = stratified_limit()
    shear = 0.0 if ri is None else f * cm * CB_STABLE / ri
    return n * (shear - f * cmp_ * CB_STABLE - CMU**3 / CB_STABLE)


def stationary_richardson(f=1.0):
    """Gradient Richardson number at which the growth rate is zero."""
    _, cm, cmp_ = stratified_limit()
    return f * cm * CB_STABLE / (f * cmp_ * CB_STABLE + CMU**3 / CB_STABLE)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load(run_dir):
    """Read ASCII line-sampler output written with
    sampling.fields = tke temperature velocity mu_turb.

    Returns t[nt], z[nz] and a dict of arrays [nt, nz].
    """
    files = sorted(glob.glob(os.path.join(run_dir, "post_processing", "sampling?????.txt")))
    if not files:
        raise RuntimeError("no sampling output in " + run_dir)
    times, rows = [], []
    for fname in files:
        with open(fname.replace(".txt", "_info.txt")) as info:
            times.append(float(info.readline().split()[1]))
        row = np.loadtxt(fname, skiprows=5, ndmin=2)
        # Rows are grouped by MPI rank in a parallel run: order them by height
        rows.append(row[np.argsort(row[:, 2], kind="stable")])
    data = np.array(rows)
    order = np.argsort(times)
    data = data[order]
    fields = {
        "k": data[:, :, 8],
        "T": data[:, :, 9],
        "u": data[:, :, 10],
        "v": data[:, :, 11],
        "w": data[:, :, 12],
        "mu": data[:, :, 13],
    }
    return np.array(times)[order], data[0, :, 2], fields


def nearest(arr, value):
    return int(np.argmin(np.abs(np.asarray(arr) - value)))


class Report:
    """Collects checks for one case and writes result.json."""

    def __init__(self, case_dir, title):
        self.case_dir = case_dir
        self.title = title
        self.checks = []
        self.notes = []

    def check(self, name, value, reference, tol, kind="rel"):
        if kind == "rel":
            err = abs(value / reference - 1.0)
        else:
            err = abs(value - reference)
        ok = bool(err <= tol)
        self.checks.append(
            {
                "name": name,
                "value": float(value),
                "reference": float(reference),
                "error": float(err),
                "tolerance": float(tol),
                "kind": kind,
                "pass": ok,
            }
        )
        return ok

    def info(self, name, value, reference=None):
        """Record a number that is reported but not used as a pass criterion."""
        entry = {"name": name, "value": float(value)}
        if reference is not None:
            entry["reference"] = float(reference)
        self.notes.append(entry)

    def write(self):
        passed = all(c["pass"] for c in self.checks)
        out = {"title": self.title, "pass": passed, "checks": self.checks, "notes": self.notes}
        name = "result_mutant_%s.json" % MUTANT if MUTANT else "result.json"
        with open(os.path.join(self.case_dir, name), "w") as fh:
            json.dump(out, fh, indent=1)
        print("== %s: %s" % (self.title, "PASS" if passed else "FAIL"))
        for c in self.checks:
            unit = "%" if c["kind"] == "rel" else ""
            scale = 100.0 if c["kind"] == "rel" else 1.0
            print(
                "   [%s] %-46s code %.6g  ref %.6g  err %.3g%s (tol %.3g%s)"
                % (
                    "ok" if c["pass"] else "XX",
                    c["name"],
                    c["value"],
                    c["reference"],
                    scale * c["error"],
                    unit,
                    scale * c["tolerance"],
                    unit,
                )
            )
        for n in self.notes:
            if "reference" in n:
                print("   [--] %-46s code %.6g  ref %.6g" % (n["name"], n["value"], n["reference"]))
            else:
                print("   [--] %-46s %.6g" % (n["name"], n["value"]))
        return passed


def new_figure(title, equation, ncols=2, note=None):
    """Figure with a title line, an equation line and ncols panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, ncols, figsize=(13.33, 6.0), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.69, bottom=0.11, wspace=0.26)
    fig.text(0.02, 0.965, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.02, 0.895, equation, fontsize=15, va="top")
    if note:
        import textwrap

        fig.text(0.02, 0.795, textwrap.fill(note, 150), fontsize=10.5, va="top", color="#333333", linespacing=1.35)
    return fig, np.atleast_1d(axes)


def save_figure(fig, case_dir):
    """Write figure.png of a case; skipped for mutant runs."""
    if not MUTANT:
        fig.savefig(os.path.join(case_dir, "figure.png"), dpi=150)
