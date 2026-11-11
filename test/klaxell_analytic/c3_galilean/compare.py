#!/usr/bin/env python3
"""C3: the decay must not depend on the value of a uniform velocity."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

rep = kl.Report(HERE, "C3 Galilean invariance")
fig, ax = kl.new_figure(
    "C3  Independence from a uniform velocity",
    r"With uniform $U$ there is no shear, so $k(t)$ must be the same for $U$ = 0, 10 and 20 m/s.",
    note="Box at z = 1e6 m, density 1.0. Neutral decay and stably stratified decay (dT/dz = 0.01 K/m, flux switch on). "
    "Lines: U = 0, symbols: U = 10 and 20 m/s.",
)
for case, c in (("neutral", "C0"), ("stratified", "C3")):
    t, z, base = kl.load(os.path.join(HERE, "runs", case + "_u00"))
    ax[0].semilogy(t, base["k"].mean(1), "-", color=c, lw=1.3, label=case + ", U = 0")
    for u, mk in ((10, "o"), (20, "s")):
        _, _, f = kl.load(os.path.join(HERE, "runs", "%s_u%02d" % (case, u)))
        diff = np.max(np.abs(f["k"][1:] / base["k"][1:] - 1.0), axis=1)
        # The stratified runs carry solver-tolerance noise in w and T (1e-7), which
        # grows relative to k as k decays by four orders of magnitude
        tol = 1.0e-8 if case == "neutral" else 1.0e-4
        rep.check("%s: largest relative difference in k, U = %d against U = 0" % (case, u), diff.max(), 0.0, tol, kind="abs")
        ax[0].semilogy(t[::2], f["k"].mean(1)[::2], mk, ms=5, mfc="none", mew=1.2, color=c, label="%s, U = %d" % (case, u))
        ax[1].semilogy(t[1:], diff + 1e-17, "-" + mk, ms=4, color=c, label="%s, U = %d" % (case, u))
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="k against time")
ax[1].set(xlabel="t [s]", ylabel="largest relative difference in k", title="Difference from the U = 0 run")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=9)
rep.write()
kl.save_figure(fig, HERE)
