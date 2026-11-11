#!/usr/bin/env python3
"""C2: the terrain branch of the model against the flat branch."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

rep = kl.Report(HERE, "C2 terrain-branch parity")
fig, ax = kl.new_figure(
    "C2  Terrain branch against flat branch",
    r"KLAxell and KransAxell hold a second copy of the model for runs with TerrainDrag.",
    note="Box at z = 1e6 m, flat terrain at z = 0 far below the box, density 1.0. The two branches should give the same k and mu_turb. "
    "Lines: flat branch, symbols: terrain branch.",
)
for m, (case, c) in enumerate((("neutral", "C0"), ("stratified", "C3"))):
    t, z, flat = kl.load(os.path.join(HERE, "runs", case + "_flat"))
    _, _, terr = kl.load(os.path.join(HERE, "runs", case + "_terrain"))
    for name in ("k", "mu"):
        diff = np.max(np.abs(terr[name][1:] / flat[name][1:] - 1.0))
        rep.check("%s: largest relative difference in %s" % (case, "k" if name == "k" else "mu_turb"), diff, 0.0, 1.0e-8, kind="abs")
    rep.info("%s: largest difference in U [m/s]" % case, np.max(np.abs(terr["u"] - flat["u"])))
    ax[0].semilogy(t, flat["k"].mean(1), "-", color=c, lw=1.3)
    ax[0].semilogy(t, terr["k"].mean(1), "o", ms=5, mfc="none", mew=1.2, color=c, label=case)
    ax[1].semilogy(t[1:], np.max(np.abs(terr["k"][1:] / flat["k"][1:] - 1.0), axis=1) + 1e-17, "-o", ms=4, color=c, label=case)
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="k against time")
ax[1].set(xlabel="t [s]", ylabel="largest relative difference in k", title="Terrain minus flat")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
rep.write()
kl.save_figure(fig, HERE)
