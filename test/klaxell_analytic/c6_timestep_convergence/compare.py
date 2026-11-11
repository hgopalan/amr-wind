#!/usr/bin/env python3
"""C6: observed order of the time integration of the TKE source."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

steps = (2.0, 1.0, 0.5, 0.25)
errors = []
for dt in steps:
    t, z, f = kl.load(os.path.join(HERE, "runs", "dt_%04d" % round(dt * 1000)))
    k = f["k"].mean(axis=1)
    exact = kl.neutral_decay(t[-1], k[0], float(kl.lscale_neutral(z.mean())))
    errors.append(abs(k[-1] / exact - 1.0))
errors = np.array(errors)
orders = np.log2(errors[:-1] / errors[1:])

rep = kl.Report(HERE, "C6 time-step convergence")
rep.check("observed order, dt = 0.5 to 0.25 s", orders[-1], 1.0, 0.15, kind="abs")
for dt, err in zip(steps, errors):
    rep.info("relative error at t = 350 s, dt = %g s" % dt, err)
for dt, order in zip(steps[:-1], orders):
    rep.info("observed order, dt = %g to %g s" % (dt, dt / 2), order)
rep.write()

fig, ax = kl.new_figure(
    "C6  Time-step convergence of the neutral decay",
    r"error $=\left|k(350\,\mathrm{s})/k_{exact}-1\right|$ for $\Delta t$ = 2, 1, 0.5, 0.25 s",
    note="Box at z = 1e6 m, no mean flow, k0 = 1, density 1.0. Observed order between the two smallest steps: %.2f." % orders[-1],
)
ax[0].loglog(steps, errors, "o-", ms=7, mfc="none", mew=1.4, color="C3", label="KLAxell")
ax[0].loglog(steps, errors[-1] * np.array(steps) / steps[-1], "k--", lw=1, label="first order")
ax[0].loglog(steps, errors[-1] * (np.array(steps) / steps[-1]) ** 2, "k:", lw=1, label="second order")
ax[0].set(xlabel=r"$\Delta t$ [s]", ylabel="relative error in k at 350 s", title="Error against time step")
ax[1].plot(steps[:-1], orders, "o-", ms=7, mfc="none", mew=1.4, color="C3")
ax[1].set(xlabel=r"coarser $\Delta t$ of the pair [s]", ylabel="observed order", title="Observed order", ylim=(0, 2.5))
for a in ax:
    a.grid(alpha=0.3, which="both")
ax[0].legend()
kl.save_figure(fig, HERE)
