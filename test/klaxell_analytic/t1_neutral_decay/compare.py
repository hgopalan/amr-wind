#!/usr/bin/env python3
"""T1: neutral decay against k0 / (1 + t/tau)^2."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

t, z, f = kl.load(os.path.join(HERE, "runs", "base"))
k = f["k"].mean(axis=1)
k0 = k[0]
length = float(kl.lscale_neutral(z.mean()))
tau = 2.0 * length / (kl.CMU**3 * np.sqrt(k0))
exact = kl.neutral_decay(t, k0, length)

rep = kl.Report(HERE, "T1 neutral decay")
for tt in (50, 350, 1000, 3500):
    i = kl.nearest(t, tt)
    rep.check("k at t = %d s" % tt, k[i], exact[i], 2.0e-3)
i1, i2 = kl.nearest(t, 1750), len(t) - 1
slope = np.log(k[i2] / k[i1]) / np.log((1 + t[i2] / tau) / (1 + t[i1] / tau))
rep.check("log slope against (1 + t/tau)", slope, -2.0, 5.0e-3)
rep.info("length scale l [m]", length)
rep.info("tau [s]", tau)
rep.info("largest spread of k over the box", np.max((f["k"].max(1) - f["k"].min(1)) / k))
rep.write()

fig, ax = kl.new_figure(
    "T1  Neutral decay in a uniform flow",
    r"$\frac{dk}{dt}=-\frac{C_\mu^3 k^{3/2}}{l}\quad\Rightarrow\quad k(t)=\frac{k_0}{(1+t/\tau)^2},\quad \tau=\frac{2l}{C_\mu^3\sqrt{k_0}}$",
    note="Box at z = 1e6 m: l = %.2f m, tau = %.0f s. U = 10 m/s, k0 = 1 m2/s2, slip walls, periodic sides." % (length, tau),
)
ax[0].plot(t, exact, "k-", lw=1.4, label="closed form")
ax[0].plot(t[::8], k[::8], "o", ms=6, mfc="none", mew=1.4, color="C3", label="KLAxell")
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="Decay in time")
ax[1].loglog(1 + t / tau, exact, "k-", lw=1.4, label="closed form, slope -2")
ax[1].loglog(1 + t[::4] / tau, k[::4], "o", ms=6, mfc="none", mew=1.4, color="C3", label="KLAxell")
ax[1].set(xlabel=r"$1+t/\tau$", ylabel=r"k [m$^2$/s$^2$]", title="Log axes")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
kl.save_figure(fig, HERE)
