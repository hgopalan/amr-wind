#!/usr/bin/env python3
"""T3: homogeneous shear, relaxation to k_eq = l^2 S^2 / Cmu^2."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

SHEAR = 0.02
rep = kl.Report(HERE, "T3 homogeneous shear")
fig, ax = kl.new_figure(
    "T3  Homogeneous shear, neutral",
    r"$\frac{dk}{dt}=C_\mu\,l\sqrt{k}\,S^2-\frac{C_\mu^3k^{3/2}}{l}\quad\Rightarrow\quad"
    r"\sqrt{k}=q_{eq}\frac{q_0+q_{eq}\tanh(rt)}{q_{eq}+q_0\tanh(rt)},\quad q_{eq}=\frac{lS}{C_\mu},\quad r=\frac{C_\mu^2S}{2}$",
    note="Box at z = 1e6 m between two moving walls, U = S (z - z_mid), S = 0.02 1/s, density 1.0. "
    "Lines: closed form, symbols: KLAxell.",
)
keq = 0.0
for name, c in (("k0_low", "C0"), ("k0_high", "C3")):
    t, z, f = kl.load(os.path.join(HERE, "runs", name))
    k = f["k"].mean(axis=1)
    length = float(kl.lscale_neutral(z.mean()))
    ref, keq = kl.source_relaxation(t, k[0], length, SHEAR**2)
    for tt in (100, 400, 1000, 2000):
        i = kl.nearest(t, tt)
        rep.check("k0 = %.1f, t = %d s" % (k[0], tt), k[i], ref[i], 3.0e-3)
    shear = np.gradient(f["u"][-1], z)
    rep.info("k0 = %.1f: shear at end, mid box [1/s]" % k[0], shear[len(z) // 2], SHEAR)
    rep.info("k0 = %.1f: spread of k over the box at end" % k[0], (f["k"][-1].max() - f["k"][-1].min()) / k[-1])
    height = z - z[0] + 0.5 * (z[1] - z[0])
    ax[0].plot(t, ref, "-", color=c, lw=1.3)
    ax[0].plot(t[::4], k[::4], "o", ms=5, mfc="none", mew=1.2, color=c, label="k0 = %.1f" % k[0])
    ax[1].plot(height, f["u"][0], "-", color=c, lw=1.0)
    ax[1].plot(height, f["u"][-1], "o", ms=5, mfc="none", mew=1.2, color=c, label="k0 = %.1f, end" % k[0])
rep.info("k_eq [m2/s2]", keq)
ax[0].axhline(keq, color="k", ls=":", lw=1)
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="k against time (dotted: k_eq)")
ax[1].set(xlabel="height in box [m]", ylabel="U [m/s]", title="Velocity: initial (line) and final (symbols)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
rep.write()
kl.save_figure(fig, HERE)
