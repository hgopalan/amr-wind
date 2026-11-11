#!/usr/bin/env python3
"""T7: unstable uniform flow, relaxation to k_eq = l^2 |N^2| / Cmu^2."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

DTDZ = -0.003
n2 = abs(kl.brunt_vaisala_sq(DTDZ))
rep = kl.Report(HERE, "T7 unstable column")
fig, ax = kl.new_figure(
    "T7  Unstable stratification, uniform flow",
    r"$\frac{dk}{dt}=C_\mu\,l\sqrt{k}\,|N^2|-\frac{C_\mu^3k^{3/2}}{l}\quad\Rightarrow\quad"
    r"\sqrt{k}=q_{eq}\frac{q_0+q_{eq}\tanh(rt)}{q_{eq}+q_0\tanh(rt)},\quad q_{eq}=\frac{l|N|}{C_\mu},\quad r=\frac{C_\mu^2|N|}{2}$",
    note="Box at z = 1e6 m, U = 10 m/s, dT/dz = -0.003 K/m, ABL.surface_temp_flux = 0 (neutral length scale), density 1.0. "
    "Lines: closed form, symbols: KLAxell.",
)
keq = 0.0
for name, c in (("k0_low", "C0"), ("k0_high", "C3")):
    t, z, f = kl.load(os.path.join(HERE, "runs", name))
    k = f["k"].mean(axis=1)
    length = float(kl.lscale_neutral(z.mean()))
    ref, keq = kl.source_relaxation(t, k[0], length, n2)
    for tt in (200, 1000, 4000):
        i = kl.nearest(t, tt)
        rep.check("k0 = %.2f, t = %d s" % (k[0], tt), k[i], ref[i], 5.0e-3)
    grad = np.gradient(f["T"][-1], z)
    rep.info("k0 = %.2f: dT/dz at end, mid box [K/m]" % k[0], grad[len(z) // 2], DTDZ)
    rep.info("k0 = %.2f: largest |w| at end [m/s]" % k[0], np.max(np.abs(f["w"][-1])))
    ax[0].plot(t, ref, "-", color=c, lw=1.3)
    ax[0].plot(t[::4], k[::4], "o", ms=5, mfc="none", mew=1.2, color=c, label="k0 = %.2f" % k[0])
    ax[1].semilogy(t, np.abs(ref - keq) + 1e-16, "-", color=c, lw=1.3)
    ax[1].semilogy(t[::4], np.abs(k[::4] - keq) + 1e-16, "o", ms=5, mfc="none", mew=1.2, color=c, label="k0 = %.2f" % k[0])
rep.info("k_eq [m2/s2]", keq)
ax[0].axhline(keq, color="k", ls=":", lw=1)
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="k against time (dotted: k_eq)")
ax[1].set(xlabel="t [s]", ylabel=r"$|k-k_{eq}|$ [m$^2$/s$^2$]", title="Distance from equilibrium", ylim=(1e-6, 2))
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
rep.write()
kl.save_figure(fig, HERE)
