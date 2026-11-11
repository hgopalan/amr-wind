#!/usr/bin/env python3
"""C7: weight of the production terms at density 1.225 (kynema-sgf issue 2104)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

RHO = 1.225
SHEAR = 0.02
n2 = kl.brunt_vaisala_sq(0.003)
rep = kl.Report(HERE, "C7 density weighting of the production terms")
fig, ax = kl.new_figure(
    "C7  Production terms at density 1.225",
    r"$\frac{dk}{dt}=f\left[C_\mu\,l\sqrt{k}\,S^2-C_\mu^\prime\,l\sqrt{k}\,N^2\right]-\frac{C_\mu^3k^{3/2}}{l},\qquad"
    r"f=1$ (model equations) or $f=\rho$ (code as written)",
    note="T2 (flux off, z = 500 m) and T3 repeated with incflo.density = 1.225. The checks here pin the present behaviour (f = rho); "
    "they are expected to change to f = 1 when kynema-sgf issue 2104 is fixed.",
)

# Buoyancy term: stratified decay, flux off
t, z, f = kl.load(os.path.join(HERE, "runs", "stratified_rho1225"))
j = kl.nearest(z, 500.0)
length = float(kl.lscale_neutral(z[j]))
k = f["k"][:, j]
ref1, ts1 = kl.stratified_decay_flux_off(t, 1.0, length, n2, 1.0)
refr, tsr = kl.stratified_decay_flux_off(t, 1.0, length, n2, RHO)
for tt in (100, 300, 500):
    i = kl.nearest(t, tt)
    rep.check("buoyancy: k at t = %d s against f = rho" % tt, k[i], refr[i], 1.0e-2)
    rep.info("buoyancy: k at t = %d s against f = 1" % tt, k[i], ref1[i])
rep.check("buoyancy: time of k = 0 against f = rho [s]", t[np.argmax(k < 1.0e-6)], tsr, 1.0e-2)
rep.info("buoyancy: time of k = 0 for f = 1 [s]", ts1)
ax[0].semilogy(t, ref1, "-", color="C0", lw=1.3, label="closed form, f = 1")
ax[0].semilogy(t, refr, "-", color="C3", lw=1.3, label="closed form, f = 1.225")
ax[0].semilogy(t[::6], k[::6], "s", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell, density 1.225")
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="Buoyancy term (stratified decay)", ylim=(1e-5, 2), xlim=(0, 800))

# Shear term: homogeneous shear
t, z, f = kl.load(os.path.join(HERE, "runs", "shear_rho1225"))
k = f["k"].mean(axis=1)
length = float(kl.lscale_neutral(z.mean()))
ref1, keq1 = kl.source_relaxation(t, k[0], length, SHEAR**2, 1.0)
refr, keqr = kl.source_relaxation(t, k[0], length, SHEAR**2, RHO)
for tt in (400, 1000, 2000):
    i = kl.nearest(t, tt)
    rep.check("shear: k at t = %d s against f = rho" % tt, k[i], refr[i], 3.0e-3)
    rep.info("shear: k at t = %d s against f = 1" % tt, k[i], ref1[i])
rep.info("shear: k_eq for f = rho", keqr, keq1)
ax[1].plot(t, ref1, "-", color="C0", lw=1.3, label="closed form, f = 1")
ax[1].plot(t, refr, "-", color="C3", lw=1.3, label="closed form, f = 1.225")
ax[1].plot(t[::4], k[::4], "s", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell, density 1.225")
ax[1].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="Shear term (homogeneous shear)")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
rep.write()
kl.save_figure(fig, HERE)
