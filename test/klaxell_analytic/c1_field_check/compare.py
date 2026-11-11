#!/usr/bin/env python3
"""C1: eddy viscosity against its formula over a sweep of stratification."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

RHO = 1.0
TINY = np.finfo(float).eps

t, z, f = kl.load(os.path.join(HERE, "runs", "base"))
k, temp, mu = f["k"][-1], f["T"][-1], f["mu"][-1]
n2 = kl.GRAV / kl.T_REF * np.gradient(temp, z)
l_s = kl.lscale_neutral(z)


def one_update(length_prev):
    """One pass of the length-scale update as written in KLAxell.cpp."""
    eps = kl.CMU**3 * k**1.5 / (length_prev + TINY)
    rt = kl.limit_rt((k / eps) ** 2 * n2)
    l_b = kl.CB_STABLE * np.sqrt(k / np.maximum(n2, TINY))
    stable = np.sqrt((l_s * l_b) ** 2 / (l_s**2 + l_b**2))
    stable = np.minimum(stable, np.sqrt(kl.CMU * k / np.maximum(n2, TINY)))
    unstable = l_s * np.sqrt(1.0 - kl.CMU**6 / kl.CB_UNSTABLE**2 * rt)
    return np.where(n2 > 0.0, stable, unstable), rt


length = np.zeros_like(z)
history = []
for _ in range(400):
    length, rt = one_update(length)
    history.append(length.copy())
# The code evaluates Rt with the length scale of the previous update
mu_ref = RHO * kl.cmu_rt(rt) * length * np.sqrt(k)
drift = np.max(np.abs(history[-1] / history[-2] - 1.0))

inner = slice(2, len(z) - 2)
err = np.abs(mu[inner] / mu_ref[inner] - 1.0)
stable = n2[inner] > 0.0
rep = kl.Report(HERE, "C1 eddy-viscosity field check")
rep.check("largest difference in mu_turb, stable cells", err[stable].max(), 0.0, 1.0e-3, kind="abs")
rep.check("largest difference in mu_turb, unstable cells", err[~stable].max(), 0.0, 1.0e-3, kind="abs")
rep.info("Rt range in the column, min", rt[inner].min())
rep.info("Rt range in the column, max", rt[inner].max())
rep.info("change of the reference length scale in its last iteration", drift)
rep.info("largest change of k from its initial value", np.max(np.abs(k / f["k"][0] - 1.0)))
rep.write()

fig, ax = kl.new_figure(
    "C1  Eddy viscosity against its formula",
    r"$\mu_t=\rho\,C_\mu(R_t)\,l\,\sqrt{k},\qquad C_\mu(R_t)=\frac{C_\mu+0.108R_t}{1+0.308R_t+0.00837R_t^2},\qquad R_t=\frac{l^2N^2}{C_\mu^6k}$",
    note="Column 0-1024 m, k = 0.5, dT/dz from -0.005 K/m at the ground to +0.01 K/m at the top, flux switch on, 40 steps of 1e-3 s. "
    "Reference: the length-scale update iterated to a fixed point (the code reuses the previous length scale in Rt).",
)
ax[0].plot(mu_ref, z, "k-", lw=1.3, label="formula")
ax[0].plot(mu[::3], z[::3], "o", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell")
ax[0].set(xlabel=r"$\mu_t$ [kg/m/s]", ylabel="z [m]", title="Eddy viscosity")
ax[1].plot(rt, z, "-", color="C0", lw=1.3, label=r"$R_t$ (limited below -1)")
ax[1].plot(length / l_s, z, "-", color="C2", lw=1.3, label=r"$l/l_s$")
ax[1].axvline(0.0, color="k", lw=0.6)
ax[1].set(xlabel="value", ylabel="z [m]", title="Stratification sweep covered")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
kl.save_figure(fig, HERE)
