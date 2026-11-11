#!/usr/bin/env python3
"""T4: stratified homogeneous shear, growth rate against the gradient Richardson number."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

DTDZ = 0.01
n2 = kl.brunt_vaisala_sq(DTDZ)
n = np.sqrt(n2)
ri_list = (0.05, 0.10, 0.15, 0.25, 0.50)
rep = kl.Report(HERE, "T4 stratified homogeneous shear")
ri_st = kl.stationary_richardson()
rt_inf, cm, cmp_ = kl.stratified_limit()
fig, ax = kl.new_figure(
    "T4  Homogeneous shear with stable stratification",
    r"$l_b\ll l_s:\ \ \frac{1}{k}\frac{dk}{dt}=N\left[\frac{C_\mu(R_{t\infty})C_b}{Ri}-C_\mu^\prime(R_{t\infty})C_b-\frac{C_\mu^3}{C_b}\right]"
    r",\quad R_{t\infty}=\frac{C_b^2}{C_\mu^6},\quad Ri=\frac{N^2}{S^2}$",
    note="Box at z = 1e6 m between two moving walls, dT/dz = 0.01 K/m, k0 = 0.01, flux switch on, density 1.0. "
    "Stationary Ri = %.4f. Lines: model ODE / formula, symbols: KLAxell." % ri_st,
)
rates = []
for m, ri in enumerate(ri_list):
    t, z, f = kl.load(os.path.join(HERE, "runs", "ri_%03d" % round(ri * 100)))
    k = f["k"].mean(axis=1)
    length = float(kl.lscale_neutral(z.mean()))
    shear2 = n2 / ri
    ode = kl.rk4(kl.stratified_rhs(length, n2, shear2), k[0], t, h=0.01)
    for tt in (100, 300, 600):
        i = kl.nearest(t, tt)
        rep.check("Ri = %.2f, k at t = %d s" % (ri, tt), k[i], ode[i], 2.0e-2)
    i1, i2 = kl.nearest(t, 10), kl.nearest(t, 40)
    rate = np.log(k[i2] / k[i1]) / (t[i2] - t[i1])
    rates.append(rate)
    rep.check("Ri = %.2f, growth rate 10-40 s [1/s]" % ri, rate, kl.stratified_growth_rate(n, ri), 3.0e-4, kind="abs")
    c = "C%d" % m
    ax[0].semilogy(t, ode, "-", color=c, lw=1.3)
    ax[0].semilogy(t[::6], k[::6], "o", ms=5, mfc="none", mew=1.2, color=c, label="Ri = %.2f" % ri)
ri_line = np.linspace(0.04, 0.55, 200)
ax[1].plot(ri_line, kl.stratified_growth_rate(n, ri_line), "k-", lw=1.3, label="formula")
ax[1].plot(ri_list, rates, "o", ms=7, mfc="none", mew=1.5, color="C3", label="KLAxell, 10-40 s")
ax[1].axhline(0.0, color="k", lw=0.6)
ax[1].axvline(ri_st, color="k", ls=":", lw=1)
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="k against time")
ax[1].set(xlabel="Ri", ylabel="growth rate [1/s]", title="Initial growth rate (dotted: stationary Ri)")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=10)
rep.info("stationary Richardson number", ri_st)
rep.info("Rt in the limit l_b << l_s", rt_inf)
rep.write()
kl.save_figure(fig, HERE)
