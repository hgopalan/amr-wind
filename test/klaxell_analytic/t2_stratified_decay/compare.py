#!/usr/bin/env python3
"""T2: stratified decay with the neutral (flux off) and stratified (flux on) length scale."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

DTDZ = 0.003
n2 = kl.brunt_vaisala_sq(DTDZ)
n = np.sqrt(n2)
rep = kl.Report(HERE, "T2 stratified decay")
heights = (100, 500, 900)
cols = ("C0", "C2", "C3")

ta, z, fa = kl.load(os.path.join(HERE, "runs", "flux_off"))
tb, _, fb = kl.load(os.path.join(HERE, "runs", "flux_on"))
ka, kb = fa["k"], fb["k"]
js = [kl.nearest(z, h) for h in heights]

# Flux off: closed form
for j in js[1:]:
    length = float(kl.lscale_neutral(z[j]))
    for tt in (100, 300, 500):
        i = kl.nearest(ta, tt)
        ref, tstar = kl.stratified_decay_flux_off(ta[i], 1.0, length, n2)
        rep.check("flux off, z = %.0f m, t = %d s" % (z[j], tt), ka[i, j], float(ref), 1.0e-2)
    below = np.argmax(ka[:, j] < 1.0e-6)
    rep.check("flux off, z = %.0f m, time of k = 0 [s]" % z[j], ta[below], tstar, 1.0e-2)

# Flux on: model ODE
tref = np.arange(0.0, tb[-1] + 1.0, 10.0)
ode = kl.rk4(kl.stratified_rhs(kl.lscale_neutral(z[js]), n2), np.ones(len(js)), tref)
for m, j in enumerate(js):
    if m == 0:
        continue
    for tt in (100, 300, 500, 800):
        i = kl.nearest(tb, tt)
        rep.check("flux on, z = %.0f m, t = %d s" % (z[j], tt), kb[i, j], ode[kl.nearest(tref, tt), m], 1.5e-2)
j = js[1]
i1, i2 = kl.nearest(tb, 500), kl.nearest(tb, 900)
rate = -np.log(kb[i2, j] / kb[i1, j]) / (tb[i2] - tb[i1])
rep.check("flux on, decay rate 500-900 s [1/s]", rate, -kl.stratified_growth_rate(n), 2.0e-2)
i = kl.nearest(ta, 300)
rep.info(
    "flux off, z = 100 m, t = 300 s (diffusion acts here)",
    ka[i, js[0]],
    float(kl.stratified_decay_flux_off(ta[i], 1.0, float(kl.lscale_neutral(z[js[0]])), n2)[0]),
)
i = kl.nearest(tb, 300)
rep.info("flux on, z = 100 m, t = 300 s (diffusion acts here)", kb[i, js[0]], ode[kl.nearest(tref, 300), 0])
rep.write()

fig, ax = kl.new_figure(
    "T2  Decay in a stably stratified uniform column",
    r"$\frac{dk}{dt}=-\frac{C_\mu^3 k^{3/2}}{l}-C_\mu^\prime\,l\sqrt{k}\,N^2,\qquad N^2=\frac{g}{T_0}\frac{dT}{dz}$"
    r"$\qquad$flux off: $l=l_s$, $k=\frac{b}{a}\tan^2(\theta_0-\sqrt{ab}\,t)$",
    note="Column 0-1024 m, U = 10 m/s, dT/dz = 0.003 K/m, k0 = 1, density 1.0. Flux on: l = l_s l_b / sqrt(l_s^2 + l_b^2), "
    "l_b = C_b sqrt(k)/N; reference is the model ODE. Lines: reference, symbols: KLAxell.",
)
for j, h, c in zip(js, heights, cols):
    ref, _ = kl.stratified_decay_flux_off(ta, 1.0, float(kl.lscale_neutral(z[j])), n2)
    ax[0].plot(ta, ref, "-", color=c, lw=1.3)
    ax[0].plot(ta[::8], ka[::8, j], "o", ms=5, mfc="none", mew=1.2, color=c, label="z = %d m" % h)
for m, (j, h, c) in enumerate(zip(js, heights, cols)):
    ax[1].semilogy(tref, ode[:, m], "-", color=c, lw=1.3)
    ax[1].semilogy(tb[::8], kb[::8, j], "o", ms=5, mfc="none", mew=1.2, color=c, label="z = %d m" % h)
ax[1].semilogy(tref, 0.6 * np.exp(kl.stratified_growth_rate(n) * tref), "k--", lw=1, label="late-time rate")
ax[0].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="Flux off: closed form", xlim=(0, 1000))
ax[1].set(xlabel="t [s]", ylabel=r"k [m$^2$/s$^2$]", title="Flux on: model ODE", ylim=(1e-6, 2))
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
kl.save_figure(fig, HERE)
