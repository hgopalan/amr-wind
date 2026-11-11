#!/usr/bin/env python3
"""T6: steady balance of TKE diffusion and dissipation away from a fixed-k boundary."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

HEIGHT = 320.0
K_BND = 1.0
M = np.sqrt(3.0) * kl.CMU / 2.0  # decay rate of k^1.5 in xi, d(xi) = dz / l

rep = kl.Report(HERE, "T6 shear-free diffusion")

# (a) box at z = 1e6 m, fixed k at the lower face, zero gradient at the top
t, z, f = kl.load(os.path.join(HERE, "runs", "elevated"))
zloc = z - z[0] + 0.5 * (z[1] - z[0])
length = float(kl.lscale_neutral(z.mean()))
ka = f["k"][-1]
exact_a = K_BND * (np.cosh(M * (HEIGHT - zloc) / length) / np.cosh(M * HEIGHT / length)) ** (2.0 / 3.0)
for zz in (50, 100, 200, 300):
    j = kl.nearest(zloc, zz)
    rep.check("elevated, k at %d m above the boundary" % zz, ka[j], exact_a[j], 2.0e-2)
rep.info("elevated, largest change over the last output interval", np.max(np.abs(f["k"][-1] / f["k"][-2] - 1.0)))
rep.info("elevated, e-folding length of k far from the top [m]", np.sqrt(3.0) * length / kl.CMU)

# (b) column on the ground, fixed k at the top face, zero gradient at the ground
tb, zb, fb = kl.load(os.path.join(HERE, "runs", "ground"))
kb = fb["k"][-1]


def xi(zz):
    return np.log(zz) / kl.KAPPA + zz / kl.LAMBDA


exact_b = K_BND * np.exp(2.0 * M / 3.0 * (xi(zb) - xi(HEIGHT)))
for zz in (20, 50, 100, 200, 300):
    j = kl.nearest(zb, zz)
    rep.check("ground, k at z = %d m" % zz, kb[j], exact_b[j], 3.0e-2)
rep.info("ground, largest change over the last output interval", np.max(np.abs(fb["k"][-1] / fb["k"][-2] - 1.0)))
rep.info("ground, exponent of the power law near the ground", kl.CMU / (np.sqrt(3.0) * kl.KAPPA))
rep.write()

fig, ax = kl.new_figure(
    "T6  Shear-free diffusion from a fixed-k boundary",
    r"$\frac{d}{dz}\!\left(2C_\mu l\sqrt{k}\frac{dk}{dz}\right)=\frac{C_\mu^3k^{3/2}}{l}\ \Rightarrow\ "
    r"\frac{d^2k^{3/2}}{d\xi^2}=m^2k^{3/2},\quad d\xi=\frac{dz}{l},\quad m=\frac{\sqrt{3}C_\mu}{2}$",
    note="No mean flow, density 1.0, run to steady state. Left: uniform l, k^1.5 ~ cosh(m (H - z)/l). "
    "Right: l = l_s(z), k = k_top (z/H)^p exp(p kappa (z - H)/lambda), p = Cmu / (sqrt(3) kappa) = %.3f."
    % (kl.CMU / (np.sqrt(3.0) * kl.KAPPA)),
)
ax[0].semilogy(zloc, exact_a, "k-", lw=1.3, label="closed form")
ax[0].semilogy(zloc[::3], ka[::3], "o", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell")
ax[0].set(xlabel="height above the fixed-k boundary [m]", ylabel=r"k [m$^2$/s$^2$]", title="(a) Box at z = 1e6 m, k fixed below")
ax[1].loglog(zb, exact_b, "k-", lw=1.3, label="closed form")
ax[1].loglog(zb[::3], kb[::3], "o", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell")
ax[1].set(xlabel="z [m]", ylabel=r"k [m$^2$/s$^2$]", title="(b) Column on the ground, k fixed at the top")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
kl.save_figure(fig, HERE)
