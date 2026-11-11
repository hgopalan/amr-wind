#!/usr/bin/env python3
"""C5: first-cell TKE forcing, geometric approach to u*^2 / Cmu^2."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

Z0 = 0.1
rep = kl.Report(HERE, "C5 first-cell TKE forcing")
fig, ax = kl.new_figure(
    "C5  First-cell TKE forcing",
    r"$k_1^{n+1}-k_1^{n}\approx\frac{k_t-k_1^{n}}{F},\qquad k_t=\frac{u_*^2}{C_\mu^2},\qquad u_*=\frac{\kappa\,U_2}{\ln(3z_1/z_0)}$",
    note="Box at z = 1e6 m, U = 10 m/s, k0 = 1, dt = 0.01 s, density 1.0. F = DragForcing.bc_forcing_time_factor. "
    "The step-to-step ratio of the increments should be 1 - 1/F. Lines: geometric sequence, symbols: KLAxell.",
)
for name, factor, c in (("factor_05", 5.0, "C0"), ("factor_20", 20.0, "C3")):
    t, z, f = kl.load(os.path.join(HERE, "runs", name))
    k1 = f["k"][:, 0]
    ustar = kl.KAPPA * f["u"][0, 1] / np.log(3.0 * z[0] / Z0)
    k_target = ustar**2 / kl.CMU**2
    inc = np.diff(k1)
    ratio = inc[1:13] / inc[0:12]
    rep.check("F = %g: ratio of successive increments" % factor, ratio.mean(), 1.0 - 1.0 / factor, 2.0e-2)
    rep.info("F = %g: first-cell k after %d steps" % (factor, len(t) - 1), k1[-1], k_target)
    rep.info("F = %g: second-cell k after %d steps" % (factor, len(t) - 1), f["k"][-1, 1])
    steps = np.arange(len(t))
    ax[0].plot(steps, k_target + (k1[0] - k_target) * (1.0 - 1.0 / factor) ** steps, "-", color=c, lw=1.3)
    ax[0].plot(steps, k1, "o", ms=4, mfc="none", mew=1.1, color=c, label="F = %g" % factor)
    ax[0].axhline(k_target, color="k", ls=":", lw=1)
    ax[1].plot(steps[1:13], ratio, "o", ms=5, mfc="none", mew=1.2, color=c, label="F = %g" % factor)
    ax[1].axhline(1.0 - 1.0 / factor, color=c, lw=1.3)
ax[0].set(xlabel="step", ylabel=r"first-cell k [m$^2$/s$^2$]", title="First-cell k (dotted: target)")
ax[1].set(xlabel="step", ylabel="ratio of successive increments", title="Ratio (lines: 1 - 1/F)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
rep.write()
kl.save_figure(fig, HERE)
