#!/usr/bin/env python3
"""C4: above ABL.meso_sponge_start the TKE follows the profile in rans_1d.info."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

SPONGE_START = 500.0
DT = 0.01

prof = np.loadtxt(os.path.join(HERE, "rans_1d.info"))
t, z, f = kl.load(os.path.join(HERE, "runs", "base"))
k = f["k"][-1]
ref = np.interp(z, prof[:, 0], prof[:, 4])
# The first cell above the start height also receives diffusion from below
above = z > SPONGE_START + 16.0
edge = kl.nearest(z, SPONGE_START + 8.0)
below = z < SPONGE_START - 16.0
# One explicit step from the profile value: the dissipation of that step remains
offset = DT * kl.CMU**3 * ref**1.5 / kl.lscale_neutral(z)

rep = kl.Report(HERE, "C4 TKE sponge")
rep.check("largest |k - profile| / profile above 516 m", np.max(np.abs(k[above] / ref[above] - 1.0)), 0.0, 1.0e-3, kind="abs")
rep.check("largest |k - k0| / k0 below 484 m", np.max(np.abs(k[below] / f["k"][0][below] - 1.0)[4:]), 0.0, 1.0e-2, kind="abs")
rep.info("mean (profile - k) above 516 m, against dt * dissipation", np.mean(ref[above] - k[above]), np.mean(offset[above]))
rep.info("k / profile - 1 in the first cell above the start height", k[edge] / ref[edge] - 1.0)
rep.write()

fig, ax = kl.new_figure(
    "C4  TKE sponge",
    r"$z>z_{sponge}$:  source $=-(k-k_{ref}(z))/\Delta t\ \Rightarrow\ k=k_{ref}(z)$ after one step",
    note="Column 0-1024 m, k0 = 1, ABL.meso_sponge_start = 500 m, profile k_ref = 0.2 + 0.0005 z, 10 steps of 0.01 s, density 1.0.",
)
ax[0].plot(f["k"][0], z, "k:", lw=1.2, label="initial")
ax[0].plot(ref, z, "k-", lw=1.3, label="profile in rans_1d.info")
ax[0].plot(k[::3], z[::3], "o", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell after 10 steps")
ax[0].axhline(SPONGE_START, color="C0", ls="--", lw=1)
ax[0].set(xlabel=r"k [m$^2$/s$^2$]", ylabel="z [m]", title="TKE")
ax[1].semilogx(np.abs(k[above] / ref[above] - 1.0) + 1e-17, z[above], "o", ms=5, mfc="none", mew=1.2, color="C3", label="KLAxell")
ax[1].semilogx(offset[above] / ref[above], z[above], "k-", lw=1.3, label=r"$\Delta t\,\varepsilon(k_{ref})/k_{ref}$")
ax[1].set(xlabel="relative difference from the profile", ylabel="z [m]", title="Above 516 m")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend()
kl.save_figure(fig, HERE)
