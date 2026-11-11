#!/usr/bin/env python3
"""T5: constant-stress layer, k = u*^2 / Cmu^2 and a log-linear velocity profile."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import klaxell_analytic as kl  # noqa: E402

Z0 = 0.1
U_LID = 10.0
HEIGHT = 128.0
RHO = 1.0

t, z, f = kl.load(os.path.join(HERE, "runs", "base"))
u, k, nut = f["u"][-1], f["k"][-1], f["mu"][-1] / RHO
l_s = kl.lscale_neutral(z)
dudz = np.gradient(u, z)
inner = (z >= 10.0) & (z <= 100.0)
stress = nut * dudz
ustar = np.sqrt(stress[inner].mean())


def profile(zz, us):
    return us / kl.KAPPA * np.log(zz / Z0) + us * (zz - Z0) / kl.LAMBDA


ustar_lid = U_LID / (np.log(HEIGHT / Z0) / kl.KAPPA + (HEIGHT - Z0) / kl.LAMBDA)
j_ref = kl.nearest(z, 10.0)
shape = u[j_ref] + profile(z, ustar) - profile(z[j_ref], ustar)

rep = kl.Report(HERE, "T5 constant-stress layer")
rep.check("k Cmu^2 / u*^2, mean over 10-100 m", (k[inner] * kl.CMU**2 / ustar**2).mean(), 1.0, 5.0e-3)
rep.check("(dU/dz) l_s / u*, mean over 10-100 m", (dudz[inner] * l_s[inner] / ustar).mean(), 1.0, 5.0e-3)
rep.check("u* against the value set by the lid [m/s]", ustar, ustar_lid, 2.0e-2)
rep.check(
    "largest profile difference over 10-100 m / U_lid",
    np.max(np.abs(u[inner] - shape[inner])) / U_LID, 0.0, 3.0e-3, kind="abs",
)
rep.info("stress spread over 10-100 m (std / mean)", stress[inner].std() / stress[inner].mean())
rep.info("change of u* over the last output interval", abs(
    np.sqrt((f["mu"][-2] / RHO * np.gradient(f["u"][-2], z))[inner].mean()) / ustar - 1.0))
rep.info("first-cell k Cmu^2 / u*^2", k[0] * kl.CMU**2 / ustar**2, 1.0)
rep.info("k range over 10-100 m, min", k[inner].min())
rep.info("k range over 10-100 m, max", k[inner].max())
rep.write()

fig, ax = kl.new_figure(
    "T5  Constant-stress layer",
    r"$P=\varepsilon,\ \ \nu_t\frac{dU}{dz}=u_*^2\quad\Rightarrow\quad k=\frac{u_*^2}{C_\mu^2},\qquad"
    r"\frac{dU}{dz}=\frac{u_*}{l_s},\qquad U=\frac{u_*}{\kappa}\ln\frac{z}{z_0}+\frac{u_*(z-z_0)}{\lambda}$",
    note="Column 0-128 m, wall model at the ground (z0 = 0.1 m), lid at 10 m/s, density 1.0, run to steady state. "
    "u* = %.4f m/s from the code's stress; %.4f m/s from the lid velocity and the profile." % (ustar, ustar_lid),
)
ax[0].semilogx(z, shape, "k-", lw=1.3, label="log-linear profile through the value at 10 m")
ax[0].semilogx(z, u[j_ref] + ustar / kl.KAPPA * np.log(z / z[j_ref]), "k:", lw=1.0, label="log law only")
ax[0].semilogx(z, u, "o", ms=4, mfc="none", mew=1.1, color="C3", label="KLAxell")
ax[0].set(xlabel="z [m]", ylabel="U [m/s]", title="Velocity")
ax[1].plot(k * kl.CMU**2 / ustar**2, z, "o", ms=4, mfc="none", mew=1.1, color="C3", label=r"$k\,C_\mu^2/u_*^2$")
ax[1].plot(dudz * l_s / ustar, z, "s", ms=4, mfc="none", mew=1.1, color="C0", label=r"$(dU/dz)\,l_s/u_*$")
ax[1].plot(stress / ustar**2, z, "^", ms=4, mfc="none", mew=1.1, color="C2", label=r"$\nu_t (dU/dz)/u_*^2$")
ax[1].axvline(1.0, color="k", lw=1.0)
ax[1].set(xlabel="normalised value (exact: 1)", ylabel="z [m]", title="Normalised profiles", xlim=(0.6, 1.4))
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=10)
kl.save_figure(fig, HERE)
