# KLAxell analytical tests

## Purpose

Fourteen small cases that compare the `KLAxell` one-equation RANS model with closed-form
solutions of its own equations. Each case isolates one or two terms of the model, so a
wrong coefficient, a wrong sign or a changed time integration shows up as a failed check.

The solver is not modified. Every case is an ordinary input deck run with `kynema_sgf`.
Python is used only after the run, to read the sampler output, evaluate the reference and
draw the figure.

All fourteen cases pass on `f/winsent` at 44344c510 (one MPI rank, CPU, double precision).

## Model equations

$$\frac{Dk}{Dt}=\nabla\cdot\left(2\nu_t\nabla k\right)+\nu_tS^2-\nu_t^\prime N^2-\varepsilon,\qquad
\varepsilon=\frac{C_\mu^3k^{3/2}}{l},\qquad \nu_t=C_\mu(R_t)\,l\sqrt{k},\qquad \nu_t^\prime=C_\mu^\prime(R_t)\,l\sqrt{k}$$

| Symbol | Meaning | Value |
|---|---|---|
| $k$ | turbulent kinetic energy | |
| $S$, $N^2$ | shear, $N^2=(g/T_0)\,dT/dz$ | |
| $l_s$ | neutral length scale $\lambda\kappa z/(\lambda+\kappa z)$, $z$ = absolute height | $\lambda=30$ m, $\kappa=0.41$ |
| $l_b$ | buoyancy length scale $C_b\sqrt{k}/N$ | $C_b=0.25$ (stable), 0.35 (unstable) |
| $R_t$ | turbulent Richardson number $l^2N^2/(C_\mu^6k)$ | |
| $C_\mu(R_t)$ | $(C_\mu+0.108R_t)/(1+0.308R_t+0.00837R_t^2)$ | $C_\mu=0.556$ |
| $C_\mu^\prime(R_t)$ | $C_\mu/(1+0.277R_t)$ | |

Two properties of the implementation shape the decks:

- **Uniform length scale.** $l_s$ uses the absolute height. A box at $z=10^6$ m has
  $l_s=29.998$ m, uniform to 1e-8, which makes the zero-dimensional closed forms exact.
- **Length-scale switch.** With `ABL.surface_temp_flux = 0` (and below
  `ABL.meso_sponge_start`) the model keeps $l=l_s$ and $R_t=0$ whatever the local
  stratification. A non-zero flux selects the stratified length scale. With slip walls
  the flux value acts only as this switch.

## Files

| Folder | Isolates | Reference |
|---|---|---|
| `t1_neutral_decay` | dissipation | $k_0/(1+t/\tau)^2$ |
| `t2_stratified_decay` | buoyancy sink, stable length scale | $\tan^2$ form (flux off), model ODE (flux on) |
| `t3_homogeneous_shear` | shear production | $\tanh$ form, $k_{eq}=l^2S^2/C_\mu^2$ |
| `t4_stratified_shear` | $C_\mu(R_t)$, $C_\mu^\prime(R_t)$, stable length scale | growth rate against Ri, stationary Ri = 0.1498 |
| `t5_constant_stress` | production = dissipation with $l_s(z)$ | $k=u_*^2/C_\mu^2$, log-linear velocity profile |
| `t6_shear_free_diffusion` | TKE diffusivity $2\nu_t$ | $\cosh$ profile (uniform $l$), power law times exponential ($l_s(z)$) |
| `t7_unstable_column` | buoyancy source | $\tanh$ form with $\lvert N^2\rvert$ |
| `c1_field_check` | `mu_turb` over a sweep of $R_t$ | formula, cell by cell |
| `c2_terrain_parity` | second copy of the model used with `TerrainDrag` | flat-branch run |
| `c3_galilean` | independence from a uniform velocity | run at U = 0 |
| `c4_tke_sponge` | TKE sponge above `ABL.meso_sponge_start` | profile in `rans_1d.info` |
| `c5_wall_forcing` | first-cell TKE forcing | geometric approach, ratio $1-1/F$ |
| `c6_timestep_convergence` | order of the source integration | first order |
| `c7_density_weighting` | production terms at density 1.225 | closed forms with the density factor |

Each folder holds its deck(s), `rans_1d.info`, `run.sh`, `compare.py`, `README.md`, and the
`result.json` and `figure.png` of the run reported here. Top level:

- `run_all.sh`: runs every case and prints the summary.
- `summarize.py`: collects every `result.json` into `summary.md`; `--mutants` adds the mutant table.
- `summary.md`: the tables below.
- `KLAxell_analytic_tests.pptx`: 22 slides with the equations, set-up and result of every
  case (listed in `.gitignore`).
- `tools/klaxell_analytic.py` (repository `tools/` folder): constants, closed forms, ODE
  integrator, sampler reader, result and figure writers shared by all cases.

## Data and acknowledgements

No external data are used. Every reference is a closed form or an ODE derived from the
model equations above and evaluated in `tools/klaxell_analytic.py`.

## How to run

Build the solver in its own build folder, then run the cases:

```
cmake -S . -B bld-kltests -DCMAKE_BUILD_TYPE=Release -DKYNEMA_SGF_ENABLE_MPI=ON
cmake --build bld-kltests -j 10
test/klaxell_analytic/run_all.sh $PWD/bld-kltests/kynema_sgf 6
```

One case on its own:

```
test/klaxell_analytic/t3_homogeneous_shear/run.sh $PWD/bld-kltests/kynema_sgf
```

Comparison only, on existing runs, with the mutant table:

```
python3 test/klaxell_analytic/summarize.py --mutants
```

The whole set takes about six minutes on six cores. The two steady-state cases
(`t5_constant_stress`, `t6_shear_free_diffusion`) take most of it. Python needs `numpy`
and `matplotlib`.

## Expected Results

| Case | Checks | Passed | Largest error / tolerance | Status |
|---|---|---|---|---|
| `t1_neutral_decay` | 5 | 5 | 0.38 | pass |
| `t2_stratified_decay` | 17 | 17 | 0.50 | pass |
| `t3_homogeneous_shear` | 8 | 8 | 0.22 | pass |
| `t4_stratified_shear` | 20 | 20 | 0.39 | pass |
| `t5_constant_stress` | 4 | 4 | 0.53 | pass |
| `t6_shear_free_diffusion` | 9 | 9 | 0.02 | pass |
| `t7_unstable_column` | 6 | 6 | 0.09 | pass |
| `c1_field_check` | 2 | 2 | 0.01 | pass |
| `c2_terrain_parity` | 4 | 4 | 0.00 | pass |
| `c3_galilean` | 4 | 4 | 0.22 | pass |
| `c4_tke_sponge` | 2 | 2 | 0.17 | pass |
| `c5_wall_forcing` | 2 | 2 | 0.22 | pass |
| `c6_timestep_convergence` | 1 | 1 | 0.00 | pass |
| `c7_density_weighting` | 7 | 7 | 0.63 | pass |

Headline numbers (density 1.0 unless stated):

| Case | Quantity | Code | Reference |
|---|---|---|---|
| T1 | k at 350 s | 0.249139 | 0.249325 |
| T2 | time at which k = 0, flux off, z = 500 m | 740 s | 741.1 s |
| T2 | decay rate, flux on, 500-900 s | 0.007698 1/s | 0.007678 1/s |
| T3 | equilibrium k | 1.16435 | 1.16437 |
| T4 | growth rate at Ri = 0.05 | 0.02786 1/s | 0.02798 1/s |
| T4 | growth rate at Ri = 0.15 (stationary Ri = 0.1498) | -0.00004 1/s | -0.00002 1/s |
| T5 | $k\,C_\mu^2/u_*^2$ over 10-100 m | 1.0008 | 1 |
| T5 | $(dU/dz)\,l_s/u_*$ over 10-100 m | 0.9996 | 1 |
| T6 | k at 100 m above the fixed-k boundary, uniform $l$ | 0.350468 | 0.350573 |
| T6 | k at z = 50 m, $l_s(z)$ | 0.0126657 | 0.0126645 |
| T7 | equilibrium k | 0.285557 | 0.285561 |
| C1 | largest difference in `mu_turb` | 1.2e-5 | 0 |
| C2 | terrain branch minus flat branch | 0 | 0 |
| C6 | observed order in time | 1.0005 | 1 |
| C7 | equilibrium k in shear at density 1.225 | 1.42634 | 1.42635 (f = 1.225); 1.16437 (f = 1) |

### Proof that the checks can fail

`summarize.py --mutants` reruns every comparison with one reference constant perturbed
($C_\mu$ +2 %, $C_b$ +5 %, $\lambda$ +5 %). No solver run is repeated. The table lists how
many checks then fail.

| Case | mutant `cmu` | mutant `cb` | mutant `lambda` |
|---|---|---|---|
| `t1_neutral_decay` | 5 of 5 fail | 0 of 5 fail | 5 of 5 fail |
| `t2_stratified_decay` | 17 of 17 fail | 9 of 17 fail | 13 of 17 fail |
| `t3_homogeneous_shear` | 8 of 8 fail | 0 of 8 fail | 8 of 8 fail |
| `t4_stratified_shear` | 18 of 20 fail | 20 of 20 fail | 2 of 20 fail |
| `t5_constant_stress` | 1 of 4 fail | 0 of 4 fail | 2 of 4 fail |
| `t6_shear_free_diffusion` | 7 of 9 fail | 0 of 9 fail | 8 of 9 fail |
| `t7_unstable_column` | 6 of 6 fail | 0 of 6 fail | 6 of 6 fail |
| `c1_field_check` | 2 of 2 fail | 2 of 2 fail | 2 of 2 fail |
| `c2_terrain_parity` | 0 of 4 fail | 0 of 4 fail | 0 of 4 fail |
| `c3_galilean` | 0 of 4 fail | 0 of 4 fail | 0 of 4 fail |
| `c4_tke_sponge` | 0 of 2 fail | 0 of 2 fail | 0 of 2 fail |
| `c5_wall_forcing` | 0 of 2 fail | 0 of 2 fail | 0 of 2 fail |
| `c6_timestep_convergence` | 1 of 1 fail | 0 of 1 fail | 1 of 1 fail |
| `c7_density_weighting` | 7 of 7 fail | 0 of 7 fail | 7 of 7 fail |

- $C_b$ enters only the stratified branch, so the neutral and flux-off cases do not see it.
- `c2` to `c5` compare runs with each other or with input values and do not depend on
  these constants. `c7_density_weighting` is their solver-side counterpart: at density
  1.225 the solver departs from the density-free closed forms by 4 % to 91 %.
- `t5_constant_stress` is the least sensitive case to these three mutants; its checks are
  ratios in which $C_\mu$ partly cancels.

## What the cases showed

1. **Density weighting of the production terms** (kynema-sgf issues 2104 and 2105). At
   `incflo.density = 1.225` the buoyancy term and the shear term both follow the closed
   forms with a factor $\rho$. In shear the equilibrium k is 1.4263 against 1.1644 without
   the factor. The base cases use density 1.0, where the factor is 1.
2. **First-order time integration of the TKE source.** The error halves with the time
   step (observed order 1.00). At `fixed_dt = 0.25` the error of the neutral decay is
   0.07 %.
3. **First-cell TKE in the constant-stress layer.** In T5 the first-cell k is 0.74 of the
   interior value $u_*^2/C_\mu^2$; the interior matches to 0.1 %.
4. **Sponge interface.** The first cell above `ABL.meso_sponge_start` differs from the
   profile by 0.4 % because TKE diffuses in from the cell below.
5. **Uniform-velocity independence under stratification** holds to 2e-5, not to round-off;
   the difference comes from solver-tolerance noise in w and T.

## Input notes

| Input | Value in the decks | Effect |
|---|---|---|
| `geometry.prob_lo` (z) | `1.0e6` | uniform $l_s$ = 30 m |
| `DragForcing.bc_forcing_time_factor` | `1.0e30` | switches off the first-cell TKE forcing (T5 and C5 use 5 and 20) |
| `ABL.meso_sponge_start` | `1.0e9` | switches off the TKE sponge and keeps the neutral branch (C4 uses 500) |
| `ABL.surface_temp_flux` | `0.0` or `1.0e-4` | selects the neutral or the stratified length scale |
| `ABL.rans_1dprofile_file` | `rans_1d.info` | required by `KransAxell`; used only by C4 |
| `ABL.linear_profile`, `ABL.bottom_velocity`, `ABL.top_velocity` | shear cases | the profile uses the absolute height, so the values carry the offset $S\,z_{mid}$ |
| `zlo.density`, `zhi.density` | on no-slip walls | must equal `incflo.density`, otherwise the constant-density check aborts |
| `zlo.tke_type = no_slip_wall`, `zlo.tke` | T6 | fixed TKE at a boundary |
| `mac_proj.mg_atol`, `nodal_proj.mg_atol` | `1.0e-10` | a uniform flow has a round-off residual; without an absolute tolerance the MAC projection runs to its iteration cap |

## Limits

- The references are derived by hand from the model as read in `KLAxell.cpp` and
  `KransAxell.cpp`. They are checked here only against the solver and the mutant runs.
- One MPI rank, CPU, double precision. One three-rank run of T2 (flux on, 300 s, four
  boxes) agreed with the serial run to 3e-8 in k. GPU, single precision and tiled runs
  were not tried.
- No mesh refinement. The length scale and sources are evaluated per level and were not
  exercised across a refinement boundary.
- T2 uses zero-dimensional references on a column where $l_s$ varies; the checks use
  z = 500 m and 900 m, where vertical diffusion changes k by less than 0.6 %.
- C2 places the terrain below the domain. It covers the formulas of the terrain branch,
  not blanked cells or the terrain wall treatment.
- The canopy terms (`ForestDrag.canopy_tke`) are not covered.

## CTest

Not registered. `run_all.sh` runs the set and `summarize.py` exits non-zero if any case
fails. `c1_field_check` and the stability functions need no time integration and are
candidates for unit tests in `unit_tests/turbulence/test_turbulence_RANS.cpp`. `t1`, `t3`,
`t7` and `c2` to `c6` run in seconds and are the natural first regression tests.

## Not in this folder

- A fix for the density weighting (issue 2104).
- CTest registration and unit-test versions of the cases.
- Tests of the canopy terms and of the terrain wall treatment.
