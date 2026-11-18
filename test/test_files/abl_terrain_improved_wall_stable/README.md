# abl_terrain_improved_wall_stable

## Purpose
Gaussian hill with `TerrainDrag`, `DragForcing`, `DragTempForcing` and the
`KLAxell` model under a stable Monin–Obukhov surface layer (`L = 150` m), with
`TerrainDrag.wall_treatment = improved`. All five near-wall corrections are on:
`KLAxell.terrain_wall_stencil`, `terrain_blanked_face_length`,
`terrain_face_stress`, `terrain_face_heat_flux` and
`DragTempForcing.blank_follow_fluid`.

## Files
- `abl_terrain_improved_wall_stable.inp`: 32 x 32 x 22 cells, 10 steps,
  `incflo.diffusion_type = 2`.
- `terrain.amrwind`: Gaussian hill.
- `rans_1d.info`: 1-D RANS profile for the initial wind and temperature.

The deck and both data files are taken from kynema/kynema-sgf#2067 (hgopalan),
without the inputs that winsent does not have.

## Data and acknowledgements
No external data are used.

## How to run
```
mpirun -np 4 kynema_sgf abl_terrain_improved_wall_stable.inp
```

## Expected Results
The run completes. Compared with `abl_terrain_original_wall_stable` after 10
steps on 4 ranks (means over the 1024 drag cells):

| Quantity | original | improved |
|---|---|---|
| TKE (m2/s2) | 0.398 | 0.356 |
| `mu_turb` | 1.21 | 2.37 |
| `temperature_mueff` | 1.08 | 5.29 |
| Temperature of the blanked cell below (K) | 263.47 | 264.23 |

The cell above the drag cell changes by less than 2 %.

## CTest
`abl_terrain_improved_wall_stable` runs the deck (no gold file yet).
