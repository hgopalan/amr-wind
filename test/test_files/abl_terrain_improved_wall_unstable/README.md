# abl_terrain_improved_wall_unstable

## Purpose
Gaussian hill with `TerrainDrag`, `DragForcing`, `DragTempForcing` and the
`KLAxell` model under an unstable Monin–Obukhov surface layer (`L = -186` m), with
`TerrainDrag.wall_treatment = improved`. All five near-wall corrections are on:
`KLAxell.terrain_wall_stencil`, `terrain_blanked_face_length`,
`terrain_face_stress`, `terrain_face_heat_flux` and
`DragTempForcing.blank_follow_fluid`.

## Files
- `abl_terrain_improved_wall_unstable.inp`: 32 x 32 x 52 cells, 10 steps,
  `incflo.diffusion_type = 2`.
- `terrain.amrwind`: Gaussian hill.
- `rans_1d.info`: 1-D RANS profile for the initial wind and temperature.

The deck and both data files are taken from kynema/kynema-sgf#2067 (hgopalan),
without the inputs that winsent does not have.

## Data and acknowledgements
No external data are used.

## How to run
```
mpirun -np 4 kynema_sgf abl_terrain_improved_wall_unstable.inp
```

## Expected Results
The run completes. With `L < 0` the face stress and the face heat flux use the
unstable branch of the stability function; the face heat flux acts only where
the temperature difference across the face runs down the gradient of the
upward flux.

## CTest
`abl_terrain_improved_wall_unstable` runs the deck (no gold file yet).
