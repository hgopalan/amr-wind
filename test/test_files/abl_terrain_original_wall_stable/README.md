# abl_terrain_original_wall_stable

## Purpose
Companion of `abl_terrain_improved_wall_stable` with
`TerrainDrag.wall_treatment = original`: the same deck run with the original
near-wall treatment, so that the old and new paths are both in the test suite.

## Files
- `abl_terrain_original_wall_stable.inp`: as `abl_terrain_improved_wall_stable`
  except `TerrainDrag.wall_treatment = original`.
- `terrain.amrwind`, `rans_1d.info`: copied from
  `abl_terrain_improved_wall_stable` (from kynema/kynema-sgf#2067, hgopalan).

## Data and acknowledgements
No external data are used.

## How to run
```
mpirun -np 4 kynema_sgf abl_terrain_original_wall_stable.inp
```

## Expected Results
The run completes. Its plotfile agrees bit for bit with the same deck run on
the code before the near-wall treatment was added. The table in
`abl_terrain_improved_wall_stable/README.md` compares the two treatments.

## CTest
`abl_terrain_original_wall_stable` runs the deck (no gold file yet).
