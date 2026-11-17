# terraindrag_wall_treatment_abort

## Purpose
`TerrainDrag.wall_treatment` accepts only `original` or `improved`. This deck
sets `improve` and must abort, rather than run with the original treatment.

## Files
- `terraindrag_wall_treatment_abort.inp`: `abl_terrain_improved_wall_stable`
  with `TerrainDrag.wall_treatment = improve` and 1 step.
- `terrain.amrwind`, `rans_1d.info`: copied from
  `abl_terrain_improved_wall_stable`.

## Data and acknowledgements
No external data are used.

## How to run
```
kynema_sgf terraindrag_wall_treatment_abort.inp
```

## Expected Results
The run aborts with
`TerrainDrag.wall_treatment must be original or improved, not improve`.

## CTest
`terraindrag_wall_treatment_abort` passes when the output matches that message
(`PASS_REGULAR_EXPRESSION`).
