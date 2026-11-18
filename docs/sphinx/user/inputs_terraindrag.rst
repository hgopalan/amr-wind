.. _inputs_terraindrag:

Section: TerrainDrag
~~~~~~~~~~~~~~~~~~~~

These parameters are active when ``TerrainDrag`` is included in
:input_param:`incflo.physics`.

TerrainDrag has two operation modes:

1. Terrain-file mode (default), where terrain and roughness are read from
   files.
2. Ocean-wave mode, activated automatically when ``OceanWaves`` physics is
   active and the ``vof`` field is not present. In this mode, terrain blanking
   and drag markers are constructed from wave fields.

.. input_param:: TerrainDrag.terrain_file

   **type:** String, optional, default = ``terrain.amrwind``

   Input file for terrain height data in terrain-file mode.

.. input_param:: TerrainDrag.roughness_file

   **type:** String, optional, default = ``terrain.roughness``

   Input file for roughness-length data in terrain-file mode.

   If this file is missing or cannot be opened, the roughness length is set
   to 0.1 m everywhere.

.. input_param:: TerrainDrag.wall_treatment

   **type:** String, optional, default = ``original``

   Near-wall treatment of the cells next to the blanked terrain with the
   ``KLAxell`` model. ``original`` keeps the original model. ``improved``
   turns on five corrections that make the cells next to the terrain behave
   like the first cells above flat ground:

   - :input_param:`KLAxell.terrain_wall_stencil`: the strain rate of the wall
     cells from the flat-ground wall stencil;
   - :input_param:`KLAxell.terrain_blanked_face_length`: the mixing length
     measured from the top face of the blanked column;
   - :input_param:`KLAxell.terrain_face_stress`: the drag-cell viscosity set
     so that the face above it carries the wall stress;
   - :input_param:`KLAxell.terrain_face_heat_flux`: the drag-cell heat
     diffusivity set so that the face above it carries the surface heat flux
     (``ABL.wall_het_model = mol``);
   - :input_param:`DragTempForcing.blank_follow_fluid`: the blanked cells
     relaxed toward the air above them instead of the soil temperature.

   With ``improved`` each correction can still be turned off on its own, and
   with ``original`` each can be turned on on its own. Any other value aborts.

   The four ``KLAxell`` corrections act only with ``turbulence.model =
   KLAxell``; :input_param:`DragTempForcing.blank_follow_fluid` acts with any
   turbulence model that uses ``DragTempForcing``. The corrections assume that
   ``DragForcing`` (momentum) and ``DragTempForcing`` (temperature) are in the
   source terms, since they size the face fluxes from the velocity and
   temperature differences that these forcings hold. With a ``KLAxell``
   correction on and ``TerrainDrag`` not in :input_param:`incflo.physics`,
   the run aborts. :input_param:`KLAxell.terrain_face_heat_flux` is turned off
   without ``ABL.wall_het_model = mol``, and
   :input_param:`KLAxell.terrain_blanked_face_length` is turned off in the
   ocean-wave mode, where the blanking follows the waves rather than the
   terrain height; set explicitly in those cases, each aborts.

   The larger drag-cell diffusivities from the face corrections are
   damped with ``incflo.diffusion_type = 2`` (implicit). Crank–Nicolson
   (``1``) does not damp large diffusion numbers, so the drag cells can
   oscillate from step to step. The test
   cases ``abl_terrain_improved_wall_stable`` and
   ``abl_terrain_improved_wall_unstable`` use ``2``.

.. input_param:: TerrainDrag.damp_east_slope

   **type:** Real, optional, default = 0.0

   Width of the east (+x) damping ramp region.

.. input_param:: TerrainDrag.damp_east_full

   **type:** Real, optional, default = 0.0

   Width of the east (+x) full-strength damping region adjacent to the boundary.

.. input_param:: TerrainDrag.damp_west_slope

   **type:** Real, optional, default = 0.0

   Width of the west (-x) damping ramp region.

.. input_param:: TerrainDrag.damp_west_full

   **type:** Real, optional, default = 0.0

   Width of the west (-x) full-strength damping region adjacent to the boundary.

.. input_param:: TerrainDrag.damp_north_slope

   **type:** Real, optional, default = 0.0

   Width of the north (+y) damping ramp region.

.. input_param:: TerrainDrag.damp_north_full

   **type:** Real, optional, default = 0.0

   Width of the north (+y) full-strength damping region adjacent to the boundary.

.. input_param:: TerrainDrag.damp_south_slope

   **type:** Real, optional, default = 0.0

   Width of the south (-y) damping ramp region.

.. input_param:: TerrainDrag.damp_south_full

   **type:** Real, optional, default = 0.0

   Width of the south (-y) full-strength damping region adjacent to the boundary.

.. input_param:: TerrainDrag.horizontal_time_scale

   **type:** Real, optional, default = 20.0

   Time scale used to convert the assembled damping coefficient into
   ``terrain_damping`` forcing (larger values reduce damping strength).

.. input_param:: TerrainDrag.horizontal_abl_height

   **type:** Real, optional, default = 1.0e15

   Height at which horizontal damping begins to turn on.

.. input_param:: TerrainDrag.horizontal_slope_end

   **type:** Real, optional, default = 1.0e30

   Height where the horizontal damping vertical multiplier reaches full value.

.. input_param:: TerrainDrag.vertical_slope

   **type:** Real, optional, default = 0.0

   Height where additional full-domain vertical damping begins to ramp on.

.. input_param:: TerrainDrag.vertical_full

   **type:** Real, optional, default = 1.0e30

   Height where additional full-domain vertical damping reaches full strength.

.. input_param:: TerrainDrag.terrain_aware_damping

   **type:** Boolean, optional, default = false

   When true, the heights used by the horizontal and vertical damping are
   measured above the local terrain (with a floor of half a cell height)
   instead of above ``geometry.prob_lo``.

Notes
-----

- The terrain drag marker field (``terrain_drag``) is derived from transitions
  in ``terrain_blank`` and is used by :doc:`inputs_Momentum_Sources` when
  ``DragForcing`` is active.
- In ocean-wave mode, TerrainDrag derives ``terrain_blank`` and
  ``terrain_height`` from wave fields and sets a uniform low roughness
  (``terrainz0 = 1.0e-4``).
- In ocean-wave mode, terrain-file and roughness-file parameters are not used.
