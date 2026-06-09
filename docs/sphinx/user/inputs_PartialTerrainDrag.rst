.. _inputs_partial_terrain_drag:

Section: PartialTerrainDrag
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PartialTerrainDrag`` module implements terrain blanking using **volume fractions** (values between 0.0 and 1.0) instead of binary blanking (0 or 1). This provides smoother transitions at terrain boundaries and more accurate representation of terrain geometry.

Key Differences from TerrainDrag
--------------------------------

1. **Volume Fraction Blanking**

   - **TerrainDrag**: Uses ``IntField`` with binary values (0 or 1)
   - **PartialTerrainDrag**: Uses ``Field`` with continuous values (0.0 to 1.0)

2. **Partial Cell Coverage**

   For cells that intersect the terrain surface:

   - Calculates the fraction of the cell volume occupied by terrain
   - Uses linear interpolation: ``fraction = (terrainHt - cell_bottom) / cell_height``
   - Clamps values to [0, 1] range

3. **Smooth Drag Transition**

   The drag field uses a smooth transition based on blanking values:

   - Full drag (1.0) at the terrain interface (where blanking transitions from >0.5 to <0.5)
   - Partial drag for partially blanked cells: ``drag = 1.0 - blanking_fraction``

4. **No Wave Support**

   All wave-related functionality has been removed:

   - No ``m_terrain_is_waves`` flag
   - No ocean wave fields (``ow_vof``, ``ow_levelset``)
   - No ``convert_waves_to_terrain_fields()`` method

Fields
------

The module declares the following fields (all prefixed with ``partial_`` to avoid conflicts):

1. **partial_terrain_blank**: Volume fraction of terrain occupation (0.0 to 1.0)
2. **partial_terrain_drag**: Drag force fraction at terrain surface (0.0 to 1.0)
3. **partial_terrainz0**: Surface roughness length
4. **partial_terrain_height**: Interpolated terrain height at each cell
5. **partial_terrain_damping**: Lateral and vertical damping coefficients

Input Parameters
----------------

Use ``PartialTerrainDrag`` prefix in your input file:

.. input_param:: PartialTerrainDrag.terrain_file

   **type:** String, optional, default = "terrain.amrwind"

   Path to the terrain height file.

.. input_param:: PartialTerrainDrag.roughness_file

   **type:** String, optional, default = "terrain.roughness"

   Path to the roughness file.

.. input_param:: PartialTerrainDrag.blanking_method

   **type:** String, optional, default = "volume_fraction"

   Blanking method selection. Options: ``"volume_fraction"`` or ``"distance_function"``.

.. input_param:: PartialTerrainDrag.smoothing_length

   **type:** Real, optional, default = 1.0

   Smoothing length for distance function approach (in grid cells). Only used when ``blanking_method = "distance_function"``. Controls the transition width at terrain boundary.

.. input_param:: PartialTerrainDrag.damp_east_slope

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (East slope).

.. input_param:: PartialTerrainDrag.damp_east_full

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (East full).

.. input_param:: PartialTerrainDrag.damp_west_slope

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (West slope).

.. input_param:: PartialTerrainDrag.damp_west_full

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (West full).

.. input_param:: PartialTerrainDrag.damp_north_slope

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (North slope).

.. input_param:: PartialTerrainDrag.damp_north_full

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (North full).

.. input_param:: PartialTerrainDrag.damp_south_slope

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (South slope).

.. input_param:: PartialTerrainDrag.damp_south_full

   **type:** Real, optional, default = 0.0

   Lateral damping parameter (South full).

.. input_param:: PartialTerrainDrag.horizontal_time_scale

   **type:** Real, optional, default = 20.0

   Time scale for horizontal damping.

.. input_param:: PartialTerrainDrag.horizontal_abl_height

   **type:** Real, optional, default = 1000.0

   Horizontal ABL height.

.. input_param:: PartialTerrainDrag.horizontal_slope_end

   **type:** Real, optional, default = 2000.0

   Horizontal slope end.

.. input_param:: PartialTerrainDrag.vertical_slope

   **type:** Real, optional, default = 1500.0

   Vertical slope.

.. input_param:: PartialTerrainDrag.vertical_full

   **type:** Real, optional, default = 2000.0

   Vertical full damping parameter.

Physics Activation
------------------

Add to your physics list:

.. code-block::

   incflo.physics = ... PartialTerrainDrag ...

Implementation Details
----------------------

Blanking Methods
~~~~~~~~~~~~~~~~

The module supports two methods for calculating partial blanking:

1. Volume Fraction Method (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``blanking_method = "volume_fraction"``

This method calculates the geometric volume fraction of terrain within each cell. For each cell with center at height ``z``:

- Cell bottom: ``z_bottom = z - dx[2]/2``
- Cell top: ``z_top = z + dx[2]/2``
- Terrain height at (x,y): ``terrainHt``

Volume fraction logic:

.. code-block:: cpp

   if (z_top <= terrainHt) {
       volume_fraction = 1.0;  // Fully below terrain
   }
   else if (z_bottom < terrainHt && z_top > terrainHt) {
       volume_fraction = (terrainHt - z_bottom) / dx[2];  // Partial
   }
   else {
       volume_fraction = 0.0;  // Fully above terrain
   }

**Advantages:**

- Geometrically accurate for coarse meshes
- Sharp interface representation
- No adjustable parameters

2. Distance Function Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``blanking_method = "distance_function"``

This method uses a smooth hyperbolic tangent function based on the distance from the cell center to the terrain surface.

.. code-block:: cpp

   distance = z - terrainHt;
   volume_fraction = 0.5 * (1.0 - tanh(distance / smoothing_length));

**Parameters:**

- ``smoothing_length``: Controls transition width (in grid cells, default = 1.0)
  - Smaller values -> sharper transitions
  - Larger values -> smoother, wider transitions

**Advantages:**

- Smooth, continuous transitions
- No grid alignment sensitivity
- Adjustable transition width
- Better for numerical stability in some solvers

**When to use:**

- **Volume Fraction**: When you need exact geometric representation, especially with coarse grids
- **Distance Function**: When you prefer smooth transitions and better numerical properties

Drag Calculation
~~~~~~~~~~~~~~~~

The drag field identifies cells where drag forces should be applied:

.. code-block:: cpp

   if (current_blanking < 0.5 && below_blanking > 0.5) {
       drag_fraction = 1.0;  // Interface cell
   }
   else if (current_blanking > 0.0 && current_blanking < 1.0) {
       drag_fraction = 1.0 - current_blanking;  // Partial drag
   }

Benefits
--------

1. **Smoother Transitions**: No sharp discontinuities at terrain boundaries
2. **Better Accuracy**: More accurate representation of terrain geometry in coarse meshes
3. **Reduced Grid Sensitivity**: Less dependent on exact alignment of terrain with grid
4. **Physical Consistency**: Volume fractions provide a physically meaningful measure

Future Enhancements
-------------------

Possible improvements:

1. Multi-point sampling within cells for better volume fraction estimates
2. Sub-cell resolution averaging
3. Distance function approach for smoother transitions (Implemented as selectable option)
4. Integration with immersed boundary forcing schemes
5. Hybrid methods combining volume fraction and distance function approaches

Compatibility
-------------

This module is independent of the original ``TerrainDrag`` module. Both can coexist in the codebase, and users can choose which one to use based on their needs.
