.. _inputs_turbulence:

Section: turbulence
~~~~~~~~~~~~~~~~~~~

This section is for setting turbulence model parameters

.. input_param:: turbulence.model

   **type:** String, optional, default = Laminar

   Specifies which turbulence model to use, by default "Laminar" is
   chosen (effectively no turbulence model).  Currently the supported
   turbulence models are "Smagorinsky", "AMD", "Kosovic", 
   "OneEqKsgsM84", "KOmegaSST", "KOmegaSSTIDDES" or "KLAxell".

   
.. input_param:: Smagorinsky_coeffs.Cs

   **type:** Real, optional, default = 0.135

   Specifies the coefficient used in the `Smagorinsky` turbulence model. 
   

   

The following inputs change how the ``KLAxell`` model treats the cells next to
the terrain with ``TerrainDrag`` (binary blanking). They are all off by
default, and with every one of them off the model is unchanged. They have no
effect on flat ground or without ``TerrainDrag``.

.. input_param:: KLAxell.terrain_wall_stencil

   **type:** Boolean, optional, default = false

   Computes the strain rate of the fluid cells next to the blanked cells with
   the stencil flat ground uses at its bottom wall, the wall being the face of
   the blanked cell. A central difference into a blanked cell reads the
   near-zero velocity that the drag holds there, which for a horizontal wall
   gives :math:`\partial U/\partial z = U_{k+1}/(2\Delta z)`, 1.4 times the
   log-law shear, and a wall-cell TKE above the value flat ground reaches.
   With this option, in each direction that has a blanked neighbor the
   tangential velocity takes the one-sided derivative
   :math:`(-3u_0 + 4u_1 - u_2)/(2\Delta x)` over the wall cell and the next
   two fluid cells, and the wall-normal velocity the derivative through its
   zero wall value, :math:`(u_0 + u_1/3)/\Delta x`. Between two blanked
   neighbors the derivative is zero; at a domain face the usual boundary
   stencil is kept.

.. input_param:: KLAxell.terrain_blanked_face_length

   **type:** Boolean, optional, default = false

   Measures the mixing length from the top face of the blanked column instead
   of the terrain height. A cell is blanked when its center is at or below
   the terrain height :math:`h`, so the resolved wall is the face at
   :math:`z_f = z_{lo} + \max(0, \Delta z\,\lfloor (h - z_{lo})/\Delta z + 1/2 \rfloor)`,
   and the height entering the mixing length is
   :math:`\max(z_c - z_f, \Delta z/2)`, as flat ground measures it from its
   wall. Measured from :math:`h` it is up to :math:`\Delta z/2` smaller in the
   first fluid cells, depending on where :math:`h` falls within its cell.
