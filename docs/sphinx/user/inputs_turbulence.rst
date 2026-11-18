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

The `Kosovic` model adds the nonlinear backscatter stress of Kosovic (1997) to
a Smagorinsky eddy viscosity. The nonlinear stress acts through the
`NonLinearSGSTerm` momentum source, which must be listed in
`ICNS.source_terms`. With `C_1 = C_2` the stress is

.. math::

   \tau^{NL}_{ij} = -(C_s \Delta)^2 C_1
   \left( S_{ik} S_{kj} + S_{ik} R_{kj} - R_{ik} S_{kj} \right),

where :math:`S` and :math:`R` are the symmetric and antisymmetric parts of
the velocity gradient :math:`\partial u_i / \partial x_j`, and the isotropic
part is absorbed in the pressure. The model also reads
:input_param:`ABL.surface_roughness_z0`, :input_param:`ABL.kappa`,
`ABL.mo_beta_m`, `ABL.mo_gamma_m` and :input_param:`ABL.surface_stability_type`.

.. input_param:: Kosovic.Cb

   **type:** Real, optional, default = 0.36

   Backscatter coefficient. It sets :math:`C_s = \sqrt{8 (1 + C_b) / (27 \pi^2)}`
   and :math:`C_1 = C_2 = \sqrt{960}\, C_b / (7 (1 + C_b) S_k)` with
   :math:`S_k = 0.5`. Must be >= 0; 0 turns the nonlinear stress off.

.. input_param:: Kosovic_coeffs.Cs

   **type:** Real, optional, default = value set by `Kosovic.Cb`

   Replaces the Smagorinsky coefficient computed from `Kosovic.Cb`. It does
   not change :math:`C_1`. Must be > 0.

.. input_param:: Kosovic.surfaceRANS

   **type:** Boolean, optional, default = false

   Blends the LES length scale with the near-wall RANS length scale
   :math:`(\kappa z / \phi_m(z/L))^2`, where :math:`z` is the height of the
   top face of the cell above the ground (the lower domain face, or the
   terrain with `TerrainDrag`).

.. input_param:: Kosovic.switchLoc

   **type:** Real, optional, default = 24.0

   Height (m) of the RANS-LES blend, :math:`f = \exp(-z/z_{switch})`. Read
   only with `Kosovic.surfaceRANS = true`. Must be > 0.

.. input_param:: Kosovic.surfaceRANSExp

   **type:** Real, optional, default = 2.0

   Exponent :math:`n` of the blend, :math:`(1-f)^n` for LES and :math:`f^n`
   for RANS. Read only with `Kosovic.surfaceRANS = true`. Must be > 0.

.. input_param:: Kosovic.LESTurbOff_transition_zaglstart

   **type:** Real, optional, default = no limit

   Height above the ground (m) where the model starts to be tapered off with
   a cosine-squared function. Must be <= `Kosovic.LESTurbOff_transition_zaglend`.

.. input_param:: Kosovic.LESTurbOff_transition_zaglend

   **type:** Real, optional, default = no limit

   Height above the ground (m) above which the eddy viscosity and the
   nonlinear stress are zero. Equal start and end heights give a step.

.. input_param:: Kosovic.muCoeff

   **type:** Real, optional, default = 3.0

   Ratio of the turbulent thermal diffusivity to the eddy viscosity (the
   inverse turbulent Prandtl number). Must be >= 0. The stability correction
   of the eddy viscosity, :math:`\sqrt{|S|^2 - 3 N^2}`, keeps the factor 3
   whatever this value is.

.. input_param:: Kosovic.writeTerms

   **type:** Boolean, optional, default = false

   Adds the fields `Nij` (the tensor
   :math:`4 (S_{ik} S_{kj} + S_{ik} R_{kj} - R_{ik} S_{kj})`, unscaled) and
   `divNij` (the momentum source) to the plot files.


   

.. input_param:: OneEqKsgsM84_coeffs.density_weighted_production

   **type:** Boolean, optional, default = false

   Selects how the shear and buoyancy production terms of the subgrid TKE
   equation are built in the `OneEqKsgsM84` model. With the default
   (``false``) both terms are built per unit mass from
   :math:`\nu_t = \mu_t / \rho`, in the same way as the dissipation term,
   before the TKE equation multiplies the whole source by density. With
   ``true`` the terms are built from :math:`\mu_t` and carry an extra factor
   of :math:`\rho` relative to dissipation. This was the behavior before
   this option was added. It is kept only to reproduce earlier results.
   The two settings give the same answer when ``incflo.density = 1``.

.. input_param:: KLAxell_coeffs.density_weighted_production

   **type:** Boolean, optional, default = false

   Same as :input_param:`OneEqKsgsM84_coeffs.density_weighted_production`,
   for the shear and buoyancy production terms of the `KLAxell` model.
