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
   

   

.. input_param:: KLAxell_coeffs.Cmu

   **type:** Real, optional, default = 0.556

   Neutral stability constant :math:`C_\mu` of the `KLAxell` model. It sets the
   eddy viscosity, and the `KransAxell` source uses the same value for the
   dissipation and for the wall value of the turbulent kinetic energy. Must be
   > 0; the run aborts otherwise. `ABL.Cmu`, if given, must equal it.

.. input_param:: KLAxell_coeffs.Cb_stable

   **type:** Real, optional, default = 0.25

   Coefficient of the buoyancy length scale :math:`C_b \sqrt{k}/N` in stable
   stratification. Must be > 0; the run aborts otherwise.

.. input_param:: KLAxell_coeffs.Cb_unstable

   **type:** Real, optional, default = 0.35

   Coefficient of the length scale in unstable stratification. Must be > 0;
   the run aborts otherwise.

`KLAxell_coeffs.Cmu_prime` and `KLAxell_coeffs.prandtl` were read but never used: the
stability function :math:`C_\mu'` and the turbulent Prandtl number follow from `Cmu` and
the turbulent Richardson number. Setting either key now aborts the run. The `KLAxell`
length scale uses :input_param:`ABL.kappa`, like the `KransAxell` wall boundary condition.
