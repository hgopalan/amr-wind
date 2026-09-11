.. _inputs_turbulence:

Section: turbulence
~~~~~~~~~~~~~~~~~~~

This section is for setting turbulence model parameters

.. input_param:: turbulence.model

   **type:** String, optional, default = Laminar

   Specifies which turbulence model to use, by default "Laminar" is
   chosen (effectively no turbulence model).  Currently the supported
   turbulence models are "Smagorinsky", "AMD", "Kosovic", 
   "OneEqKsgsM84", "KOmegaSST", "KOmegaSSTIDDES", "KLAxell" or
   "KLAxellSeparation". "KLAxellSeparation" is the "KLAxell" model with
   optional treatments for separating flow over terrain; with every
   treatment disabled it gives the same results as "KLAxell".

.. input_param:: KLAxellSeparation.pressure_gradient_sensor

   **type:** Boolean, optional, default = false

   Computes the pressure-gradient sensor of the "KLAxellSeparation" model
   and stores it in the field ``pressure_gradient_sensor``: the pressure
   gradient along the local flow direction,
   :math:`\hat{u}_i \, \partial p / \partial x_i`, divided by
   :math:`\rho \, (k + c_u |\mathbf{u}|^2) / L`, where :math:`L` is the
   turbulent length scale. Inside the boundary layer the turbulent kinetic
   energy sets the scale; above it, where the turbulent kinetic energy
   vanishes, the velocity term keeps the sensor bounded. The sensor is a
   diagnostic and does not change the solution; add
   ``pressure_gradient_sensor`` to ``io.outputs`` to write it.

.. input_param:: KLAxellSeparation_coeffs.sensor_velocity_weight

   **type:** Real, optional, default = 0.05

   Weight :math:`c_u` of the velocity term in the scale of the
   pressure-gradient sensor of the "KLAxellSeparation" model.

.. input_param:: KLAxellSeparation_coeffs.sensor_threshold

   **type:** Real, optional, default = 0.1

   Sensor value :math:`s_T` above which the treatments of the
   "KLAxellSeparation" model start to act. Their weight ramps linearly from 0
   at :math:`s_T` to 1 at :math:`2 s_T`, so the treatments switch on smoothly
   instead of changing from one cell to the next. Must be positive. The
   default is provisional until the treatments are calibrated.

.. input_param:: KLAxellSeparation.realizable_cmu

   **type:** Boolean, optional, default = false

   Limits the eddy viscosity of the "KLAxellSeparation" model where the
   pressure-gradient sensor :math:`s` exceeds
   ``KLAxellSeparation_coeffs.sensor_threshold`` (:math:`s_T`). There, the eddy
   viscosity and the shear and buoyancy production are divided by
   :math:`1 + c_s \, g \, \max(0, \Sigma / C_\mu - 1)`, with the gate
   :math:`g = \min(1, \max(0, (s - s_T) / s_T))` and
   :math:`\Sigma = L S / \sqrt{k}`, which equals :math:`C_\mu` in an
   equilibrium log layer. For large :math:`\Sigma` the eddy viscosity tends
   to :math:`\rho \, C_\mu(R_t) \, C_\mu k / (c_s S)`. Elsewhere the model
   is unchanged. Requires ``KLAxellSeparation.pressure_gradient_sensor =
   true``.

.. input_param:: KLAxellSeparation_coeffs.realizable_cmu_strength

   **type:** Real, optional, default = 1.0

   Strength :math:`c_s` of the realizable :math:`C_\mu` limiter of the
   "KLAxellSeparation" model; 0 leaves the eddy viscosity unchanged.

   
.. input_param:: Smagorinsky_coeffs.Cs

   **type:** Real, optional, default = 0.135

   Specifies the coefficient used in the `Smagorinsky` turbulence model. 
   

   
