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

.. input_param:: KLAxellSeparation.sensor_source

   **type:** String, optional, default = ``pressure``

   How the pressure-gradient sensor of the "KLAxellSeparation" model is
   evaluated. ``pressure`` uses the pressure gradient as described above.
   ``velocity`` uses the Bernoulli estimate of the same quantity from the
   velocity field, :math:`-\hat{u}_i \, \partial (|\mathbf{u}|^2 / 2) /
   \partial x_i` divided by :math:`(k + c_u |\mathbf{u}|^2) / L`, and stores
   it in the same field. The treatments gated by the sensor change the
   pressure field more than the velocity field, so the velocity estimate
   does not feed those changes back into the gate. It also responds to the
   deceleration by turbulent stresses. With ``velocity`` the sensor is
   multiplied by the smallest fluid weight of the six face neighbors, which
   makes it zero in cells next to the terrain.

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

.. input_param:: KLAxellSeparation_coeffs.gate_relaxation_time

   **type:** Real, optional, default = 0

   Relaxation time :math:`\tau` (s) of the gate of the "KLAxellSeparation"
   treatments. With 0 the gate follows the sensor instantly. With a positive
   value the gate is stored in the field ``separation_gate`` and, once per
   time step, relaxes toward the ramp value
   :math:`g^* = \min(1, \max(0, (s - s_T) / s_T))` of the latest sensor as
   :math:`g \leftarrow g^* + (g - g^*) \exp(-\Delta t / \tau)`; the
   treatments use the stored gate, which lags the flow by one step. This damps
   a feedback in which the treatments change the flow faster than the sensor
   settles. The gate starts at 0, is kept in checkpoint files and is
   interpolated on regrid; where the sensor stays below :math:`s_T` it stays
   exactly 0. Requires ``KLAxellSeparation.pressure_gradient_sensor = true``;
   must not be negative. With the instantaneous gate the realizable
   :math:`C_\mu` limiter produced grid-scale stripes in the eddy viscosity on
   a smooth hill; 10 s removed them with the same effect on the separation
   (30 s gave the same result), so a value of about 10 s is suggested.

.. input_param:: KLAxellSeparation.production_cap

   **type:** Boolean, optional, default = false

   Caps the shear production :math:`P` of turbulent kinetic energy in the
   "KLAxellSeparation" model where the pressure-gradient sensor fires. The
   production above :math:`C_P \varepsilon` is removed with the same gate
   :math:`g` as ``KLAxellSeparation.realizable_cmu``, so the shear production
   in the source becomes :math:`P - g \max(0, P - C_P \varepsilon)`. This limits
   the build-up of turbulent kinetic energy at stagnation points, such as the
   windward foot of a hill. Elsewhere the source is unchanged. With a positive
   ``KLAxellSeparation_coeffs.gate_relaxation_time`` the cap uses the stored,
   time-relaxed gate, the same one as the realizable :math:`C_\mu` limiter.
   Requires ``KLAxellSeparation.pressure_gradient_sensor = true``.

.. input_param:: KLAxellSeparation_coeffs.production_cap_ratio

   **type:** Real, optional, default = 10.0

   Ratio :math:`C_P` of the production cap of the "KLAxellSeparation" model;
   must be positive.

.. input_param:: KLAxellSeparation.destruction_boost

   **type:** Boolean, optional, default = false

   Increases the dissipation of turbulent kinetic energy in the
   "KLAxellSeparation" model where the pressure-gradient sensor fires: the
   source loses an additional :math:`g (c_d - 1) \varepsilon`, with the same
   gate :math:`g` as ``KLAxellSeparation.realizable_cmu`` (the stored,
   time-relaxed gate when ``KLAxellSeparation_coeffs.gate_relaxation_time`` is
   positive). Only the source of the turbulent kinetic energy changes; the
   length scale, the eddy viscosity and the dissipation used in :math:`R_t`
   are unchanged. Elsewhere the source is unchanged. Requires
   ``KLAxellSeparation.pressure_gradient_sensor = true``.

.. input_param:: KLAxellSeparation_coeffs.destruction_boost_factor

   **type:** Real, optional, default = 1.2

   Factor :math:`c_d` on the dissipation where the destruction boost of the
   "KLAxellSeparation" model acts; 1 leaves the source unchanged. Must not be
   below 1.

   
.. input_param:: Smagorinsky_coeffs.Cs

   **type:** Real, optional, default = 0.135

   Specifies the coefficient used in the `Smagorinsky` turbulence model. 
   

   
