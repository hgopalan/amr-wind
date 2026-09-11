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

   
.. input_param:: Smagorinsky_coeffs.Cs

   **type:** Real, optional, default = 0.135

   Specifies the coefficient used in the `Smagorinsky` turbulence model. 
   

   
