#ifndef ABLFIELDINIT_H
#define ABLFIELDINIT_H

#include "amr-wind/core/Field.H"

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Box.H"
#include "AMReX_Geometry.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "AMReX_Gpu.H"

namespace amr_wind {

/** Initialize ABL fields
 */
class ABLFieldInit
{
    static_assert(AMREX_SPACEDIM == 3, "ABL requires 3 dimensional mesh");

public:
    ABLFieldInit();

    void operator()(
        const amrex::Box& vbx,
        const amrex::Geometry& geom,
        const amrex::Array4<amrex::Real>& velocity,
        const amrex::Array4<amrex::Real>& density,
        const amrex::Array4<amrex::Real>& tracer) const;

    /** Add temperature perturbations
     *
     *  This uses amrex::Random and, therefore, executes the loop on CPU and
     *  pushes the field to device.
     */
    void perturb_temperature(
        const int lev,
        const amrex::Geometry& geom,
        Field& temperature);

    //! Flag indicating whether temperature field needs perturbations
    bool add_temperature_perturbations() const { return m_perturb_theta; }

private:
    //! Initial velocity components
    amrex::Vector<amrex::Real> m_vel;


    ///@{
    //! Lookup table containing temperature variation w.r.t. height above
    //! terrain
    amrex::Vector<amrex::Real> m_theta_heights;
    amrex::Vector<amrex::Real> m_theta_values;
    ///@}

    // Device copies of the above arrays
    amrex::Gpu::DeviceVector<amrex::Real> m_thht_d;
    amrex::Gpu::DeviceVector<amrex::Real> m_thvv_d;

    //! Initial density field
    amrex::Real m_rho;

    //! Amplitude of fluctuations in x-direction
    amrex::Real m_deltaU{1.0};

    //! Amplitude of fluctuations in y-direction
    amrex::Real m_deltaV{1.0};

    //! Number of sinusoidal waves in x-direction
    amrex::Real m_Uperiods{4.0};

    //! Number of sinusoidal waves in y-direction
    amrex::Real m_Vperiods{4.0};

    //! Reference height for velocity perturbations
    amrex::Real m_ref_height{50.0};

    //! Amplitude of temperature perturbations
    amrex::Real m_deltaT{0.8};

    //! Mean for Gaussian number generator
    amrex::Real m_theta_gauss_mean{0.0};

    //! Variance for Gaussian number generator
    amrex::Real m_theta_gauss_var{1.0};

    //! Cutoff height for temperature fluctuations
    amrex::Real m_theta_cutoff_height{1.0e16};

    //! Perturb initial velocity field with sinusoidal fluctuations
    bool m_perturb_vel{true};

    //! Perturb temperature field with random fluctuations
    bool m_perturb_theta{false};
};

} // namespace amr_wind

#endif /* ABLFIELDINIT_H */