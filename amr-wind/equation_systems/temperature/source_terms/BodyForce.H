#ifndef BODYFORCE_H
#define BODYFORCE_H

#include "amr-wind/equation_systems/temperature/TemperatureSource.H"
#include "amr-wind/core/SimTime.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <cstddef>

namespace amr_wind::pde::temperature {

class BodyForce : public TemperatureSource::Register<BodyForce>
{

public:
    static std::string identifier() { return "BodyForce"; }

    explicit BodyForce(const CFDSim& /*sim*/);

    ~BodyForce() override;

    void operator()(
        const int lev,
        const amrex::MFIter& /*mfi*/,
        const amrex::Box& bx,
        const FieldState /*fstate*/,
        const amrex::Array4<amrex::Real>& src_term) const override;

    void read_bforce_profile(const std::string& filename);

private:
    const SimTime& m_time;
    const amrex::AmrCore& m_mesh;

    std::string m_type{"height_varying"};
    std::string m_bforce_file;
    size_t m_bforce_profile_nhts;

    amrex::Gpu::DeviceVector<amrex::Real> m_prof_theta;
    amrex::Gpu::DeviceVector<amrex::Real> m_ht;
};

} // namespace amr_wind::pde::temperature

#endif
