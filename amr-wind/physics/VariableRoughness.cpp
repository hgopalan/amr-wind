#include "amr-wind/physics/TerrainDrag.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_iMultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_ParReduce.H"
#include "amr-wind/utilities/trig_ops.H"
#include "amr-wind/utilities/IOManager.H"
#include "amr-wind/utilities/io_utils.H"
#include "amr-wind/utilities/linear_interpolation.H"

namespace amr_wind::terraindrag {

namespace {} // namespace

TerrainDrag::TerrainDrag(CFDSim& sim)
    : m_sim(sim)
    , m_repo(sim.repo())
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_terrainz0(sim.repo().declare_field("terrainz0", 1, 1, 1))
{
    std::string roughness_file("terrain.roughness");
    amrex::ParmParse pp(identifier());
    pp.query("roughness_file", roughness_file);

    std::ifstream file(roughness_file, std::ios::in);
    if (file.good()) {
        ioutils::read_flat_grid_file(
            roughness_file, m_xrough, m_yrough, m_z0rough);
    }
    file.close();

    m_sim.io_manager().register_io_var("terrainz0");
}

void TerrainDrag::post_init_actions()
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::post_init_actions");
    const auto& geom_vec = m_sim.repo().mesh().Geom();
    const int nlevels = m_sim.repo().num_active_levels();
    for (int level = 0; level < nlevels; ++level) {
        const auto& geom = geom_vec[level];
        const auto& dx = geom.CellSizeArray();
        const auto& prob_lo = geom.ProbLoArray();
        auto& velocity = m_velocity(level);
        auto& terrainz0 = m_terrainz0(level);
        // Copy Roughness to gpu
        const auto xrough_size = m_xrough.size();
        const auto yrough_size = m_yrough.size();
        const auto z0rough_size = m_z0rough.size();
        amrex::Gpu::DeviceVector<amrex::Real> device_xrough(xrough_size);
        amrex::Gpu::DeviceVector<amrex::Real> device_yrough(yrough_size);
        amrex::Gpu::DeviceVector<amrex::Real> device_z0rough(z0rough_size);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_xrough.begin(), m_xrough.end(),
            device_xrough.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_yrough.begin(), m_yrough.end(),
            device_yrough.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_z0rough.begin(), m_z0rough.end(),
            device_z0rough.begin());
        const auto* xrough_ptr = device_xrough.data();
        const auto* yrough_ptr = device_yrough.data();
        const auto* z0rough_ptr = device_z0rough.data();
        for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            auto levelz0 = terrainz0.array(mfi);
            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
                    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];

                    amrex::Real roughz0 = 0.1;
                    if (xrough_size > 0) {
                        roughz0 = interp::bilinear(
                            xrough_ptr, xrough_ptr + xrough_size, yrough_ptr,
                            yrough_ptr + yrough_size, z0rough_ptr, x, y);
                    }
                    levelz0(i, j, k, 0) = roughz0;
                });
        }
    }
}

void TerrainDrag::pre_init_actions()
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::pre_init_actions");
}
} // namespace amr_wind::terraindrag
