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
    , m_terrain_blank(sim.repo().declare_int_field("terrain_blank", 1, 1, 1))
    , m_terrain_drag(sim.repo().declare_int_field("terrain_drag", 1, 1, 1))
    , m_terrain_height(sim.repo().declare_field("terrain_height", 1, 1, 1))
{
    std::string terrain_file("terrain.amrwind");
    amrex::ParmParse pp(identifier());
    pp.query("terrain_file", terrain_file);

    ioutils::read_flat_grid_file(
        terrain_file, m_xterrain, m_yterrain, m_zterrain);

    m_sim.io_manager().register_output_int_var("terrain_drag");
    m_sim.io_manager().register_output_int_var("terrain_blank");
    m_sim.io_manager().register_io_var("terrain_height");
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
        auto& blanking = m_terrain_blank(level);
        auto& terrain_height = m_terrain_height(level);
        auto& drag = m_terrain_drag(level);
        // copy terrain data to gpu
        const auto xterrain_size = m_xterrain.size();
        const auto yterrain_size = m_yterrain.size();
        const auto zterrain_size = m_zterrain.size();
        amrex::Gpu::DeviceVector<amrex::Real> device_xterrain(xterrain_size);
        amrex::Gpu::DeviceVector<amrex::Real> device_yterrain(yterrain_size);
        amrex::Gpu::DeviceVector<amrex::Real> device_zterrain(zterrain_size);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_xterrain.begin(), m_xterrain.end(),
            device_xterrain.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_yterrain.begin(), m_yterrain.end(),
            device_yterrain.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_zterrain.begin(), m_zterrain.end(),
            device_zterrain.begin());
        const auto* xterrain_ptr = device_xterrain.data();
        const auto* yterrain_ptr = device_yterrain.data();
        const auto* zterrain_ptr = device_zterrain.data();
        for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            auto levelBlanking = blanking.array(mfi);
            auto levelDrag = drag.array(mfi);
            auto levelheight = terrain_height.array(mfi);
            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
                    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
                    const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
                    const amrex::Real terrainHt = interp::bilinear(
                        xterrain_ptr, xterrain_ptr + xterrain_size,
                        yterrain_ptr, yterrain_ptr + yterrain_size,
                        zterrain_ptr, x, y);
                    levelBlanking(i, j, k, 0) =
                        static_cast<int>(z <= terrainHt);
                    levelheight(i, j, k, 0) =
                        std::max(std::abs(z - terrainHt), 0.5 * dx[2]);
                });

            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    if ((levelBlanking(i, j, k, 0) == 0) && (k > 0) &&
                        (levelBlanking(i, j, k - 1, 0) == 1)) {
                        levelDrag(i, j, k, 0) = 1;
                    } else {
                        levelDrag(i, j, k, 0) = 0;
                    }
                });
        }
    }
}

void TerrainDrag::pre_init_actions()
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::pre_init_actions");
}
} // namespace amr_wind::terraindrag
