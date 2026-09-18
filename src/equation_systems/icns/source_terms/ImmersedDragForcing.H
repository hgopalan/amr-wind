#ifndef IMMERSEDDRAGFORCING_H
#define IMMERSEDDRAGFORCING_H

#include "src/equation_systems/icns/MomentumSource.H"
#include "src/core/SimTime.H"
#include "src/CFDSim.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

/** Momentum forcing for terrain represented by the ImmersedTerrain physics
 *
 *  \ingroup icns_src
 *
 *  Consumes ``terrain_fraction`` declared by ImmersedTerrain. Immersed drag
 *  in every cell with a non-zero drag weight w, relaxing the velocity toward
 *  zero at the rate C = w * drag_coefficient / dz. The relaxation is
 *  integrated exactly over the time step, C_eff = (1 - exp(-C dt)) / dt, so it
 *  is stable and monotone for any C and dt with no ad hoc limiters.
 *
 *  The drag weight follows ``ImmersedTerrain.drag_weight``: ``fraction`` uses
 *  w = beta, ``center`` uses w = 1 where the cell center is inside the
 *  terrain. With laminar flow and ``fraction`` weighting the partial cell is
 *  a fluid cell whose wall is carried by the no-slip wall flux, so only cells
 *  entirely inside the terrain take the drag.
 *
 *  With ``ImmersedTerrain.implicit_projection`` the drag is applied inside the
 *  projections instead and this source adds nothing.
 */
class ImmersedDragForcing : public MomentumSource::Register<ImmersedDragForcing>
{
public:
    static std::string identifier() { return "ImmersedDragForcing"; }

    explicit ImmersedDragForcing(const CFDSim& sim);

    ~ImmersedDragForcing() override;

    void operator()(
        int lev, FieldState fstate, amrex::MultiFab& src_term) const override;

private:
    const SimTime& m_time;
    const CFDSim& m_sim;
    const amrex::AmrCore& m_mesh;
    const Field& m_velocity;

    //! Immersed drag relaxation rate is drag_coefficient / dz
    amrex::Real m_drag_coefficient{10.0_rt};
    //! ``fraction`` or ``center`` drag weight, read from ImmersedTerrain
    std::string m_drag_weight{"fraction"};
    //! Terrain fraction at or above which ``center`` treats a cell as solid
    amrex::Real m_solid_threshold{0.5_rt};
    //! turbulence.model = Laminar
    bool m_is_laminar{false};
};

} // namespace kynema_sgf::pde::icns

#endif
