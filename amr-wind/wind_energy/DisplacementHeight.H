#ifndef DISPLACEMENTHEIGHT_H
#define DISPLACEMENTHEIGHT_H

#include <cmath>
#include "AMReX_Gpu.H"
#include "AMReX_REAL.H"
#include "amr-wind/utilities/trig_ops.H"

using namespace amrex::literals;

/** Helpers for the zero-plane displacement height (d) used by canopy/forest
 *  surface-layer models.
 *  \ingroup we_abl
 *
 *  With a displacement height the log law becomes
 *
 *      U(z) = (u_star / kappa) * [ ln((z - d) / z0) - psi_m((z - d) / L) ]
 *
 *  The displacement field is provided per cell by TerrainDrag
 *  (terrain_displacement) and is only used with ABL.wall_het_model = mol.
 */
namespace amr_wind::displacement {

//! Name of the cell-centered displacement height field
inline constexpr const char* field_name = "terrain_displacement";

/** Effective height above the zero-plane displacement.
 *
 *  The result is clipped from below at e*z0 so that ln((z-d)/z0) >= 1. This
 *  avoids singular/negative log-law arguments when the canopy is taller than
 *  the wall-model sampling height (z <= d + z0).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
displaced_height(const amrex::Real z, const amrex::Real d, const amrex::Real z0)
{
    constexpr amrex::Real euler = 2.718281828459045_rt;
    return amrex::max<amrex::Real>(z - d, euler * z0);
}

/** Device-callable Businger-Dyer momentum stability correction.
 *  Identical to MOData::calc_psi_m, which is host-only.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real psi_m(
    const amrex::Real zeta, const amrex::Real beta_m, const amrex::Real gamma_m)
{
    if (zeta > 0) {
        return -gamma_m * zeta;
    }
    const amrex::Real x = std::sqrt(std::sqrt(1.0_rt - (beta_m * zeta)));
    return (2.0_rt * std::log(0.5_rt * (1.0_rt + x))) +
           std::log(0.5_rt * (1.0_rt + x * x)) - (2.0_rt * std::atan(x)) +
           amr_wind::utils::half_pi();
}

} // namespace amr_wind::displacement

#endif /* DISPLACEMENTHEIGHT_H */
