#include <cmath>
#include "amr-wind/turbulence/LES/Kosovic.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/fvm/nonLinearSum.H"
#include "amr-wind/fvm/strainrate.H"
#include "amr-wind/fvm/divergence.H"
#include "AMReX_REAL.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/wind_energy/ABL.H"

using namespace amrex::literals;

namespace amr_wind {
namespace turbulence {

template <typename Transport>
// cppcheck-suppress uninitMemberVar
Kosovic<Transport>::Kosovic(CFDSim& sim)
    : TurbModelBase<Transport>(sim)
    , m_vel(sim.repo().get_field("velocity"))
    , m_rho(sim.repo().get_field("density"))
    , m_Nij(sim.repo().declare_field("Nij", 9, 1, 1))
    , m_divNij(sim.repo().declare_field("divNij", 3))
{
    amrex::ParmParse pp("Kosovic");
    pp.query("Cb", m_Cb);
    m_Cs = std::sqrt(
        8.0_rt * (1.0_rt + m_Cb) /
        (27.0_rt * static_cast<amrex::Real>(M_PI) *
         static_cast<amrex::Real>(M_PI)));
    m_C1 = std::sqrt(960.0_rt) * m_Cb / (7.0_rt * (1.0_rt + m_Cb) * m_Sk);
    m_C2 = m_C1;
    pp.query("surfaceRANS", m_surfaceRANS);
    if (m_surfaceRANS) {
        m_surfaceFactor = 1;
        pp.query("switchLoc", m_switchLoc);
        pp.query("surfaceRANSExp", m_surfaceRANSExp);
    } else {
        m_surfaceFactor = 0;
    }
    pp.query("writeTerms", m_writeTerms);
    if (m_writeTerms) {
        this->m_sim.io_manager().register_io_var("Nij");
        this->m_sim.io_manager().register_io_var("divNij");
    }
    pp.query("LESOff", m_LESTurnOff);
    amrex::ParmParse pp_abl("ABL");
    pp_abl.query("wall_het_model", m_wall_het_model);
    pp_abl.query("monin_obukhov_length", m_monin_obukhov_length);
    pp_abl.query("kappa", m_kappa);
    pp_abl.query("mo_gamma_m", m_gamma_m);
    pp_abl.query("mo_beta_m", m_beta_m);
    pp_abl.query("surface_roughness_z0", m_surface_roughness_z0);
}

template <typename Transport>
void Kosovic<Transport>::update_turbulent_viscosity(
    const FieldState fstate, const DiffusionType /*unused*/)
{
    BL_PROFILE(
        "amr-wind::" + this->identifier() + "::update_turbulent_viscosity");

    auto& mu_turb = this->mu_turb();
    const auto& repo = mu_turb.repo();
    const auto& vel = m_vel.state(fstate);
    const auto& den = m_rho.state(fstate);
    const auto& geom_vec = repo.mesh().Geom();
    const amrex::Real Cs_sqr = this->m_Cs * this->m_Cs;

    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");
    const auto* m_terrain_blank =
        has_terrain ? &this->m_sim.repo().get_int_field("terrain_blank")
                    : nullptr;
    const auto* m_terrain_drag =
        has_terrain ? &this->m_sim.repo().get_int_field("terrain_drag")
                    : nullptr;
    const auto* m_terrain_height =
        has_terrain ? &this->m_sim.repo().get_field("terrain_height") : nullptr;
    const auto* m_terrain_z0 =
        has_terrain ? &this->m_sim.repo().get_field("terrainz0") : nullptr;

    // Populate strainrate into the turbulent viscosity arrays to avoid creating
    // a temporary buffer
    fvm::strainrate(mu_turb, vel);
    // Non-linear component Nij is computed here and goes into Body Forcing
    fvm::nonlinearsum(m_Nij, vel);
    fvm::divergence(m_divNij, m_Nij);

    const int nlevels = repo.num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];
        const auto& problo = repo.mesh().Geom(lev).ProbLoArray();

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);
        const amrex::Real ds_sqr = ds * ds;
        const amrex::Real smag_factor = Cs_sqr * ds_sqr;

        // Hoist invariants and reuse constants
        const amrex::Real inv_dz = 1.0_rt / dz;
        const amrex::Real locLESTurnOff = m_LESTurnOff;
        const amrex::Real locSwitchLoc = m_switchLoc;
        const amrex::Real locSurfaceRANSExp = m_surfaceRANSExp;
        const amrex::Real locSurfaceFactor = m_surfaceFactor;
        const amrex::Real locC1 = m_C1;
        const amrex::Real monin_obukhov_length = m_monin_obukhov_length;
        const amrex::Real kappa = m_kappa;
        const amrex::Real surface_roughness_z0 = m_surface_roughness_z0;

        // Precompute denominator for log-law when terrain roughness is uniform
        const amrex::Real base_log_denom =
            std::log(1.5_rt * dz / surface_roughness_z0) -
            ((m_wall_het_model == "mol")
                 ? MOData::calc_psi_m(
                       1.5_rt * dz / monin_obukhov_length, m_beta_m, m_gamma_m)
                 : 0.0_rt);

        const auto& mu_arrs = mu_turb(lev).arrays();
        const auto& rho_arrs = den(lev).const_arrays();
        const auto& vel_arrs = vel(lev).const_arrays();
        const auto& divNij_arrs = (this->m_divNij)(lev).arrays();

        const auto& blank_arrs = has_terrain
                                     ? (*m_terrain_blank)(lev).const_arrays()
                                     : amrex::MultiArray4<const int>();
        const auto& drag_arrs = has_terrain
                                    ? (*m_terrain_drag)(lev).const_arrays()
                                    : amrex::MultiArray4<const int>();
        const auto& height_arrs = has_terrain
                                      ? (*m_terrain_height)(lev).const_arrays()
                                      : amrex::MultiArray4<const amrex::Real>();
        const auto& z0_arrs = has_terrain
                                  ? (*m_terrain_z0)(lev).const_arrays()
                                  : amrex::MultiArray4<const amrex::Real>();

        // Precompute neutral/unstable neighbor term once per level
        const amrex::Real non_neutral_neighbour =
            (m_wall_het_model == "mol")
                ? MOData::calc_psi_m(
                      1.5_rt * dz / monin_obukhov_length, m_beta_m, m_gamma_m)
                : 0.0_rt;

        // Precompute stress factor scale constants
        const amrex::Real smag_stress_scale = smag_factor * 0.25_rt * locC1;

        amrex::ParallelFor(
            mu_turb(lev),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                const amrex::Real rho = rho_arrs[nbx](i, j, k);

                // Height above ground (terrain-aware)
                amrex::Real x3 = problo[2] + (k + 0.5_rt) * dz;
                if (has_terrain) {
                    x3 = amrex::max<amrex::Real>(
                        x3 - height_arrs[nbx](i, j, k, 0), 0.5_rt * dz);
                }

                // Precompute exponential switches
                const amrex::Real fmu = std::exp(-x3 / locSwitchLoc);
                const amrex::Real turnOff = std::exp(-x3 / locLESTurnOff);

                // Stability function phiM
                amrex::Real phiM;
                if (monin_obukhov_length < 0.0_rt) {
                    // phiM = (1 - 16*z/L)^(-1/4) = 1 / sqrt(sqrt(1 - 16*z/L))
                    const amrex::Real t =
                        1.0_rt - 16.0_rt * x3 / monin_obukhov_length;
                    // Keep original behavior; no clamping to avoid changing physics
                    phiM = 1.0_rt / std::sqrt(std::sqrt(t));
                } else {
                    phiM = 1.0_rt + 5.0_rt * x3 / monin_obukhov_length;
                }

                // Terrain-aware wall distance
                const amrex::Real wall_distance =
                    has_terrain
                        ? amrex::max<amrex::Real>(
                              (k + 1) * dz - height_arrs[nbx](i, j, k, 0), dz)
                        : (k + 1) * dz;

                // RANS length scale: (0.41*wall_distance/phiM)^2
                const amrex::Real tmp = 0.41_rt * wall_distance / phiM;
                const amrex::Real ransL = tmp * tmp;

                // Surface blending powers (compute once)
                const amrex::Real fmu_pow = std::pow(fmu, locSurfaceRANSExp);
                const amrex::Real one_minus_fmu_pow =
                    std::pow(1.0_rt - fmu, locSurfaceRANSExp);

                // Viscosity blend
                const amrex::Real viscosityScale =
                    locSurfaceFactor *
                        (one_minus_fmu_pow * smag_factor +
                         fmu_pow * ransL) +
                    (1.0_rt - locSurfaceFactor) * smag_factor;

                // Terrain blanking factor
                const amrex::Real blankTerrain =
                    has_terrain ? 1 - blank_arrs[nbx](i, j, k, 0) : 1.0_rt;

                // Apply baseline turbulent viscosity
                mu_arrs[nbx](i, j, k) *=
                    rho * viscosityScale * turnOff * blankTerrain;

                // Terrain drag flag (0 or 1). If no drag, skip expensive log-law work.
                const amrex::Real drag =
                    has_terrain ? drag_arrs[nbx](i, j, k, 0) : 0.0_rt;

                if (drag > 0.0_rt) {
                    // log-law friction velocity ustar
                    const amrex::Real ux_above = vel_arrs[nbx](i, j, k + 1, 0);
                    const amrex::Real uy_above = vel_arrs[nbx](i, j, k + 1, 1);
                    const amrex::Real m_above =
                        std::sqrt(ux_above * ux_above + uy_above * uy_above);

                    const amrex::Real local_z0 =
                        has_terrain
                            ? amrex::max<amrex::Real>(
                                  z0_arrs[nbx](i, j, k, 0), 1.0e-4_rt)
                            : surface_roughness_z0;

                    const amrex::Real denom =
                        (has_terrain
                             ? (std::log(1.5_rt * dz / local_z0) -
                                non_neutral_neighbour)
                             : base_log_denom);

                    const amrex::Real ustar = m_above * kappa / denom;

                    // Shear magnitude gradient dM/dz
                    const amrex::Real ux0 = vel_arrs[nbx](i, j, k, 0);
                    const amrex::Real uy0 = vel_arrs[nbx](i, j, k, 1);
                    const amrex::Real m0 = std::sqrt(ux0 * ux0 + uy0 * uy0);

                    const amrex::Real uxm1 = vel_arrs[nbx](i, j, k - 1, 0);
                    const amrex::Real uym1 = vel_arrs[nbx](i, j, k - 1, 1);
                    const amrex::Real mm1 = std::sqrt(uxm1 * uxm1 + uym1 * uym1);

                    const amrex::Real dMdz =
                        amrex::max<amrex::Real>((m0 - mm1) * inv_dz, 0.01_rt);

                    const amrex::Real mut_loglaw =
                        2.0_rt * ustar * ustar * rho / dMdz;

                    // Blend baseline mu_turb with log-law contribution
                    mu_arrs[nbx](i, j, k) =
                        mu_arrs[nbx](i, j, k) * (1.0_rt - drag) +
                        drag * mut_loglaw;
                }

                // Stress scaling (compute once, reuse for all components)
                const amrex::Real stressScale =
                    locSurfaceFactor *
                        (one_minus_fmu_pow * smag_stress_scale +
                         fmu_pow * ransL) +
                    (1.0_rt - locSurfaceFactor) * smag_stress_scale;

                const amrex::Real stressFactor =
                    rho * stressScale * turnOff * blankTerrain;

                divNij_arrs[nbx](i, j, k, 0) *= stressFactor;
                divNij_arrs[nbx](i, j, k, 1) *= stressFactor;
                divNij_arrs[nbx](i, j, k, 2) *= stressFactor;
            });
    }
    amrex::Gpu::streamSynchronize();

    mu_turb.fillpatch(this->m_sim.time().current_time());
}

template <typename Transport>
void Kosovic<Transport>::update_alphaeff(Field& alphaeff)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_alphaeff");

    auto lam_alpha = (this->m_transport).alpha();
    auto& mu_turb = this->m_mu_turb;
    auto& repo = mu_turb.repo();

    // Hoist coefficient outside the kernel
    const amrex::Real muCoeff = (m_monin_obukhov_length < 0) ? 3.0_rt : 1.0_rt;

    const int nlevels = repo.num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& muturb_arrs = mu_turb(lev).const_arrays();
        const auto& alphaeff_arrs = alphaeff(lev).arrays();
        const auto& lam_diff_arrs = (*lam_alpha)(lev).const_arrays();

        amrex::ParallelFor(
            mu_turb(lev),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                alphaeff_arrs[nbx](i, j, k) =
                    lam_diff_arrs[nbx](i, j, k) +
                    muCoeff * muturb_arrs[nbx](i, j, k);
            });
    }
    amrex::Gpu::streamSynchronize();

    alphaeff.fillpatch(this->m_sim.time().current_time());
}

template <typename Transport>
void Kosovic<Transport>::parse_model_coeffs()
{
    const std::string coeffs_dict = this->model_name() + "_coeffs";
    amrex::ParmParse pp(coeffs_dict);
    pp.query("Cs", this->m_Cs);
}

template <typename Transport>
TurbulenceModel::CoeffsDictType Kosovic<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{{"Cb", this->m_Cb}};
}

} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(Kosovic);

} // namespace amr_wind
