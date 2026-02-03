#include "amr-wind/turbulence/RANS/KLAxell.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/fvm/gradient.H"
#include "amr-wind/fvm/strainrate.H"
#include "amr-wind/turbulence/turb_utils.H"
#include "amr-wind/equation_systems/tke/TKE.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace amr_wind {
namespace turbulence {

template <typename Transport>
KLAxell<Transport>::KLAxell(CFDSim& sim)
    : TurbModelBase<Transport>(sim)
    , m_vel(sim.repo().get_field("velocity"))
    , m_turb_lscale(sim.repo().declare_field("turb_lscale", 1))
    , m_shear_prod(sim.repo().declare_field("shear_prod", 1))
    , m_buoy_prod(sim.repo().declare_field("buoy_prod", 1))
    , m_dissip(sim.repo().declare_field("dissipation", 1))
    , m_rho(sim.repo().get_field("density"))
    , m_temperature(sim.repo().get_field("temperature"))
{
    auto& tke_eqn =
        sim.pde_manager().register_transport_pde(pde::TKE::pde_name());
    m_tke = &(tke_eqn.fields().field);
    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) {
        amrex::Abort("KLAxell model only works with ABL physics");
    }
    {
        amrex::ParmParse pp("ABL");
        pp.get("surface_temp_flux", m_surf_flux);
        pp.query("meso_sponge_start", m_meso_sponge_start);
    }

    {
        amrex::ParmParse pp("incflo");
        pp.queryarr("gravity", m_gravity);
    }

    // TKE source term to be added to PDE
    turb_utils::inject_turbulence_src_terms(
        pde::TKE::pde_name(), {"KransAxell"});
}

template <typename Transport>
void KLAxell<Transport>::parse_model_coeffs()
{
    const std::string coeffs_dict = this->model_name() + "_coeffs";
    amrex::ParmParse pp(coeffs_dict);
    pp.query("Cmu", this->m_Cmu);
    pp.query("Cmu_prime", this->m_Cmu_prime);
    pp.query("Cb_stable", this->m_Cb_stable);
    pp.query("Cb_unstable", this->m_Cb_unstable);
    pp.query("prandtl", this->m_prandtl);
}

template <typename Transport>
TurbulenceModel::CoeffsDictType KLAxell<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{{
        {"Cmu", this->m_Cmu},
        {"Cmu_prime", this->m_Cmu_prime},
        {"Cb_stable", this->m_Cb_stable},
        {"Cb_unstable", this->m_Cb_unstable},
        {"prandtl", this->m_prandtl}}};
}

template <typename Transport>
void KLAxell<Transport>::update_turbulent_viscosity(
    const FieldState fstate, const DiffusionType /*unused*/)
{
    BL_PROFILE(
        "amr-wind::" + this->identifier() + "::update_turbulent_viscosity");

    auto gradT = (this->m_sim.repo()).create_scratch_field(3, 0);
    fvm::gradient(*gradT, m_temperature.state(fstate));

    const auto& vel = this->m_vel.state(fstate);
    fvm::strainrate(this->m_shear_prod, vel);

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const auto beta = (this->m_transport).beta();
    const amrex::Real Cmu = m_Cmu;
    const amrex::Real Cb_stable = m_Cb_stable;
    const amrex::Real Cb_unstable = m_Cb_unstable;
    const amrex::Real surf_flux = m_surf_flux;
    const auto tiny = std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real lengthscale_switch = m_meso_sponge_start;

    // Pre-compute powers!
    const amrex::Real powCmu3 = Cmu * Cmu * Cmu;
    const amrex::Real powCmu6 = powCmu3 * powCmu3;

    auto& mu_turb = this->mu_turb();
    const auto& den = this->m_rho.state(fstate);
    const auto& repo = mu_turb.repo();
    const auto& geom_vec = repo.mesh().Geom();
    const int nlevels = repo.num_active_levels();

    const amrex::Real Rtc = -1.0_rt;
    const amrex::Real Rtmin = -3.0_rt;
    const amrex::Real lambda = 30.0_rt;
    const amrex::Real kappa = 0.41_rt;

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];
        const auto& problo = repo.mesh().Geom(lev).ProbLoArray();
        const amrex::Real dz = geom.CellSize()[2];

        const auto& mu_arrs = mu_turb(lev).arrays();
        const auto& rho_arrs = den(lev).const_arrays();
        const auto& gradT_arrs = (*gradT)(lev).const_arrays();
        const auto& tlscale_arrs = (this->m_turb_lscale)(lev).arrays();
        const auto& tke_arrs = (*this->m_tke)(lev).arrays();
        const auto& buoy_prod_arrs = (this->m_buoy_prod)(lev).arrays();
        const auto& shear_prod_arrs = (this->m_shear_prod)(lev).arrays();
        const auto& beta_arrs = (*beta)(lev).const_arrays();

        // Add terrain components if any
        const bool has_terrain = this->m_sim.repo().int_field_exists("terrain_blank");
        if (has_terrain) {
            const auto* m_terrain_height =
                &this->m_sim.repo().get_field("terrain_height");
            const auto* m_terrain_blank =
                &this->m_sim.repo().get_int_field("terrain_blank");
            const auto& ht_arrs = (*m_terrain_height)(lev).const_arrays();
            const auto& blank_arrs = (*m_terrain_blank)(lev).const_arrays();
            amrex::ParallelFor(
                mu_turb(lev),
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                    // Register cache
                    const amrex::Real gradT0 = gradT_arrs[nbx](i, j, k, 0);
                    const amrex::Real gradT1 = gradT_arrs[nbx](i, j, k, 1);
                    const amrex::Real gradT2 = gradT_arrs[nbx](i, j, k, 2);
                    const amrex::Real beta_val = beta_arrs[nbx](i, j, k);
                    const amrex::Real stratification =
                        -(gradT0 * gravity[0] + gradT1 * gravity[1] + gradT2 * gravity[2]) * beta_val;
                    const amrex::Real z = amrex::max<amrex::Real>(
                        problo[2] + (k + 0.5_rt) * dz - ht_arrs[nbx](i, j, k),
                        0.5_rt * dz);
                    const amrex::Real lscale_s = (lambda * kappa * z) / (lambda + kappa * z);
                    const amrex::Real safe_strat = amrex::max<amrex::Real>(stratification, tiny);
                    const amrex::Real tke = tke_arrs[nbx](i, j, k);
                    const amrex::Real lscale_b = Cb_stable * amrex::Math::sqrt(tke / safe_strat);
                    const amrex::Real tke_sqrt = amrex::Math::sqrt(tke);
                    const amrex::Real tke_1p5 = tke * tke_sqrt;
                    amrex::Real& tlscale_val = tlscale_arrs[nbx](i, j, k);
                    const amrex::Real epsilon = powCmu3 * tke_1p5 / (tlscale_val + tiny);

                    const amrex::Real tke_over_epsilon = tke / (epsilon + tiny);
                    const amrex::Real Rt_uncorrected = tke_over_epsilon * tke_over_epsilon * stratification;
                    amrex::Real Rt = (Rt_uncorrected > Rtc)
                        ? Rt_uncorrected
                        : amrex::max<amrex::Real>(
                              Rt_uncorrected,
                              Rt_uncorrected - ((Rt_uncorrected - Rtc) * (Rt_uncorrected - Rtc)) /
                                (Rt_uncorrected + Rtmin - 2.0_rt * Rtc));
                    amrex::Real new_tlscale;
                    if (stratification > 0) {
                        const amrex::Real mixed = (lscale_s * lscale_b);
                        new_tlscale = amrex::Math::sqrt(
                            (mixed * mixed) / (lscale_s * lscale_s + lscale_b * lscale_b));
                        new_tlscale = amrex::min<amrex::Real>(
                            new_tlscale,
                            amrex::Math::sqrt(Cmu * tke / stratification));
                    } else {
                        new_tlscale = lscale_s * amrex::Math::sqrt(
                            amrex::max<amrex::Real>(
                                0.0_rt,
                                1.0_rt - powCmu6 / (Cb_unstable * Cb_unstable) * Rt));
                    }
                    if ((amrex::Math::abs(surf_flux) < 1.0e-5_rt) && (z <= lengthscale_switch)) {
                        new_tlscale = lscale_s;
                        Rt = 0.0_rt;
                    }
                    tlscale_val = new_tlscale;

                    const amrex::Real Rt2 = Rt * Rt;
                    const amrex::Real Cmu_Rt = (Cmu + 0.108_rt * Rt) /
                        (1.0_rt + 0.308_rt * Rt + 0.00837_rt * Rt2);
                    const amrex::Real mut_val = rho_arrs[nbx](i, j, k) * Cmu_Rt * new_tlscale * tke_sqrt
                        * (1.0_rt - blank_arrs[nbx](i, j, k));
                    mu_arrs[nbx](i, j, k) = mut_val;
                    const amrex::Real Cmu_prime_Rt = Cmu / (1.0_rt + 0.277_rt * Rt);
                    const amrex::Real muPrime = rho_arrs[nbx](i, j, k) * Cmu_prime_Rt * new_tlscale * tke_sqrt
                        * (1.0_rt - blank_arrs[nbx](i, j, k));
                    buoy_prod_arrs[nbx](i, j, k) = -muPrime * stratification;
                    shear_prod_arrs[nbx](i, j, k) *= shear_prod_arrs[nbx](i, j, k) * mut_val;
                });
        } else {
            // No terrain version
            amrex::ParallelFor(
                mu_turb(lev),
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                    // Register cache
                    const amrex::Real gradT0 = gradT_arrs[nbx](i, j, k, 0);
                    const amrex::Real gradT1 = gradT_arrs[nbx](i, j, k, 1);
                    const amrex::Real gradT2 = gradT_arrs[nbx](i, j, k, 2);
                    const amrex::Real beta_val = beta_arrs[nbx](i, j, k);
                    const amrex::Real stratification =
                        -(gradT0 * gravity[0] + gradT1 * gravity[1] + gradT2 * gravity[2]) * beta_val;
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dz;
                    const amrex::Real lscale_s = (lambda * kappa * z) / (lambda + kappa * z);
                    const amrex::Real tke = tke_arrs[nbx](i, j, k);
                    const amrex::Real safe_strat = amrex::max<amrex::Real>(stratification, tiny);
                    const amrex::Real lscale_b = Cb_stable * amrex::Math::sqrt(tke / safe_strat);
                    const amrex::Real tke_sqrt = amrex::Math::sqrt(tke);
                    const amrex::Real tke_1p5 = tke * tke_sqrt;
                    amrex::Real& tlscale_val = tlscale_arrs[nbx](i, j, k);
                    const amrex::Real epsilon = powCmu3 * tke_1p5 / (tlscale_val + tiny);

                    const amrex::Real tke_over_epsilon = tke / (epsilon + tiny);
                    const amrex::Real Rt_uncorrected = tke_over_epsilon * tke_over_epsilon * stratification;
                    amrex::Real Rt = (Rt_uncorrected > Rtc)
                        ? Rt_uncorrected
                        : amrex::max<amrex::Real>(
                              Rt_uncorrected,
                              Rt_uncorrected - ((Rt_uncorrected - Rtc) * (Rt_uncorrected - Rtc)) /
                                (Rt_uncorrected + Rtmin - 2.0_rt * Rtc));
                    amrex::Real new_tlscale;
                    if (stratification > 0) {
                        const amrex::Real mixed = (lscale_s * lscale_b);
                        new_tlscale = amrex::Math::sqrt(
                            (mixed * mixed) / (lscale_s * lscale_s + lscale_b * lscale_b));
                        new_tlscale = amrex::min<amrex::Real>(
                            new_tlscale,
                            amrex::Math::sqrt(Cmu * tke / stratification));
                    } else {
                        new_tlscale = lscale_s * amrex::Math::sqrt(
                            amrex::max<amrex::Real>(
                                0.0_rt,
                                1.0_rt - powCmu6 / (Cb_unstable * Cb_unstable) * Rt));
                    }
                    if ((amrex::Math::abs(surf_flux) < 1.0e-5_rt) && (z <= lengthscale_switch)) {
                        new_tlscale = lscale_s;
                        Rt = 0.0_rt;
                    }
                    tlscale_val = new_tlscale;

                    // Cmu_Rt and Cmu_prime_Rt
                    const amrex::Real Rt2 = Rt * Rt;
                    const amrex::Real Cmu_Rt = (Cmu + 0.108_rt * Rt) /
                        (1.0_rt + 0.308_rt * Rt + 0.00837_rt * Rt2);
                    const amrex::Real mut_val = rho_arrs[nbx](i, j, k) * Cmu_Rt * new_tlscale * tke_sqrt;
                    mu_arrs[nbx](i, j, k) = mut_val;
                    const amrex::Real Cmu_prime_Rt = Cmu / (1.0_rt + 0.277_rt * Rt);
                    const amrex::Real muPrime = rho_arrs[nbx](i, j, k) * Cmu_prime_Rt * new_tlscale * tke_sqrt;
                    buoy_prod_arrs[nbx](i, j, k) = -muPrime * stratification;
                    shear_prod_arrs[nbx](i, j, k) *= shear_prod_arrs[nbx](i, j, k) * mut_val;
                });
        }
    }
    amrex::Gpu::streamSynchronize();

    mu_turb.fillpatch(this->m_sim.time().current_time());
}

template <typename Transport>
void KLAxell<Transport>::update_alphaeff(Field& alphaeff)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_alphaeff");
    auto lam_alpha = (this->m_transport).alpha();
    auto& mu_turb = this->m_mu_turb;
    auto& repo = mu_turb.repo();
    auto gradT = (this->m_sim.repo()).create_scratch_field(3, 0);
    fvm::gradient(*gradT, m_temperature);
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const auto beta = (this->m_transport).beta();
    const amrex::Real Cmu = m_Cmu;
    const amrex::Real powCmu3 = Cmu * Cmu * Cmu;

    const int nlevels = repo.num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& muturb_arrs = mu_turb(lev).arrays();
        const auto& alphaeff_arrs = alphaeff(lev).arrays();
        const auto& lam_diff_arrs = (*lam_alpha)(lev).arrays();
        const auto& tke_arrs = (*this->m_tke)(lev).arrays();
        const auto& gradT_arrs = (*gradT)(lev).const_arrays();
        const auto& tlscale_arrs = (this->m_turb_lscale)(lev).arrays();
        const auto& beta_arrs = (*beta)(lev).const_arrays();
        const amrex::Real Rtc = -1.0_rt;
        const amrex::Real Rtmin = -3.0_rt;
        amrex::ParallelFor(
            mu_turb(lev),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                const amrex::Real gradT0 = gradT_arrs[nbx](i, j, k, 0);
                const amrex::Real gradT1 = gradT_arrs[nbx](i, j, k, 1);
                const amrex::Real gradT2 = gradT_arrs[nbx](i, j, k, 2);
                const amrex::Real beta_val = beta_arrs[nbx](i, j, k);
                const amrex::Real stratification =
                    -(gradT0 * gravity[0] + gradT1 * gravity[1] + gradT2 * gravity[2]) * beta_val;
                const amrex::Real tke = tke_arrs[nbx](i, j, k);
                const amrex::Real tke_sqrt = amrex::Math::sqrt(tke);
                const amrex::Real tke_1p5 = tke * tke_sqrt;
                const amrex::Real tlscale = tlscale_arrs[nbx](i, j, k);
                const amrex::Real epsilon = powCmu3 * tke_1p5 / (tlscale + 1.e-16_rt);
                const amrex::Real tke_over_epsilon = tke / (epsilon + 1.e-16_rt);
                const amrex::Real Rt_uncorrected =
                    tke_over_epsilon * tke_over_epsilon * stratification;
                amrex::Real Rt = (Rt_uncorrected > Rtc)
                    ? Rt_uncorrected
                    : amrex::max<amrex::Real>(
                          Rt_uncorrected,
                          Rt_uncorrected - ((Rt_uncorrected - Rtc) * (Rt_uncorrected - Rtc)) /
                              (Rt_uncorrected + Rtmin - 2.0_rt * Rtc));
                const amrex::Real prandtlRt = (1.0_rt + 0.193_rt * Rt) / (1.0_rt + 0.0302_rt * Rt);
                alphaeff_arrs[nbx](i, j, k) =
                    lam_diff_arrs[nbx](i, j, k) + muturb_arrs[nbx](i, j, k) / prandtlRt;
            });
    }
    amrex::Gpu::streamSynchronize();

    alphaeff.fillpatch(this->m_sim.time().current_time());
}

template <typename Transport>
void KLAxell<Transport>::update_scalar_diff(
    Field& deff, const std::string& name)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_scalar_diff");

    if (name == pde::TKE::var_name()) {
        auto& mu_turb = this->mu_turb();
        deff.setVal(0.0_rt);
        field_ops::saxpy(
            deff, 2.0_rt, mu_turb, 0, 0, deff.num_comp(), deff.num_grow());
    } else {
        amrex::Abort(
            "KLAxell:update_scalar_diff not implemented for field " + name);
    }
}

template <typename Transport>
void KLAxell<Transport>::post_advance_work()
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::post_advance_work");
}

} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(KLAxell);

} // namespace amr_wind
