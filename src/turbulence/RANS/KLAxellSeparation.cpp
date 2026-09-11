#include <limits>

#include "src/turbulence/RANS/KLAxellSeparation.H"
#include "src/equation_systems/PDEBase.H"
#include "src/turbulence/TurbModelDefs.H"
#include "src/fvm/gradient.H"
#include "src/fvm/strainrate.H"
#include "src/utilities/math_ops.H"
#include "AMReX_ParmParse.H"

using namespace amrex::literals;

namespace kynema_sgf {
namespace turbulence {

template <typename Transport>
KLAxellSeparation<Transport>::KLAxellSeparation(CFDSim& sim)
    : KLAxell<Transport>(sim)
{
    amrex::ParmParse pp("KLAxellSeparation");
    pp.query("pressure_gradient_sensor", m_use_pressure_gradient_sensor);
    pp.query("realizable_cmu", m_use_realizable_cmu);
    pp.query("sensor_source", m_sensor_source);
    if (m_sensor_source != "pressure" && m_sensor_source != "velocity") {
        amrex::Abort(
            "KLAxellSeparation.sensor_source must be pressure or velocity");
    }
    if (m_use_realizable_cmu && !m_use_pressure_gradient_sensor) {
        amrex::Abort(
            "KLAxellSeparation.realizable_cmu requires "
            "KLAxellSeparation.pressure_gradient_sensor = true");
    }
    if (m_use_pressure_gradient_sensor) {
        m_pressure_gradient_sensor =
            &sim.repo().declare_field("pressure_gradient_sensor", 1);
    }
}

template <typename Transport>
KLAxellSeparation<Transport>::~KLAxellSeparation() = default;

template <typename Transport>
void KLAxellSeparation<Transport>::parse_model_coeffs()
{
    KLAxell<Transport>::parse_model_coeffs();
    const std::string coeffs_dict = this->model_name() + "_coeffs";
    amrex::ParmParse pp(coeffs_dict);
    pp.query("sensor_velocity_weight", m_sensor_velocity_weight);
    pp.query("sensor_threshold", m_sensor_threshold);
    pp.query("realizable_cmu_strength", m_realizable_cmu_strength);
    if (m_sensor_threshold <= 0.0_rt) {
        amrex::Abort(
            "KLAxellSeparation_coeffs.sensor_threshold must be positive");
    }
}

template <typename Transport>
TurbulenceModel::CoeffsDictType
KLAxellSeparation<Transport>::model_coeffs() const
{
    auto coeffs = KLAxell<Transport>::model_coeffs();
    coeffs["sensor_velocity_weight"] = m_sensor_velocity_weight;
    coeffs["sensor_threshold"] = m_sensor_threshold;
    coeffs["realizable_cmu_strength"] = m_realizable_cmu_strength;
    return coeffs;
}

template <typename Transport>
void KLAxellSeparation<Transport>::post_init_actions()
{
    KLAxell<Transport>::post_init_actions();
    m_geometry = this->m_sim.repo().create_scratch_field(
        klaxell_separation::geom_ncomp, 1);
    if (m_use_pressure_gradient_sensor) {
        m_pressure_gradient_sensor->setVal(0.0_rt);
    }
    if (m_use_realizable_cmu) {
        m_strain = this->m_sim.repo().create_scratch_field(1, 0);
    }
    if (m_use_pressure_gradient_sensor && (m_sensor_source == "velocity")) {
        m_grad_vel = this->m_sim.repo().create_scratch_field(
            AMREX_SPACEDIM * AMREX_SPACEDIM, 0);
    }
}

template <typename Transport>
void KLAxellSeparation<Transport>::post_regrid_actions()
{
    KLAxell<Transport>::post_regrid_actions();
    m_geometry = this->m_sim.repo().create_scratch_field(
        klaxell_separation::geom_ncomp, 1);
    if (m_use_pressure_gradient_sensor) {
        m_pressure_gradient_sensor->setVal(0.0_rt);
    }
    if (m_use_realizable_cmu) {
        m_strain = this->m_sim.repo().create_scratch_field(1, 0);
    }
    if (m_use_pressure_gradient_sensor && (m_sensor_source == "velocity")) {
        m_grad_vel = this->m_sim.repo().create_scratch_field(
            AMREX_SPACEDIM * AMREX_SPACEDIM, 0);
    }
}

template <typename Transport>
void KLAxellSeparation<Transport>::update_turbulent_viscosity(
    const FieldState fstate, const DiffusionType /*unused*/)
{
    BL_PROFILE(
        "kynema-sgf::" + this->identifier() + "::update_turbulent_viscosity");

    fvm::gradient(*this->m_gradT, this->m_temperature.state(fstate));

    const auto& vel = this->m_vel.state(fstate);
    fvm::strainrate(this->m_shear_prod, vel);
    const bool use_velocity_sensor =
        m_use_pressure_gradient_sensor && (m_sensor_source == "velocity");
    if (use_velocity_sensor) {
        fvm::gradient(*m_grad_vel, vel);
    }

    const auto beta = (this->m_transport).beta();
    auto& mu_turb = this->mu_turb();
    const int nlevels = mu_turb.repo().num_active_levels();
    if (m_use_realizable_cmu) {
        // The closure turns the strain rate into the shear production; keep
        // the strain rate for the limiter
        for (int lev = 0; lev < nlevels; ++lev) {
            amrex::MultiFab::Copy(
                (*m_strain)(lev), (this->m_shear_prod)(lev), 0, 0, 1, 0);
        }
    }
    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");
    for (int lev = 0; lev < nlevels; ++lev) {
        if (has_terrain) {
            terrain_drag_geometry(lev);
        } else {
            flat_geometry(lev);
        }
        closure(lev, fstate, *beta);
        if (use_velocity_sensor) {
            velocity_sensor(lev, fstate);
        } else if (m_use_pressure_gradient_sensor) {
            pressure_gradient_sensor(lev, fstate);
        }
        if (m_use_realizable_cmu) {
            realizable_cmu(lev);
        }
    }
    amrex::Gpu::streamSynchronize();

    mu_turb.fillpatch(this->m_sim.time().current_time());
}

template <typename Transport>
void KLAxellSeparation<Transport>::flat_geometry(const int lev)
{
    const auto& geom = this->m_sim.repo().mesh().Geom(lev);
    const auto& problo = geom.ProbLoArray();
    const amrex::Real dz = geom.CellSize()[2];
    const auto& geom_arrs = (*m_geometry)(lev).arrays();

    amrex::ParallelFor(
        (*m_geometry)(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            geom_arrs[nbx](i, j, k, klaxell_separation::geom_height) =
                problo[2] + ((k + 0.5_rt) * dz);
            geom_arrs[nbx](i, j, k, klaxell_separation::geom_fluid_weight) =
                1.0_rt;
        });
}

template <typename Transport>
void KLAxellSeparation<Transport>::terrain_drag_geometry(const int lev)
{
    const auto& geom = this->m_sim.repo().mesh().Geom(lev);
    const auto& problo = geom.ProbLoArray();
    const amrex::Real dz = geom.CellSize()[2];
    const auto& geom_arrs = (*m_geometry)(lev).arrays();
    const auto& ht_arrs =
        this->m_sim.repo().get_field("terrain_height")(lev).const_arrays();
    const auto& blank_arrs =
        this->m_sim.repo().get_int_field("terrain_blank")(lev).const_arrays();

    amrex::ParallelFor(
        (*m_geometry)(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            geom_arrs[nbx](i, j, k, klaxell_separation::geom_height) =
                amrex::max<amrex::Real>(
                    problo[2] + ((k + 0.5_rt) * dz) - ht_arrs[nbx](i, j, k),
                    0.5_rt * dz);
            geom_arrs[nbx](i, j, k, klaxell_separation::geom_fluid_weight) =
                1.0_rt - static_cast<amrex::Real>(blank_arrs[nbx](i, j, k));
        });
}

template <typename Transport>
void KLAxellSeparation<Transport>::closure(
    const int lev, const FieldState fstate, const ScratchField& beta)
{
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        this->m_gravity[0], this->m_gravity[1], this->m_gravity[2]};
    const amrex::Real Cmu = this->m_Cmu;
    const amrex::Real Cb_stable = this->m_Cb_stable;
    const amrex::Real Cb_unstable = this->m_Cb_unstable;
    const amrex::Real Rtc = -1.0_rt;
    const amrex::Real Rtmin = -3.0_rt;
    const amrex::Real lambda = 30.0_rt;
    const amrex::Real kappa = 0.41_rt;
    const amrex::Real surf_flux = this->m_surf_flux;
    const auto tiny = std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real lengthscale_switch = this->m_meso_sponge_start;

    auto& mu_turb = this->mu_turb();
    const auto& den = this->m_rho.state(fstate);
    const auto& mu_arrs = mu_turb(lev).arrays();
    const auto& rho_arrs = den(lev).const_arrays();
    const auto& gradT_arrs = (*this->m_gradT)(lev).const_arrays();
    const auto& tlscale_arrs = (this->m_turb_lscale)(lev).arrays();
    const auto& tke_arrs = (*this->m_tke)(lev).const_arrays();
    const auto& buoy_prod_arrs = (this->m_buoy_prod)(lev).arrays();
    const auto& shear_prod_arrs = (this->m_shear_prod)(lev).arrays();
    const auto& beta_arrs = beta(lev).const_arrays();
    const auto& geom_arrs = (*m_geometry)(lev).const_arrays();

    // Same operations, in the same order, as the KLAxell kernels so that the
    // model reproduces KLAxell when no treatment is enabled
    amrex::ParallelFor(
        mu_turb(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            const amrex::Real z =
                geom_arrs[nbx](i, j, k, klaxell_separation::geom_height);
            const amrex::Real fluid_weight =
                geom_arrs[nbx](i, j, k, klaxell_separation::geom_fluid_weight);
            const amrex::Real stratification =
                -((gradT_arrs[nbx](i, j, k, 0) * gravity[0]) +
                  (gradT_arrs[nbx](i, j, k, 1) * gravity[1]) +
                  (gradT_arrs[nbx](i, j, k, 2) * gravity[2])) *
                beta_arrs[nbx](i, j, k);
            const amrex::Real lscale_s =
                (lambda * kappa * z) / (lambda + (kappa * z));
            const amrex::Real lscale_b =
                Cb_stable * std::sqrt(
                                tke_arrs[nbx](i, j, k) /
                                amrex::max<amrex::Real>(stratification, tiny));
            const amrex::Real epsilon =
                utils::powi(Cmu, 3) * std::pow(tke_arrs[nbx](i, j, k), 1.5_rt) /
                (tlscale_arrs[nbx](i, j, k) + tiny);
            amrex::Real Rt = utils::powi(tke_arrs[nbx](i, j, k) / epsilon, 2) *
                             stratification;
            Rt = (Rt > Rtc) ? Rt
                            : amrex::max<amrex::Real>(
                                  Rt, Rt - (utils::powi(Rt - Rtc, 2) /
                                            (Rt + Rtmin - (2.0_rt * Rtc))));
            tlscale_arrs[nbx](i, j, k) =
                (stratification > 0)
                    ? std::sqrt(
                          utils::powi(lscale_s * lscale_b, 2) /
                          (utils::powi(lscale_s, 2) + utils::powi(lscale_b, 2)))
                    : lscale_s *
                          std::sqrt(
                              1.0_rt - (utils::powi(Cmu, 6) *
                                        utils::powi(Cb_unstable, -2) * Rt));
            tlscale_arrs[nbx](i, j, k) =
                (stratification > 0)
                    ? amrex::min<amrex::Real>(
                          tlscale_arrs[nbx](i, j, k),
                          std::sqrt(
                              Cmu * tke_arrs[nbx](i, j, k) / stratification))
                    : tlscale_arrs[nbx](i, j, k);
            tlscale_arrs[nbx](i, j, k) =
                (std::abs(surf_flux) < 1.0e-5_rt && z <= lengthscale_switch)
                    ? lscale_s
                    : tlscale_arrs[nbx](i, j, k);
            Rt = (std::abs(surf_flux) < 1.0e-5_rt && z <= lengthscale_switch)
                     ? 0.0_rt
                     : Rt;
            const amrex::Real Cmu_Rt =
                (Cmu + (0.108_rt * Rt)) /
                (1.0_rt + (0.308_rt * Rt) + (0.00837_rt * utils::powi(Rt, 2)));
            mu_arrs[nbx](i, j, k) =
                rho_arrs[nbx](i, j, k) * Cmu_Rt * tlscale_arrs[nbx](i, j, k) *
                std::sqrt(tke_arrs[nbx](i, j, k)) * fluid_weight;
            const amrex::Real Cmu_prime_Rt = Cmu / (1.0_rt + (0.277_rt * Rt));
            const amrex::Real muPrime = rho_arrs[nbx](i, j, k) * Cmu_prime_Rt *
                                        tlscale_arrs[nbx](i, j, k) *
                                        std::sqrt(tke_arrs[nbx](i, j, k)) *
                                        fluid_weight;
            buoy_prod_arrs[nbx](i, j, k) = -muPrime * stratification;
            shear_prod_arrs[nbx](i, j, k) *=
                shear_prod_arrs[nbx](i, j, k) * mu_arrs[nbx](i, j, k);
        });
}

template <typename Transport>
void KLAxellSeparation<Transport>::pressure_gradient_sensor(
    const int lev, const FieldState fstate)
{
    const auto tiny = std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real c_u = m_sensor_velocity_weight;

    const auto& vel_arrs = this->m_vel.state(fstate)(lev).const_arrays();
    const auto& rho_arrs = this->m_rho.state(fstate)(lev).const_arrays();
    const auto& tke_arrs = (*this->m_tke)(lev).const_arrays();
    const auto& tlscale_arrs = (this->m_turb_lscale)(lev).const_arrays();
    const auto& gp_arrs =
        this->m_sim.repo().get_field("gp")(lev).const_arrays();
    const auto& geom_arrs = (*m_geometry)(lev).const_arrays();
    const auto& sensor_arrs = (*m_pressure_gradient_sensor)(lev).arrays();

    // (u / |u|) . grad(p) divided by rho (k + c_u |u|^2) / L. The divisors are
    // floored so that the sensor stays finite in still air, and the fluid
    // weight zeroes it inside the terrain.
    amrex::ParallelFor(
        (*m_pressure_gradient_sensor)(lev),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            const auto& vel = vel_arrs[nbx];
            const auto& gp = gp_arrs[nbx];
            const amrex::Real umag_sqr = (vel(i, j, k, 0) * vel(i, j, k, 0)) +
                                         (vel(i, j, k, 1) * vel(i, j, k, 1)) +
                                         (vel(i, j, k, 2) * vel(i, j, k, 2));
            const amrex::Real scale =
                geom_arrs[nbx](i, j, k, klaxell_separation::geom_fluid_weight) *
                tlscale_arrs[nbx](i, j, k) /
                (amrex::max<amrex::Real>(std::sqrt(umag_sqr), tiny) *
                 amrex::max<amrex::Real>(
                     rho_arrs[nbx](i, j, k) *
                         (tke_arrs[nbx](i, j, k) + (c_u * umag_sqr)),
                     tiny));
            sensor_arrs[nbx](i, j, k) = ((vel(i, j, k, 0) * gp(i, j, k, 0)) +
                                         (vel(i, j, k, 1) * gp(i, j, k, 1)) +
                                         (vel(i, j, k, 2) * gp(i, j, k, 2))) *
                                        scale;
        });
}

template <typename Transport>
void KLAxellSeparation<Transport>::velocity_sensor(
    const int lev, const FieldState fstate)
{
    const auto tiny = std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real c_u = m_sensor_velocity_weight;

    // Ghost cells of the fluid weight: copied from the neighboring boxes and
    // across periodic boundaries, 1 (fluid) on the other domain boundaries
    auto& geom_mf = (*m_geometry)(lev);
    geom_mf.setBndry(1.0_rt);
    geom_mf.FillBoundary(this->m_sim.repo().mesh().Geom(lev).periodicity());

    const auto& vel_arrs = this->m_vel.state(fstate)(lev).const_arrays();
    const auto& gradvel_arrs = (*m_grad_vel)(lev).const_arrays();
    const auto& tke_arrs = (*this->m_tke)(lev).const_arrays();
    const auto& tlscale_arrs = (this->m_turb_lscale)(lev).const_arrays();
    const auto& geom_arrs = geom_mf.const_arrays();
    const auto& sensor_arrs = (*m_pressure_gradient_sensor)(lev).arrays();

    // -(u . grad)(|u|^2 / 2) / |u| = -u_m u_n d(u_n)/d(x_m) / |u|, divided by
    // (k + c_u |u|^2) / L with the divisors floored as in the pressure sensor.
    // The gradient field stores d(u_n)/d(x_m) in component n * 3 + m. The
    // smallest fluid weight of the face neighbors zeroes the sensor next to
    // the terrain.
    amrex::ParallelFor(
        (*m_pressure_gradient_sensor)(lev),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            const auto& vel = vel_arrs[nbx];
            const auto& gradvel = gradvel_arrs[nbx];
            const auto& geo = geom_arrs[nbx];
            amrex::Real advection = 0.0_rt;
            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                for (int m = 0; m < AMREX_SPACEDIM; ++m) {
                    advection += vel(i, j, k, m) * vel(i, j, k, n) *
                                 gradvel(i, j, k, (n * AMREX_SPACEDIM) + m);
                }
            }
            const amrex::Real umag_sqr = (vel(i, j, k, 0) * vel(i, j, k, 0)) +
                                         (vel(i, j, k, 1) * vel(i, j, k, 1)) +
                                         (vel(i, j, k, 2) * vel(i, j, k, 2));
            const int w = klaxell_separation::geom_fluid_weight;
            const amrex::Real neighbor_weight = amrex::min(
                geo(i - 1, j, k, w), geo(i + 1, j, k, w), geo(i, j - 1, k, w),
                geo(i, j + 1, k, w), geo(i, j, k - 1, w), geo(i, j, k + 1, w));
            const amrex::Real scale =
                geo(i, j, k, w) * neighbor_weight * tlscale_arrs[nbx](i, j, k) /
                (amrex::max<amrex::Real>(std::sqrt(umag_sqr), tiny) *
                 amrex::max<amrex::Real>(
                     tke_arrs[nbx](i, j, k) + (c_u * umag_sqr), tiny));
            sensor_arrs[nbx](i, j, k) = -advection * scale;
        });
}

template <typename Transport>
void KLAxellSeparation<Transport>::realizable_cmu(const int lev)
{
    const auto tiny = std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real Cmu = this->m_Cmu;
    const amrex::Real threshold = m_sensor_threshold;
    const amrex::Real strength = m_realizable_cmu_strength;

    const auto& mu_arrs = this->mu_turb()(lev).arrays();
    const auto& buoy_prod_arrs = (this->m_buoy_prod)(lev).arrays();
    const auto& shear_prod_arrs = (this->m_shear_prod)(lev).arrays();
    const auto& tke_arrs = (*this->m_tke)(lev).const_arrays();
    const auto& tlscale_arrs = (this->m_turb_lscale)(lev).const_arrays();
    const auto& strain_arrs = (*m_strain)(lev).const_arrays();
    const auto& sensor_arrs = (*m_pressure_gradient_sensor)(lev).const_arrays();

    // Divide by 1 + c_s g max(0, Sigma / Cmu - 1) with the sensor gate g; the
    // factor is exactly 1 where g = 0, so those cells keep the KLAxell values
    amrex::ParallelFor(
        this->mu_turb()(lev),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            // Linear ramp from 0 at the threshold to 1 at twice the threshold,
            // exactly 0 below it
            const amrex::Real gate = amrex::min<amrex::Real>(
                amrex::max<amrex::Real>(
                    (sensor_arrs[nbx](i, j, k) - threshold) / threshold,
                    0.0_rt),
                1.0_rt);
            const amrex::Real sigma =
                tlscale_arrs[nbx](i, j, k) * strain_arrs[nbx](i, j, k) /
                amrex::max<amrex::Real>(
                    std::sqrt(tke_arrs[nbx](i, j, k)), tiny);
            const amrex::Real factor =
                1.0_rt / (1.0_rt + (strength * gate *
                                    amrex::max<amrex::Real>(
                                        (sigma / Cmu) - 1.0_rt, 0.0_rt)));
            mu_arrs[nbx](i, j, k) *= factor;
            buoy_prod_arrs[nbx](i, j, k) *= factor;
            shear_prod_arrs[nbx](i, j, k) *= factor;
        });
}

} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(KLAxellSeparation);

} // namespace kynema_sgf
