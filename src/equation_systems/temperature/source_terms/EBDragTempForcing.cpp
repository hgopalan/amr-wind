#include "src/equation_systems/temperature/source_terms/EBDragTempForcing.H"
#include "src/utilities/IOManager.H"
#include "src/wind_energy/MOData.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"
#include "AMReX_Random.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::temperature {

EBDragTempForcing::EBDragTempForcing(const CFDSim& sim)
    : m_time(sim.time())
    , m_sim(sim)
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_temperature(sim.repo().get_field("temperature"))
{
    amrex::ParmParse pp("EBDragTempForcing");
    pp.query("drag_coefficient", m_drag_coefficient);
    pp.query("soil_temperature", m_soil_temperature);
    pp.query("bc_forcing_time_factor", m_forcing_time_factor);
    amrex::ParmParse pp_abl("ABL");
    pp_abl.query("wall_het_model", m_wall_het_model);
    pp_abl.query("monin_obukhov_length", m_monin_obukhov_length);
    pp_abl.query("kappa", m_kappa);
    pp_abl.query("mo_gamma_m", m_gamma_m);
    pp_abl.query("mo_beta_m", m_beta_m);
    pp_abl.query("mo_gamma_m", m_gamma_h);
    pp_abl.query("mo_beta_m", m_beta_h);

    amrex::ParmParse pp_incflo("incflo");
    pp_incflo.queryarr("gravity", m_gravity);
}

EBDragTempForcing::~EBDragTempForcing() = default;

void EBDragTempForcing::operator()(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term) const
{
    auto const& src_arrs = src_term.arrays();
    auto const& vel =
        m_velocity.state(field_impl::dof_state(fstate))(lev).const_arrays();
    auto const& temperature =
        m_temperature.state(field_impl::dof_state(fstate))(lev).const_arrays();
    const int is_eb = this->m_sim.repo().field_exists("eb_blank") ? 1 : 0;
    if (is_eb == 0) {
        amrex::Abort("Need EB blanking variable to use this source term");
    }
    auto const& blank =
        this->m_sim.repo().get_field("eb_blank")(lev).const_arrays();

    const auto& geom = m_mesh.Geom(lev);
    const auto& dx = geom.CellSizeArray();
    const amrex::Real drag_coefficient = m_drag_coefficient;
    const auto& dt = m_time.delta_t();
    const amrex::Real time_factor = m_forcing_time_factor * dt;
    const amrex::Real Cd = drag_coefficient / dx[2];
    const amrex::Real kappa = m_kappa;
    const amrex::Real cd_max = 10.0_rt;
    const amrex::Real T0 = m_soil_temperature;
    const amrex::Real z0 = 0.1_rt;

    amrex::ParallelFor(
        src_term, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int /*n*/) {
            const amrex::Real ux1 = vel[nbx](i, j, k, 0);
            const amrex::Real uy1 = vel[nbx](i, j, k, 1);
            const amrex::Real uz1 = vel[nbx](i, j, k, 2);
            const amrex::Real theta = temperature[nbx](i, j, k, 0);
            const amrex::Real m =
                std::sqrt((ux1 * ux1) + (uy1 * uy1) + (uz1 * uz1));
            const amrex::Real CdM =
                std::min(Cd / (m + kynema_sgf::constants::EPS), cd_max / dx[2]);
            src_arrs[nbx](i, j, k, 0) -=
                (CdM * (theta - T0) * blank[nbx](i, j, k, 0));
        });
}

} // namespace kynema_sgf::pde::temperature
