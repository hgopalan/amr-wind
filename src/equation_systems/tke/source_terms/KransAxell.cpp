#include <AMReX_Orientation.H>

#include "src/equation_systems/tke/source_terms/KransAxell.H"
#include "src/CFDSim.H"
#include "src/turbulence/TurbulenceModel.H"
#include "src/utilities/linear_interpolation.H"
#include "src/utilities/constants.H"
#include "src/utilities/math_ops.H"
#include "src/utilities/io_utils.H"

using namespace amrex::literals;
namespace kynema_sgf::pde::tke {

KransAxell::KransAxell(const CFDSim& sim)
    : m_turb_lscale(sim.repo().get_field("turb_lscale"))
    , m_shear_prod(sim.repo().get_field("shear_prod"))
    , m_buoy_prod(sim.repo().get_field("buoy_prod"))
    , m_dissip(sim.repo().get_field("dissipation"))
    , m_tke(sim.repo().get_field("tke"))
    , m_time(sim.time())
    , m_sim(sim)
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_transport(sim.transport_model())
{
    AMREX_ALWAYS_ASSERT(sim.turbulence_model().model_name() == "KLAxell");
    // Dissipation and the wall value of k use the C_mu of the eddy viscosity
    const auto coeffs = sim.turbulence_model().model_coeffs();
    m_Cmu = coeffs.at("Cmu");
    amrex::ParmParse pp("ABL");
    if (pp.contains("Cmu")) {
        amrex::Real abl_cmu = m_Cmu;
        pp.get("Cmu", abl_cmu);
        if (abl_cmu != m_Cmu) {
            amrex::Abort(
                "KransAxell: ABL.Cmu differs from KLAxell_coeffs.Cmu. "
                "KransAxell now takes C_mu from KLAxell_coeffs.Cmu; remove "
                "ABL.Cmu or set both to the same value.");
        }
    }
    pp.query("kappa", m_kappa);
    pp.query("surface_roughness_z0", m_z0);
    pp.query("surface_temp_flux", m_heat_flux);
    pp.query("meso_sponge_start", m_meso_start);
    pp.query("rans_1dprofile_file", m_1d_rans);
    pp.query("horizontal_sponge_tke", m_horizontal_sponge);
    if (!m_1d_rans.empty()) {
        const auto prof = ioutils::read_rans_1d_profile(m_1d_rans);
        m_wind_heights = prof.z;
        m_tke_values = prof.tke;
        int num_wind_values = static_cast<int>(m_wind_heights.size());
        m_wind_heights_d.resize(num_wind_values);
        m_tke_values_d.resize(num_wind_values);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_wind_heights.begin(),
            m_wind_heights.end(), m_wind_heights_d.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_tke_values.begin(), m_tke_values.end(),
            m_tke_values_d.begin());
    } else {
        amrex::Abort("Cannot find 1-D RANS profile file " + m_1d_rans);
    }
    m_stability = surface_stability::parse(
        identifier(), sim.transport_model().reference_temperature());
    surface_stability::require_obukhov_length(m_stability, identifier());

    amrex::ParmParse pp_incflo("incflo");
    pp_incflo.queryarr("gravity", m_gravity);

    amrex::ParmParse pp_drag("DragForcing");
    pp_drag.query("bc_forcing_time_factor", m_forcing_time_factor);
    if (!(m_forcing_time_factor > 0.0_rt)) {
        amrex::Abort("KransAxell: DragForcing.bc_forcing_time_factor must be > 0");
    }
    pp_drag.query("sponge_west", m_sponge_west);
    pp_drag.query("sponge_east", m_sponge_east);
    pp_drag.query("sponge_south", m_sponge_south);
    pp_drag.query("sponge_north", m_sponge_north);
    if (m_sponge_west) {
        pp_drag.get("sponge_distance_west", m_sponge_distance_west);
    }
    if (m_sponge_east) {
        pp_drag.get("sponge_distance_east", m_sponge_distance_east);
    }
    if (m_sponge_south) {
        pp_drag.get("sponge_distance_south", m_sponge_distance_south);
    }
    if (m_sponge_north) {
        pp_drag.get("sponge_distance_north", m_sponge_distance_north);
    }
    if (m_horizontal_sponge) {
        // A west/south distance > 0 (or east/north < 0) puts the whole domain
        // inside the sponge
        if (!(m_sponge_west || m_sponge_east || m_sponge_south ||
              m_sponge_north)) {
            amrex::Abort(
                "KransAxell: ABL.horizontal_sponge_tke needs at least one of "
                "DragForcing.sponge_west, sponge_east, sponge_south, "
                "sponge_north");
        }
        if (m_sponge_west && !(m_sponge_distance_west < 0.0_rt)) {
            amrex::Abort(
                "KransAxell: DragForcing.sponge_distance_west must be < 0");
        }
        if (m_sponge_east && !(m_sponge_distance_east > 0.0_rt)) {
            amrex::Abort(
                "KransAxell: DragForcing.sponge_distance_east must be > 0");
        }
        if (m_sponge_south && !(m_sponge_distance_south < 0.0_rt)) {
            amrex::Abort(
                "KransAxell: DragForcing.sponge_distance_south must be < 0");
        }
        if (m_sponge_north && !(m_sponge_distance_north > 0.0_rt)) {
            amrex::Abort(
                "KransAxell: DragForcing.sponge_distance_north must be > 0");
        }
    }

    m_canopy = forestdrag::parse_canopy_turbulence();
}

KransAxell::~KransAxell() = default;

void KransAxell::operator()(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term) const
{
    // Built on the grids of this call: a field kept from an earlier call
    // would keep the old grids after a regrid
    amrex::MultiFab ref_theta(
        src_term.boxArray(), src_term.DistributionMap(), 1, 0);
    m_transport.ref_theta_fill(lev, ref_theta);

    const auto& geom = m_mesh.Geom(lev);
    const auto& problo = geom.ProbLoArray();
    const auto& probhi = geom.ProbHiArray();
    const auto& dx = geom.CellSizeArray();
    const auto& dt = m_time.delta_t();
    const amrex::Real heat_flux = m_heat_flux;
    const amrex::Real Cmu = m_Cmu;
    const amrex::Real kappa = m_kappa;
    const amrex::Real time_factor = m_forcing_time_factor;
    const amrex::Real z0 = m_z0;
    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");
    const amrex::Real sponge_start = m_meso_start;
    const auto vsize = m_wind_heights_d.size();
    const auto* wind_heights_d = m_wind_heights_d.data();
    const auto* tke_values_d = m_tke_values_d.data();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    amrex::Real psi_m = 0.0_rt;
    if (m_stability.non_neutral) {
        psi_m = surface_stability::psi_m(
            1.5_rt * dx[2] / m_stability.obukhov_len, m_stability.beta_m,
            m_stability.gamma_m);
    }

    auto const& src_arrs = src_term.arrays();
    auto const& vel_arrs =
        m_velocity.state(field_impl::dof_state(fstate))(lev).const_arrays();
    auto const& tlscale_arrs = m_turb_lscale(lev).const_arrays();
    auto const& shear_prod_arrs = m_shear_prod(lev).const_arrays();
    auto const& buoy_prod_arrs = m_buoy_prod(lev).const_arrays();
    auto const& dissip_arrs = m_dissip(lev).arrays();
    auto const& tke_arrs = m_tke(lev).const_arrays();
    auto const& ref_theta_arrs = ref_theta.const_arrays();

    // Main kernel
    amrex::ParallelFor(
        src_term, amrex::IntVect(0), 1,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int) {
            const auto& vel = vel_arrs[nbx];
            const auto& tlscale_arr = tlscale_arrs[nbx];
            const auto& shear_prod_arr = shear_prod_arrs[nbx];
            const auto& buoy_prod_arr = buoy_prod_arrs[nbx];
            const auto& dissip_arr = dissip_arrs[nbx];
            const auto& tke_arr = tke_arrs[nbx];
            const auto& ref_theta_arr = ref_theta_arrs[nbx];

            amrex::Real bcforcing = 0;
            const amrex::Real z = problo[2] + ((k + 0.5_rt) * dx[2]);
            if (k == 0) {
                // Heights above the wall at prob_lo[2]: zwall for the first
                // cell, 3 zwall for the cell above it
                const amrex::Real zwall = 0.5_rt * dx[2];
                const amrex::Real ux = vel(i, j, k + 1, 0);
                const amrex::Real uy = vel(i, j, k + 1, 1);
                const amrex::Real m = std::sqrt((ux * ux) + (uy * uy));
                const amrex::Real ustar =
                    m * kappa / (std::log(3.0_rt * zwall / z0) - psi_m);
                const amrex::Real T0 = ref_theta_arr(i, j, k);
                const amrex::Real hf = std::abs(gravity[2]) / T0 * heat_flux;
                const amrex::Real rans_b = amrex::max<amrex::Real>(hf, 0.0_rt) *
                                           kappa * zwall / utils::powi(Cmu, 3);
                const amrex::Real tke_exact = std::pow(
                    ustar * ustar * ustar / (Cmu * Cmu * Cmu) + rans_b,
                    2.0_rt / 3.0_rt);
                bcforcing = (tke_exact - tke_arr(i, j, k)) / (time_factor * dt);
            }
            amrex::Real ref_tke = tke_arr(i, j, k);
            if (z > sponge_start) {
                ref_tke = (vsize > 0)
                              ? interp::linear(
                                    wind_heights_d, wind_heights_d + vsize,
                                    tke_values_d, z)
                              : tke_arr(i, j, k, 0);
            }
            const amrex::Real sponge_forcing =
                1.0_rt / dt * (tke_arr(i, j, k) - ref_tke);
            dissip_arr(i, j, k) =
                utils::powi(Cmu, 3) * std::pow(tke_arr(i, j, k), 1.5_rt) /
                (tlscale_arr(i, j, k) + kynema_sgf::constants::EPS);
            src_arrs[nbx](i, j, k) +=
                shear_prod_arr(i, j, k) + buoy_prod_arr(i, j, k) -
                dissip_arr(i, j, k) -
                ((1.0_rt - static_cast<int>(has_terrain)) *
                 (sponge_forcing - bcforcing));
        });

    if (has_terrain) {
        const amrex::Real z0_min = 1.0e-4_rt;
        const auto& terrain_blank =
            this->m_sim.repo().get_int_field("terrain_blank");
        const auto& terrain_drag =
            this->m_sim.repo().get_int_field("terrain_drag");
        auto& terrain_height = this->m_sim.repo().get_field("terrain_height");
        auto& terrainz0 = this->m_sim.repo().get_field("terrainz0");

        auto const& blank_arrs = terrain_blank(lev).const_arrays();
        auto const& drag_arrs = terrain_drag(lev).const_arrays();
        auto const& terrain_height_arrs = terrain_height(lev).const_arrays();
        auto const& terrainz0_arrs = terrainz0(lev).const_arrays();

        amrex::ParallelFor(
            src_term, amrex::IntVect(0), 1,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int) {
                const auto& vel = vel_arrs[nbx];
                const auto& tke_arr = tke_arrs[nbx];
                const auto& blank_arr = blank_arrs[nbx];
                const auto& drag_arr = drag_arrs[nbx];
                const auto& terrain_height_arr = terrain_height_arrs[nbx];
                const auto& terrainz0_arr = terrainz0_arrs[nbx];
                const auto& ref_theta_arr = ref_theta_arrs[nbx];

                const amrex::Real cell_z0 =
                    (drag_arr(i, j, k) *
                     amrex::max<amrex::Real>(terrainz0_arr(i, j, k), z0_min)) +
                    ((1.0_rt - drag_arr(i, j, k)) * z0);
                amrex::Real ux = vel(i, j, k + 1, 0);
                amrex::Real uy = vel(i, j, k + 1, 1);
                amrex::Real z = 0.5_rt * dx[2];
                amrex::Real m = std::sqrt((ux * ux) + (uy * uy));
                const amrex::Real ustar =
                    m * kappa / (std::log(3.0_rt * z / cell_z0) - psi_m);
                const amrex::Real T0 = ref_theta_arr(i, j, k);
                const amrex::Real hf = std::abs(gravity[2]) / T0 * heat_flux;
                const amrex::Real rans_b = amrex::max<amrex::Real>(hf, 0.0_rt) *
                                           kappa * z / utils::powi(Cmu, 3);
                const amrex::Real tke_exact = std::pow(
                    ustar * ustar * ustar / (Cmu * Cmu * Cmu) + rans_b,
                    2.0_rt / 3.0_rt);
                const amrex::Real terrainforcing =
                    (tke_exact - tke_arr(i, j, k)) / (time_factor * dt);
                amrex::Real bcforcing = 0.0_rt;
                if (k == 0) {
                    bcforcing = (1 - blank_arr(i, j, k)) * terrainforcing;
                }
                ux = vel(i, j, k, 0);
                uy = vel(i, j, k, 1);
                const amrex::Real uz = vel(i, j, k, 2);
                m = std::sqrt((ux * ux) + (uy * uy) + (uz * uz));
                const amrex::Real Cd = amrex::min<amrex::Real>(
                    10.0_rt / (dx[2] * m + kynema_sgf::constants::EPS),
                    100.0_rt / dx[2]);
                const amrex::Real dragforcing = -Cd * m * tke_arr(i, j, k, 0);
                z = amrex::max<amrex::Real>(
                    problo[2] + ((k + 0.5_rt) * dx[2]) -
                        terrain_height_arr(i, j, k),
                    0.5_rt * dx[2]);
                amrex::Real ref_tke = tke_arr(i, j, k);
                if (z > sponge_start) {
                    ref_tke = (vsize > 0)
                                  ? interp::linear(
                                        wind_heights_d, wind_heights_d + vsize,
                                        tke_values_d, z)
                                  : tke_arr(i, j, k, 0);
                }
                const amrex::Real sponge_forcing =
                    1.0_rt / dt * (tke_arr(i, j, k) - ref_tke);
                src_arrs[nbx](i, j, k) =
                    ((1.0_rt - blank_arr(i, j, k)) * src_arrs[nbx](i, j, k)) +
                    (drag_arr(i, j, k) * terrainforcing) +
                    (blank_arr(i, j, k) * dragforcing) -
                    (static_cast<int>(has_terrain) *
                     (sponge_forcing - bcforcing));
            });

    }

    // Horizontal TKE sponge (ABL.horizontal_sponge_tke), with or without
    // terrain. Inside any active DragForcing sponge zone, k is set to the
    // 1-D profile value in one time step: the source is -(k - k_ref) / dt.
    if (m_horizontal_sponge) {
        const amrex::Real start_east = probhi[0] - m_sponge_distance_east;
        const amrex::Real start_west = problo[0] - m_sponge_distance_west;
        const amrex::Real start_north = probhi[1] - m_sponge_distance_north;
        const amrex::Real start_south = problo[1] - m_sponge_distance_south;
        const bool sponge_east = m_sponge_east;
        const bool sponge_west = m_sponge_west;
        const bool sponge_south = m_sponge_south;
        const bool sponge_north = m_sponge_north;

        amrex::ParallelFor(
            src_term, amrex::IntVect(0), 1,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int) {
                const auto& tke_arr = tke_arrs[nbx];
                const amrex::Real x = problo[0] + ((i + 0.5_rt) * dx[0]);
                const amrex::Real y = problo[1] + ((j + 0.5_rt) * dx[1]);
                const amrex::Real z = problo[2] + ((k + 0.5_rt) * dx[2]);
                const bool in_sponge = (sponge_west && (x < start_west)) ||
                                       (sponge_east && (x > start_east)) ||
                                       (sponge_south && (y < start_south)) ||
                                       (sponge_north && (y > start_north));
                if (!in_sponge) {
                    return;
                }
                const amrex::Real ref_tke =
                    (vsize > 0)
                        ? interp::linear(
                              wind_heights_d, wind_heights_d + vsize,
                              tke_values_d, z)
                        : tke_arr(i, j, k, 0);
                src_arrs[nbx](i, j, k, 0) -=
                    (tke_arr(i, j, k) - ref_tke) / dt;
            });
    }

    // Wake production and short-circuit dissipation in the forest canopy
    if (m_canopy.m_enabled) {
        if (!m_sim.repo().field_exists("forest_drag")) {
            amrex::Abort(
                "KransAxell: ForestDrag.canopy_tke needs ForestDrag in "
                "incflo.physics");
        }
        forestdrag::add_canopy_tke_source(
            src_term, m_sim.repo().get_field("forest_drag")(lev),
            m_velocity.state(field_impl::dof_state(fstate))(lev), m_tke(lev),
            m_canopy.m_beta_p, m_canopy.m_beta_d, dt);
    }
}

} // namespace kynema_sgf::pde::tke
