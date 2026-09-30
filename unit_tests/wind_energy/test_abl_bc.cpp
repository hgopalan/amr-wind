#include "abl_test_utils.H"
#include "src/utilities/trig_ops.H"
#include "ks_test_utils/iter_tools.H"
#include "ks_test_utils/test_utils.H"
#include "src/incflo.H"

#include "AMReX_Gpu.H"
#include "AMReX_Random.H"
#include "src/equation_systems/icns/icns.H"
#include "src/equation_systems/icns/icns_ops.H"
#include "src/equation_systems/icns/MomentumSource.H"
#include "src/equation_systems/icns/source_terms/BodyForce.H"
#include "src/equation_systems/icns/source_terms/ABLForcing.H"
#include "src/equation_systems/icns/source_terms/GeostrophicForcing.H"
#include "src/equation_systems/icns/source_terms/CoriolisForcing.H"
#include "src/equation_systems/icns/source_terms/BoussinesqBuoyancy.H"
#include "src/equation_systems/icns/source_terms/DensityBuoyancy.H"
#include "src/equation_systems/icns/source_terms/HurricaneForcing.H"
#include "src/equation_systems/icns/source_terms/RayleighDamping.H"
#include "src/utilities/math_ops.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {
amrex::Real
get_val_at_kindex(kynema_sgf::Field& field, const int comp, const int kref)
{
    const int lev = 0;
    amrex::Real error_total = 0;

    error_total += amrex::ReduceSum(
        field(lev), 0,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx,
            amrex::Array4<amrex::Real const> const& f_arr) -> amrex::Real {
            amrex::Real error = 0;

            amrex::Loop(bx, [=, &error](int i, int j, int k) {
                // Check if current cell is just above lower wall
                if (k == kref) {
                    // Add field value to output
                    error += f_arr(i, j, k, comp);
                }
            });

            return error;
        });
    amrex::ParallelDescriptor::ReduceRealSum(error_total);
    return error_total;
}
/** Largest magnitude in the ghost layer below the domain
 *
 *  A value that is not finite counts as the largest real, so a NaN or an
 *  infinity cannot hide in the maximum.
 *
 *  \param field Field whose lower ghost layer is checked
 *  \return Largest magnitude found there
 */
amrex::Real max_abs_below_zlo(kynema_sgf::Field& field)
{
    const auto& domain = field.repo().mesh().Geom(0).Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);
    amrex::Real result = amrex::ReduceMax(
        field(0), 1,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx,
            amrex::Array4<amrex::Real const> const& f_arr) -> amrex::Real {
            amrex::Real vmax = 0.0_rt;
            amrex::Loop(bx, [=, &vmax](int i, int j, int k) {
                // Ghost cells directly below the domain only
                if ((k == dlo.z - 1) && (i >= dlo.x) && (i <= dhi.x) &&
                    (j >= dlo.y) && (j <= dhi.y)) {
                    const amrex::Real v = f_arr(i, j, k);
                    vmax = amrex::max(
                        vmax, std::isfinite(v)
                                  ? std::abs(v)
                                  : std::numeric_limits<amrex::Real>::max());
                }
            });
            return vmax;
        });
    amrex::ParallelDescriptor::ReduceRealMax(result);
    return result;
}

void init_velocity(kynema_sgf::Field& fld, amrex::Real vval, int dir)
{
    const int nlevels = fld.repo().num_active_levels();

    // Initialize entire field to 0
    fld.setVal(0.0_rt);

    for (int lev = 0; lev < nlevels; ++lev) {
        fld(lev).setVal(vval, dir, 1);
    }
}
} // namespace

using ICNSFields = kynema_sgf::pde::
    FieldRegOp<kynema_sgf::pde::ICNS, kynema_sgf::fvm::Godunov>;

TEST_F(ABLMeshTest, abl_local_wall_model)
{
    constexpr amrex::Real tol =
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;
    constexpr amrex::Real mu = 0.01_rt;
    constexpr amrex::Real vval = 5.0_rt;
    constexpr amrex::Real dt = 0.1_rt;
    constexpr amrex::Real kappa = 0.4_rt;
    constexpr amrex::Real z0 = 0.11_rt;
    int dir = 0;
    populate_parameters();
    {
        amrex::ParmParse pp("geometry");
        amrex::Vector<int> periodic{{1, 1, 0}};
        pp.addarr("is_periodic", periodic);
    }
    {
        amrex::ParmParse pp("zlo");
        pp.add("type", (std::string) "wall_model");
    }
    {
        amrex::ParmParse pp("zhi");
        pp.add("type", (std::string) "slip_wall");
    }
    {
        amrex::ParmParse pp("incflo");
        pp.add("diffusion_type", 0);
    }
    {
        amrex::ParmParse pp("transport");
        pp.add("viscosity", mu);
    }
    {
        amrex::ParmParse pp("time");
        pp.add("fixed_dt", dt);
    }
    {
        amrex::ParmParse pp("ABL");
        pp.add("wall_shear_stress_type", (std::string) "local");
        pp.add("kappa", kappa);
        pp.add("surface_roughness_z0", z0);
    }
    initialize_mesh();

    // Set up solver-related routines
    auto& pde_mgr = sim().pde_manager();
    pde_mgr.register_icns();
    sim().create_turbulence_model();
    sim().turbulence_model().post_init_actions();
    sim().init_physics();

    // Specify velocity as uniform in x direction
    auto& velocity = sim().repo().get_field("velocity");
    init_velocity(velocity, vval, dir);
    // Specify density as unity
    auto& density = sim().repo().get_field("density");
    density.setVal(1.0_rt);
    // Perform post init for physics: turns on wall model
    for (auto& pp : sim().physics()) {
        pp->post_init_actions();
    }

    // Advance states to prepare for time step
    pde_mgr.advance_states();

    // Initialize icns pde
    auto& icns_eq = pde_mgr.icns();
    icns_eq.initialize();
    // Initialize viscosity
    sim().turbulence_model().update_turbulent_viscosity(
        kynema_sgf::FieldState::Old, DiffusionType::Crank_Nicolson);
    icns_eq.compute_mueff(kynema_sgf::FieldState::Old);

    // Check test setup by verifying mu
    const auto& viscosity = sim().repo().get_field("velocity_mueff");
    EXPECT_NEAR(mu, utils::field_max(viscosity), tol);
    EXPECT_NEAR(mu, utils::field_min(viscosity), tol);

    // Zero source term and convection term to focus on diffusion
    auto& src = icns_eq.fields().src_term;
    auto& adv = icns_eq.fields().conv_term;
    src.setVal(0.0_rt);
    adv.setVal(0.0_rt);

    // Calculate diffusion term
    icns_eq.compute_diffusion_term(kynema_sgf::FieldState::Old);
    // Setup mask_cell array to avoid errors in solve
    auto& mask_cell = sim().repo().declare_int_field("mask_cell", 1, 1);
    mask_cell.setVal(1);
    // Compute result with just diffusion term
    icns_eq.compute_predictor_rhs(DiffusionType::Explicit);

    // Get resulting velocity in first cell
    const amrex::Real vbase = get_val_at_kindex(velocity, dir, 0) / 8 / 8;

    // Calculate expected velocity after one step
    const amrex::Real dz = sim().mesh().Geom(0).CellSizeArray()[2];
    const amrex::Real zref = 0.5_rt * dz;
    const amrex::Real utau = kappa * vval / (std::log(zref / z0));
    const amrex::Real tau_wall = kynema_sgf::utils::powi(utau, 2);
    const amrex::Real vexpct = vval + (dt * (0.0_rt - tau_wall) / dz);
    EXPECT_NEAR(vexpct, vbase, tol);
}

// Before the turbulence model has run the effective diffusivity is zero, and
// the wall heat flux must then be left out rather than divided by it
TEST_F(ABLMeshTest, abl_temperature_wall_model_with_zero_diffusivity)
{
    populate_parameters();
    {
        amrex::ParmParse pp("geometry");
        amrex::Vector<int> periodic{{1, 1, 0}};
        pp.addarr("is_periodic", periodic);
    }
    {
        amrex::ParmParse pp("zlo");
        pp.add("type", (std::string) "wall_model");
    }
    {
        amrex::ParmParse pp("zhi");
        pp.add("type", (std::string) "slip_wall");
    }
    {
        amrex::ParmParse pp("ABL");
        pp.add("surface_temp_flux", 0.1_rt);
    }
    initialize_mesh();

    auto& pde_mgr = sim().pde_manager();
    pde_mgr.register_icns();
    sim().create_turbulence_model();
    sim().init_physics();

    auto& velocity = sim().repo().get_field("velocity");
    init_velocity(velocity, 5.0_rt, 0);
    sim().repo().get_field("density").setVal(1.0_rt);
    for (auto& pp : sim().physics()) {
        pp->post_init_actions();
    }
    pde_mgr.advance_states();

    auto& temperature = sim().repo().get_field("temperature");
    sim().repo().get_field("temperature_mueff").setVal(0.0_rt);
    temperature.setVal(300.0_rt);
    temperature.apply_bc_funcs(kynema_sgf::FieldState::New);

    EXPECT_EQ(max_abs_below_zlo(temperature), 0.0_rt);
}

TEST_F(ABLMeshTest, abl_donelan_wall_model)
{
    constexpr amrex::Real tol =
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;
    constexpr amrex::Real mu = 0.01_rt;
    constexpr amrex::Real vval = 35.0_rt;
    constexpr amrex::Real dt = 0.1_rt;
    constexpr amrex::Real kappa = 0.4_rt;
    constexpr amrex::Real z0 = 0.11_rt;
    int dir = 0;
    populate_parameters();
    {
        amrex::ParmParse pp("geometry");
        amrex::Vector<int> periodic{{1, 1, 0}};
        pp.addarr("is_periodic", periodic);
    }
    {
        amrex::ParmParse pp("zlo");
        pp.add("type", (std::string) "wall_model");
    }
    {
        amrex::ParmParse pp("zhi");
        pp.add("type", (std::string) "slip_wall");
    }
    {
        amrex::ParmParse pp("incflo");
        pp.add("diffusion_type", 0);
    }
    {
        amrex::ParmParse pp("transport");
        pp.add("viscosity", mu);
    }
    {
        amrex::ParmParse pp("time");
        pp.add("fixed_dt", dt);
    }
    {
        amrex::ParmParse pp("ABL");
        pp.add("wall_shear_stress_type", (std::string) "donelan");
        pp.add("kappa", kappa);
        pp.add("surface_roughness_z0", z0);
    }
    initialize_mesh();

    // Set up solver-related routines
    auto& pde_mgr = sim().pde_manager();
    pde_mgr.register_icns();
    sim().create_turbulence_model();
    sim().turbulence_model().post_init_actions();
    sim().init_physics();

    // Specify velocity as uniform in x direction
    auto& velocity = sim().repo().get_field("velocity");
    init_velocity(velocity, vval, dir);
    // Specify density as unity
    auto& density = sim().repo().get_field("density");
    density.setVal(1.0_rt);
    // Perform post init for physics: turns on wall model
    for (auto& pp : sim().physics()) {
        pp->post_init_actions();
    }

    // Advance states to prepare for time step
    pde_mgr.advance_states();

    // Initialize icns pde
    auto& icns_eq = pde_mgr.icns();
    icns_eq.initialize();
    // Initialize viscosity
    sim().turbulence_model().update_turbulent_viscosity(
        kynema_sgf::FieldState::Old, DiffusionType::Crank_Nicolson);
    icns_eq.compute_mueff(kynema_sgf::FieldState::Old);

    // Check test setup by verifying mu
    const auto& viscosity = sim().repo().get_field("velocity_mueff");
    EXPECT_NEAR(mu, utils::field_max(viscosity), tol);
    EXPECT_NEAR(mu, utils::field_min(viscosity), tol);

    // Zero source term and convection term to focus on diffusion
    auto& src = icns_eq.fields().src_term;
    auto& adv = icns_eq.fields().conv_term;
    src.setVal(0.0_rt);
    adv.setVal(0.0_rt);

    // Calculate diffusion term
    icns_eq.compute_diffusion_term(kynema_sgf::FieldState::Old);
    // Setup mask_cell array to avoid errors in solve
    auto& mask_cell = sim().repo().declare_int_field("mask_cell", 1, 1);
    mask_cell.setVal(1);
    // Compute result with just diffusion term
    icns_eq.compute_predictor_rhs(DiffusionType::Explicit);

    // Get resulting velocity in first cell
    const amrex::Real vbase = get_val_at_kindex(velocity, dir, 0) / 8 / 8;

    // Calculate expected velocity after one step
    const amrex::Real dz = sim().mesh().Geom(0).CellSizeArray()[2];
    const amrex::Real tau_wall = 0.0024_rt * kynema_sgf::utils::powi(vval, 2);
    const amrex::Real vexpct = vval + (dt * (0.0_rt - tau_wall) / dz);
    EXPECT_NEAR(vexpct, vbase, tol);
}

} // namespace kynema_sgf_tests
