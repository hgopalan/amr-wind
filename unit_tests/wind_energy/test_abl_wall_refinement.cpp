#include "abl_test_utils.H"
#include "ks_test_utils/iter_tools.H"
#include "src/wind_energy/ABL.H"
#include "src/wind_energy/ABLWallFunction.H"
#include "src/utilities/tagging/CartBoxRefinement.H"

#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParReduce.H"
#include "AMReX_iMultiFab.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {

//! Fill a field with offset + slope * log(z / z0) per component, a function
//! of the wall-normal coordinate only. The ghost cells below the wall repeat
//! the first cell so that every plane average reads finite values.
void init_log_profile(
    kynema_sgf::Field& fld,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> offset,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> slope,
    const amrex::Real z0)
{
    const auto& mesh = fld.repo().mesh();
    const int nlevels = fld.repo().num_active_levels();
    const int ncomp = fld.num_comp();
    for (int lev = 0; lev < nlevels; ++lev) {
        const amrex::Real dz = mesh.Geom(lev).CellSizeArray()[2];
        const amrex::Real zlo = mesh.Geom(lev).ProbLoArray()[2];
        const auto& farrs = fld(lev).arrays();
        amrex::ParallelFor(
            fld(lev), fld.num_grow(), ncomp,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) {
                const int kk = amrex::max(k, 0);
                const amrex::Real z = zlo + ((kk + 0.5_rt) * dz);
                farrs[nbx](i, j, k, n) =
                    offset[n] + (slope[n] * std::log(z / z0));
            });
    }
    amrex::Gpu::streamSynchronize();
}

//! Multiply the valid cells of a level that are covered by the next finer
//! level by a factor
void scale_covered_cells(
    kynema_sgf::Field& fld, const int lev, const amrex::Real factor)
{
    const auto& mesh = fld.repo().mesh();
    const auto mask = amrex::makeFineMask(
        fld(lev), mesh.boxArray(lev + 1), mesh.refRatio(lev), 0, 1);
    const auto& farrs = fld(lev).arrays();
    const auto& mask_arrs = mask.const_arrays();
    amrex::ParallelFor(
        fld(lev), amrex::IntVect(0), fld.num_comp(),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) {
            if (mask_arrs[nbx](i, j, k) == 1) {
                farrs[nbx](i, j, k, n) *= factor;
            }
        });
    amrex::Gpu::streamSynchronize();
}

//! Mean, over the wall-adjacent cells of a level that are not covered by a
//! finer level, of the wall flux implied by the wall-model ghost value:
//! ghost * mueff / rho of the first cell
amrex::Real wall_flux_mean(
    const kynema_sgf::Field& fld,
    const kynema_sgf::Field& mueff,
    const kynema_sgf::Field& rho,
    const int comp,
    const int lev,
    amrex::Real& ncells)
{
    const auto& mesh = fld.repo().mesh();
    const int nlevels = fld.repo().num_active_levels();
    const int kwall = mesh.Geom(lev).Domain().smallEnd(2);

    amrex::iMultiFab level_mask;
    if (lev < nlevels - 1) {
        level_mask = amrex::makeFineMask(
            mesh.boxArray(lev), mesh.DistributionMap(lev),
            mesh.boxArray(lev + 1), mesh.refRatio(lev), 1, 0);
    } else {
        level_mask.define(
            mesh.boxArray(lev), mesh.DistributionMap(lev), 1, 0,
            amrex::MFInfo());
        level_mask.setVal(1);
    }

    const auto& f_arrs = fld(lev).const_arrays();
    const auto& mu_arrs = mueff(lev).const_arrays();
    const auto& rho_arrs = rho(lev).const_arrays();
    const auto& mask_arrs = level_mask.const_arrays();

    using SumTuple = amrex::GpuTuple<amrex::Real, amrex::Real>;
    const SumTuple sums = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Real, amrex::Real>{}, fld(lev),
        amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) -> SumTuple {
            if (k != kwall) {
                return {0.0_rt, 0.0_rt};
            }
            const auto msk =
                static_cast<amrex::Real>(mask_arrs[box_no](i, j, k));
            const amrex::Real flux = f_arrs[box_no](i, j, k - 1, comp) *
                                     mu_arrs[box_no](i, j, k) /
                                     rho_arrs[box_no](i, j, k);
            return {msk * flux, msk};
        });
    amrex::GpuArray<amrex::Real, 2> vals{
        amrex::get<0>(sums), amrex::get<1>(sums)};
    amrex::ParallelDescriptor::ReduceRealSum(vals.data(), 2);
    ncells = vals[1];
    return vals[0] / amrex::max(vals[1], 1.0_rt);
}

} // namespace

/** ABL mesh with a second level covering part of the wall-modeled boundary
 *
 *  The domain is 120 x 120 x 1000 with 8 x 8 x 64 cells on level 0; level 1
 *  covers x < 60 and z < 250, so half of the wall is seen by level-1 cells
 *  whose centers sit at half the height of the level-0 wall cells.
 */
class ABLWallRefinementTest : public ABLMeshTest
{
protected:
    void populate_parameters() override
    {
        ABLMeshTest::populate_parameters();
        {
            amrex::ParmParse pp("amr");
            pp.add("max_level", 1);
            pp.add("max_grid_size", 64);
            pp.add("blocking_factor", 4);
            pp.add("n_error_buf", 0);
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<int> periodic{{1, 1, 0}};
            pp.addarr("is_periodic", periodic);
        }
        {
            amrex::ParmParse pp("zlo");
            pp.add("type", (std::string) "wall_model");
            pp.add("temperature_type", (std::string) "wall_model");
        }
        {
            amrex::ParmParse pp("zhi");
            pp.add("type", (std::string) "slip_wall");
            pp.add("temperature_type", (std::string) "fixed_gradient");
        }
        {
            amrex::ParmParse pp("incflo");
            pp.add("diffusion_type", 0);
        }
        {
            amrex::ParmParse pp("transport");
            pp.add("viscosity", m_mu);
            pp.add("laminar_prandtl", 1.0_rt);
        }
        {
            amrex::ParmParse pp("ABL");
            pp.add("kappa", m_kappa);
            pp.add("surface_roughness_z0", m_z0);
            pp.add("surface_temp_flux", m_qwall);
            pp.add("wall_shear_stress_type", m_shear_stress_type);
            if (m_log_law_height > 0.0_rt) {
                pp.add("log_law_height", m_log_law_height);
            }
        }

        std::stringstream ss;
        ss << "1 // Number of levels" << '\n';
        ss << "1 // Number of boxes at this level" << '\n';
        ss << m_refine_box << '\n';

        create_mesh_instance<RefineMesh>();
        auto box_refine =
            std::make_unique<kynema_sgf::CartBoxRefinement>(sim());
        box_refine->read_inputs(mesh(), ss);
        if (mesh<RefineMesh>() != nullptr) {
            mesh<RefineMesh>()->refine_criteria_vec().push_back(
                std::move(box_refine));
        }
    }

    //! Set up the two-level mesh with plane-uniform logarithmic profiles and
    //! fill the wall-model ghost cells of velocity and temperature
    void init_wall_fields()
    {
        populate_parameters();
        initialize_mesh();
        ASSERT_EQ(sim().repo().num_active_levels(), 2);

        auto& pde_mgr = sim().pde_manager();
        pde_mgr.register_icns();
        sim().create_turbulence_model();
        sim().init_physics();

        auto& repo = sim().repo();
        auto& velocity = repo.get_field("velocity");
        auto& temperature = repo.get_field("temperature");
        auto& density = repo.get_field("density");
        auto& vel_mueff = repo.get_field("velocity_mueff");
        auto& temp_mueff = repo.get_field("temperature_mueff");

        // Logarithmic wind and temperature profiles, uniform in every plane
        const amrex::Real wind_cos = std::cos(m_wind_angle);
        const amrex::Real wind_sin = std::sin(m_wind_angle);
        init_log_profile(
            velocity, {0.0_rt, 0.0_rt, 0.0_rt},
            {m_ustar / m_kappa * wind_cos, m_ustar / m_kappa * wind_sin,
             0.0_rt},
            m_z0);
        init_log_profile(
            temperature, {m_theta0, 0.0_rt, 0.0_rt},
            {m_thetastar / m_kappa, 0.0_rt, 0.0_rt}, m_z0);
        if (m_covered_scale != 1.0_rt) {
            // Level-0 cells covered by level 1, which the means of level 0
            // must not see
            scale_covered_cells(velocity, 0, m_covered_scale);
            scale_covered_cells(temperature, 0, m_covered_scale);
        }
        density.setVal(1.0_rt);
        vel_mueff.setVal(m_mu);
        temp_mueff.setVal(m_mu);

        // Wall function: plane averages, friction velocity, custom BCs
        for (auto& pp : sim().physics()) {
            pp->post_init_actions();
        }
        pde_mgr.advance_states();

        // Fill the wall-model ghost cells of velocity and temperature
        velocity.apply_bc_funcs(kynema_sgf::FieldState::Old);
        temperature.apply_bc_funcs(kynema_sgf::FieldState::Old);
    }

    //! Check that the mean wall stress and heat flux of every level match
    //! the friction velocity and the surface heat flux of the wall function
    void check_level_fluxes()
    {
        constexpr amrex::Real tol =
            std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

        init_wall_fields();

        auto& repo = sim().repo();
        const auto& velocity = repo.get_field("velocity");
        const auto& temperature = repo.get_field("temperature");
        const auto& density = repo.get_field("density");
        const auto& vel_mueff = repo.get_field("velocity_mueff");
        const auto& temp_mueff = repo.get_field("temperature_mueff");
        const amrex::Real wind_cos = std::cos(m_wind_angle);
        const amrex::Real wind_sin = std::sin(m_wind_angle);

        const auto& abl = sim().physics_manager().get<kynema_sgf::ABL>();
        const auto& mo = abl.abl_wall_function().mo();
        ASSERT_GT(mo.utau, 0.0_rt);
        const amrex::Real utau2 = mo.utau * mo.utau;
        // The heat flux is a small difference of temperatures of order
        // theta0, so its roundoff scales with theta0
        const amrex::Real tol_q = tol * m_theta0 * mo.utau;

        // Every level must return the same mean stress, u_*^2 along the mean
        // wind, and the same mean heat flux, the specified surface flux (the
        // ghost value is the wall-normal gradient, so the flux is minus it)
        for (int lev = 0; lev < 2; ++lev) {
            amrex::Real ncells = 0.0_rt;
            const amrex::Real taux =
                wall_flux_mean(velocity, vel_mueff, density, 0, lev, ncells);
            // Both levels must own part of the wall
            ASSERT_GT(ncells, 0.0_rt) << "level " << lev;
            const amrex::Real tauy =
                wall_flux_mean(velocity, vel_mueff, density, 1, lev, ncells);
            const amrex::Real qwall = -wall_flux_mean(
                temperature, temp_mueff, density, 0, lev, ncells);
            EXPECT_NEAR(taux, utau2 * wind_cos, tol * utau2) << "level " << lev;
            EXPECT_NEAR(tauy, utau2 * wind_sin, tol * utau2) << "level " << lev;
            EXPECT_NEAR(qwall, m_qwall, tol_q) << "level " << lev;
        }
    }

    //! Check that the Donelan model uses the mean wind at the reference
    //! height on every level: the stress of the wall cells of each level is
    //! Cd(U_ref) |u_h| u_h with the drag coefficient of the reference-height
    //! mean wind, not of the first-cell mean wind of that level
    void check_donelan_reference_height()
    {
        constexpr amrex::Real tol =
            std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

        init_wall_fields();

        auto& repo = sim().repo();
        const auto& velocity = repo.get_field("velocity");
        const auto& density = repo.get_field("density");
        const auto& vel_mueff = repo.get_field("velocity_mueff");
        const amrex::Real wind_cos = std::cos(m_wind_angle);
        const amrex::Real wind_sin = std::sin(m_wind_angle);

        const auto& abl = sim().physics_manager().get<kynema_sgf::ABL>();
        const auto& wall_func = abl.abl_wall_function();
        // The mesh has per-level mean quantities, which Donelan must not use
        ASSERT_NE(&wall_func.mo(1), &wall_func.mo());
        const amrex::Real wspd_ref = wall_func.mo().vmag_mean;
        // Drag coefficient of ShearStressDonelan, in its linear range for
        // the mean wind of the reference height and of both first cells
        ASSERT_GT(wspd_ref, 5.0_rt);
        ASSERT_LT(wspd_ref, 25.0_rt);
        const amrex::Real cd = 0.001_rt + (7.0e-5_rt * (wspd_ref - 5.0_rt));

        for (int lev = 0; lev < 2; ++lev) {
            // Wind speed of the first cell of this level, uniform in plane
            const amrex::Real z1 =
                0.5_rt * repo.mesh().Geom(lev).CellSizeArray()[2];
            const amrex::Real wspd = m_ustar / m_kappa * std::log(z1 / m_z0);
            ASSERT_GT(wspd, 5.0_rt) << "level " << lev;
            const amrex::Real tau = cd * wspd * wspd;
            amrex::Real ncells = 0.0_rt;
            const amrex::Real taux =
                wall_flux_mean(velocity, vel_mueff, density, 0, lev, ncells);
            ASSERT_GT(ncells, 0.0_rt) << "level " << lev;
            const amrex::Real tauy =
                wall_flux_mean(velocity, vel_mueff, density, 1, lev, ncells);
            EXPECT_NEAR(taux, tau * wind_cos, tol * tau) << "level " << lev;
            EXPECT_NEAR(tauy, tau * wind_sin, tol * tau) << "level " << lev;
        }
    }

    //! Check that the mean quantities of every level are those of the
    //! wall-adjacent cells that the level owns, with the level-0 cells
    //! covered by level 1 set to other values
    void check_covered_cells_excluded()
    {
        constexpr amrex::Real tol =
            std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

        m_covered_scale = 2.0_rt;
        init_wall_fields();

        const auto& mesh = sim().repo().mesh();
        const auto& abl = sim().physics_manager().get<kynema_sgf::ABL>();
        const auto& wall_func = abl.abl_wall_function();
        const amrex::Real wind_cos = std::cos(m_wind_angle);
        const amrex::Real wind_sin = std::sin(m_wind_angle);
        for (int lev = 0; lev < 2; ++lev) {
            const auto& mo_lev = wall_func.mo(lev);
            ASSERT_NE(&mo_lev, &wall_func.mo()) << "level " << lev;
            // Plane-uniform first-cell values of the cells the level owns
            const amrex::Real z1 = 0.5_rt * mesh.Geom(lev).CellSizeArray()[2];
            const amrex::Real wspd = m_ustar / m_kappa * std::log(z1 / m_z0);
            const amrex::Real theta =
                m_theta0 + (m_thetastar / m_kappa * std::log(z1 / m_z0));
            EXPECT_NEAR(mo_lev.zref, z1, tol * z1) << "level " << lev;
            EXPECT_NEAR(mo_lev.vel_mean[0], wspd * wind_cos, tol * wspd)
                << "level " << lev;
            EXPECT_NEAR(mo_lev.vel_mean[1], wspd * wind_sin, tol * wspd)
                << "level " << lev;
            EXPECT_NEAR(mo_lev.vmag_mean, wspd, tol * wspd) << "level " << lev;
            EXPECT_NEAR(
                mo_lev.Su_mean, wspd * wspd * wind_cos, tol * wspd * wspd)
                << "level " << lev;
            EXPECT_NEAR(
                mo_lev.Sv_mean, wspd * wspd * wind_sin, tol * wspd * wspd)
                << "level " << lev;
            EXPECT_NEAR(mo_lev.theta_mean, theta, tol * m_theta0)
                << "level " << lev;
        }
    }

    //! Level-1 box (xlo ylo zlo xhi yhi zhi), by default over x < 60, all
    //! y and z < 250
    std::string m_refine_box{"0.0 0.0 0.0 60.0 120.0 250.0"};
    std::string m_shear_stress_type{"moeng"};
    //! Factor on the level-0 cells covered by level 1 (1: plane uniform)
    amrex::Real m_covered_scale{1.0_rt};
    //! Reference height of the plane averages, first-cell height if <= 0
    amrex::Real m_log_law_height{0.0_rt};
    amrex::Real m_ustar{0.5_rt};
    const amrex::Real m_thetastar{-0.05_rt};
    const amrex::Real m_wind_angle{0.5_rt};
    const amrex::Real m_theta0{300.0_rt};
    const amrex::Real m_mu{0.01_rt};
    const amrex::Real m_kappa{0.41_rt};
    const amrex::Real m_z0{0.1_rt};
    const amrex::Real m_qwall{0.02_rt};
};

TEST_F(ABLWallRefinementTest, moeng_wall_model_level_consistent)
{
    m_shear_stress_type = "moeng";
    check_level_fluxes();
}

TEST_F(ABLWallRefinementTest, schumann_wall_model_level_consistent)
{
    m_shear_stress_type = "schumann";
    check_level_fluxes();
}

TEST_F(ABLWallRefinementTest, constant_wall_model_level_consistent)
{
    m_shear_stress_type = "constant";
    check_level_fluxes();
}

TEST_F(ABLWallRefinementTest, donelan_wall_model_keeps_reference_height)
{
    // Donelan selects its drag coefficient from the mean wind at the
    // reference height, here 10 m as in the hurricane boundary layer case,
    // with a wind strong enough for the linear range of the coefficient
    m_shear_stress_type = "donelan";
    m_log_law_height = 10.0_rt;
    m_ustar = 1.0_rt;
    check_donelan_reference_height();
}

TEST_F(ABLWallRefinementTest, level_means_exclude_covered_wall_cells)
{
    check_covered_cells_excluded();
}

TEST_F(ABLWallRefinementTest, refinement_aloft_keeps_reference_height)
{
    // Level-1 box that does not reach the wall: level 0 alone owns the
    // wall and keeps the plane averages at the reference height, as on a
    // single-level mesh
    m_refine_box = "0.0 0.0 500.0 60.0 120.0 750.0";
    init_wall_fields();

    const auto& abl = sim().physics_manager().get<kynema_sgf::ABL>();
    const auto& wall_func = abl.abl_wall_function();
    EXPECT_EQ(&wall_func.mo(0), &wall_func.mo());
    EXPECT_EQ(&wall_func.mo(1), &wall_func.mo());
}

} // namespace kynema_sgf_tests
