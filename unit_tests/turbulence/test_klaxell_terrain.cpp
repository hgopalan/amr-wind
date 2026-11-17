#include "ks_test_utils/MeshTest.H"
#include "ks_test_utils/test_utils.H"
#include "src/physics/TerrainDrag.H"
#include "src/equation_systems/temperature/source_terms/DragTempForcing.H"
#include "src/turbulence/TurbulenceModel.H"
#include "src/turbulence/RANS/KLAxell.H"
#include "src/wind_energy/SurfaceStability.H"
#include "src/utilities/math_ops.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include <cmath>
#include <limits>
#include <vector>

using namespace amrex::literals;

namespace {
// Plateau of height h (100 m by default) for x in [449, 576], flat ground
// elsewhere
void write_terrain(const std::string& fname, const amrex::Real h = 100.0_rt)
{
    std::ofstream os(fname);
    os << "6\n2\n";
    os << "0.0\n448.0\n449.0\n576.0\n577.0\n1024.0\n";
    os << "0.0\n1024.0\n";
    os << "0.0\n0.0\n0.0\n0.0\n";
    for (int n = 0; n < 4; ++n) {
        os << h << "\n";
    }
    os << "0.0\n0.0\n0.0\n0.0\n";
}

// z u v T tke, read by KransAxell for the mesoscale sponge
void write_rans_profile(const std::string& fname)
{
    std::ofstream os(fname);
    os << "0 8 0 300 0.1\n1000 8 0 300 0.1\n4000 8 0 300 0.1\n";
}

void set_bool(const std::string& prefix, const char* key, const bool v)
{
    amrex::ParmParse pp(prefix);
    pp.remove(key);
    pp.add(key, v);
}

void set_string(
    const std::string& prefix, const char* key, const std::string& v)
{
    amrex::ParmParse pp(prefix);
    pp.remove(key);
    pp.add(key, v);
}

//! Velocity (s z, v, 0) including ghost cells
void init_shear(
    kynema_sgf::Field& vel, const amrex::Real s, const amrex::Real v)
{
    const auto& mesh = vel.repo().mesh();
    const int nlevels = vel.repo().num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& dx = mesh.Geom(lev).CellSizeArray();
        const auto& problo = mesh.Geom(lev).ProbLoArray();
        const auto& varrs = vel(lev).arrays();
        amrex::ParallelFor(
            vel(lev), vel.num_grow(),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                const amrex::Real z = problo[2] + ((k + 0.5_rt) * dx[2]);
                varrs[nbx](i, j, k, 0) = s * z;
                varrs[nbx](i, j, k, 1) = v;
                varrs[nbx](i, j, k, 2) = 0.0_rt;
            });
    }
    amrex::Gpu::streamSynchronize();
}

//! Temperature T0 + dTdz z including ghost cells
void init_temperature(
    kynema_sgf::Field& temp, const amrex::Real T0, const amrex::Real dTdz)
{
    const auto& mesh = temp.repo().mesh();
    const int nlevels = temp.repo().num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& dx = mesh.Geom(lev).CellSizeArray();
        const auto& problo = mesh.Geom(lev).ProbLoArray();
        const auto& tarrs = temp(lev).arrays();
        amrex::ParallelFor(
            temp(lev), temp.num_grow(),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                const amrex::Real z = problo[2] + ((k + 0.5_rt) * dx[2]);
                tarrs[nbx](i, j, k) = T0 + (dTdz * z);
            });
    }
    amrex::Gpu::streamSynchronize();
}

//! Horizontal velocity u_s in the blanked cells, including ghost cells
void set_blank_velocity(
    kynema_sgf::Field& vel,
    const kynema_sgf::IntField& blank,
    const amrex::Real us)
{
    const int nlevels = vel.repo().num_active_levels();
    const amrex::IntVect ng = amrex::min(vel.num_grow(), blank.num_grow());
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& varrs = vel(lev).arrays();
        const auto& barrs = blank(lev).const_arrays();
        amrex::ParallelFor(
            vel(lev), ng, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                if (barrs[nbx](i, j, k) == 1) {
                    varrs[nbx](i, j, k, 0) = us;
                    varrs[nbx](i, j, k, 1) = us;
                }
            });
    }
    amrex::Gpu::streamSynchronize();
}
} // namespace

namespace kynema_sgf_tests {

/** KLAxell with the TerrainDrag fields: plateau terrain on a 32 x 32 x 16
 *  mesh (dx = dy = dz = 32 m), neutral shear flow (s z, v, 0), uniform TKE.
 *  On the plateau (i = 14 ... 17) the cells k = 0, 1, 2 are blanked and
 *  k = 3 is the drag cell.
 */
class KLAxellTerrainTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();
        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{32, 32, 16}};
            pp.addarr("n_cell", ncell);
            pp.add("blocking_factor", 2);
            if (m_max_grid_size > 0) {
                pp.add("max_grid_size", m_max_grid_size);
            }
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> probhi{{1024.0_rt, 1024.0_rt, 512.0_rt}};
            pp.addarr("prob_hi", probhi);
        }
        {
            amrex::ParmParse pp("turbulence");
            pp.add("model", (std::string) "KLAxell");
        }
        {
            amrex::ParmParse pp("incflo");
            amrex::Vector<std::string> physics{"ABL"};
            pp.addarr("physics", physics);
            pp.add("density", m_rho0);
            amrex::Vector<amrex::Real> vvec{8.0, 0.0, 0.0};
            pp.addarr("velocity", vvec);
            amrex::Vector<amrex::Real> gvec{0.0, 0.0, -9.81};
            pp.addarr("gravity", gvec);
        }
        {
            amrex::ParmParse pp("transport");
            pp.add("viscosity", m_mu_lam);
            pp.add("reference_temperature", 300.0_rt);
            pp.add("thermal_expansion_coefficient", m_beta);
        }
        {
            amrex::ParmParse pp("ABL");
            pp.add("initial_wind_profile", true);
            pp.add("rans_1dprofile_file", (std::string) "rans_1d.info");
            amrex::Vector<amrex::Real> hts{0.0_rt, 100.0_rt, 4000.0_rt};
            pp.addarr("temperature_heights", hts);
            pp.addarr("wind_heights", hts);
            amrex::Vector<amrex::Real> t_vals{300.0_rt, 300.0_rt, 300.0_rt};
            pp.addarr("temperature_values", t_vals);
            amrex::Vector<amrex::Real> u_vals{8.0_rt, 8.0_rt, 8.0_rt};
            pp.addarr("u_values", u_vals);
            amrex::Vector<amrex::Real> v_vals{0.0_rt, 0.0_rt, 0.0_rt};
            pp.addarr("v_values", v_vals);
            amrex::Vector<amrex::Real> tke_vals{0.1_rt, 0.1_rt, 0.1_rt};
            pp.addarr("tke_values", tke_vals);
            pp.add("surface_temp_flux", 0.0_rt);
            pp.add("surface_roughness_z0", m_z0);
            // Keeps the neutral length scale and disables the sponge
            pp.add("meso_sponge_start", 1.0e5_rt);
        }
        {
            amrex::ParmParse pp("TerrainDrag");
            // The roughness without a roughness file is 0.1 m = m_z0
            pp.add("terrain_file", (std::string) "terrain.amrwind");
        }
        {
            amrex::ParmParse pp("DragTempForcing");
            pp.add("soil_temperature", m_soil_temperature);
        }
    }

    void setup()
    {
        write_terrain("terrain.amrwind", m_plateau_height);
        write_rans_profile("rans_1d.info");
        populate_parameters();
        initialize_mesh();
        auto& pde_mgr = sim().pde_manager();
        pde_mgr.register_icns();
        sim().init_physics();
        sim().create_transport_model();
        if (m_with_terrain) {
            m_terrain =
                std::make_unique<kynema_sgf::terraindrag::TerrainDrag>(sim());
            const int nlevels = sim().repo().num_active_levels();
            for (int lev = 0; lev < nlevels; ++lev) {
                m_terrain->initialize_fields(
                    lev, sim().repo().mesh().Geom(lev));
            }
        }
        sim().create_turbulence_model();
        sim().turbulence_model().post_init_actions();
        init_shear(sim().repo().get_field("velocity"), m_shear, m_vspan);
        sim().repo().get_field("density").setVal(m_rho0);
        sim().repo().get_field("temperature").setVal(300.0_rt);
        sim().repo().get_field("tke").setVal(m_tke);
        sim().repo().get_field("turb_lscale").setVal(10.0_rt);
        sim().time().delta_t() = m_dt;
    }

    void update_viscosity()
    {
        sim().turbulence_model().update_turbulent_viscosity(
            kynema_sgf::FieldState::New, DiffusionType::Crank_Nicolson);
    }

    //! Resolved strain rate sqrt(shear_prod / mu_turb) of a cell
    [[nodiscard]] amrex::Real strain(const int i, const int j, const int k)
    {
        const auto& mu = sim().repo().get_field("mu_turb");
        const auto& prod = sim().repo().get_field("shear_prod");
        return std::sqrt(
            utils::field_probe(prod, 0, i, j, k) /
            utils::field_probe(mu, 0, i, j, k));
    }

    //! Neutral surface-layer mixing length at height z above the wall
    [[nodiscard]] static amrex::Real lscale(const amrex::Real z)
    {
        const amrex::Real lambda = 30.0_rt;
        const amrex::Real kappa = 0.41_rt;
        return (lambda * kappa * z) / (lambda + (kappa * z));
    }

    //! Neutral KLAxell viscosity rho Cmu l(z) sqrt(k)
    [[nodiscard]] amrex::Real mu_rans(const amrex::Real z) const
    {
        return m_rho0 * m_Cmu * lscale(z) * std::sqrt(m_tke);
    }

    [[nodiscard]] amrex::Real mu(const int i, const int j, const int k)
    {
        return utils::field_probe(
            sim().repo().get_field("mu_turb"), 0, i, j, k);
    }

    //! Drag-cell viscosity of terrain_face_stress for the stability
    //! correction psi of the reference height
    void check_face_stress(const amrex::Real psi)
    {
        const amrex::Real mref =
            std::sqrt((u_shear(4) * u_shear(4)) + (m_vspan * m_vspan));
        const amrex::Real us =
            0.41_rt * mref / (std::log(1.5_rt * m_dz / m_z0) - psi);
        const amrex::Real mu_face = m_rho0 * us * us * m_dz / (m_shear * m_dz);
        const amrex::Real mu_above = mu_rans(44.0_rt);
        EXPECT_NEAR(mu(15, 10, 4), mu_above, m_tol * mu_above);
        EXPECT_NEAR(
            mu(15, 10, 3), (2.0_rt * mu_face) - mu_above, m_tol * mu_face);
        EXPECT_NEAR(mu(5, 5, 8), mu_rans(272.0_rt), m_tol * mu_rans(272.0_rt));
    }

    //! Checks of the original path (every option off) at each cell an
    //! option changes, with a stable profile and a blanked velocity
    void check_original_path()
    {
        init_stable_temperature();
        const amrex::Real us = -3.0_rt;
        set_blank_velocity(
            sim().repo().get_field("velocity"),
            sim().repo().get_int_field("terrain_blank"), us);
        update_viscosity();
        // Central difference into the blanked cell (terrain_wall_stencil)
        const amrex::Real dudz = (u_shear(4) - us) / (2.0_rt * m_dz);
        const amrex::Real dvdz = (m_vspan - us) / (2.0_rt * m_dz);
        EXPECT_NEAR(
            strain(15, 10, 3), std::sqrt((dudz * dudz) + (dvdz * dvdz)),
            m_tol * m_shear);
        // Mixing length from the terrain height (terrain_blanked_face_length)
        EXPECT_NEAR(mu(15, 10, 4), mu_rans(44.0_rt), m_tol * mu_rans(44.0_rt));
        // Model viscosity in the drag cell (terrain_face_stress)
        EXPECT_NEAR(mu(15, 10, 3), mu_rans(16.0_rt), m_tol * mu_rans(16.0_rt));
        // Model heat diffusivity in the drag cell (terrain_face_heat_flux)
        const amrex::Real a3 = utils::field_probe(update_alpha(), 0, 15, 10, 3);
        EXPECT_NEAR(a3, model_alpha(15, 10, 3), m_tol * a3);
        // Soil relaxation of the blanked cells (blank_follow_fluid)
        set_blank_velocity(
            sim().repo().get_field("velocity"),
            sim().repo().get_int_field("terrain_blank"), 0.0_rt);
        const amrex::Real T1 = 300.0_rt + (m_dTdz * 1.5_rt * m_dz);
        const amrex::Real soil = -blank_drag_rate() * (T1 - m_soil_temperature);
        EXPECT_NEAR(
            utils::field_probe(drag_temp_source("src"), 0, 15, 10, 1), soil,
            m_tol * std::abs(soil));
    }

    //! Stable surface layer: ABL.wall_het_model = mol with the Obukhov
    //! length m_L and a potential temperature rising at m_dTdz
    void set_stable_mol()
    {
        amrex::ParmParse pp("ABL");
        pp.add("wall_het_model", (std::string) "mol");
        pp.add("monin_obukhov_length", m_L);
    }

    void init_stable_temperature()
    {
        init_temperature(
            sim().repo().get_field("temperature"), 300.0_rt, m_dTdz);
    }

    //! Update and return the effective thermal diffusivity
    kynema_sgf::Field& update_alpha()
    {
        // update_alphaeff fills the ghost cells, so the field needs its BCs
        auto& alphaeff = sim().repo().declare_cc_field("alphaeff", 1, 1, 1);
        alphaeff.set_default_fillpatch_bc(sim().time());
        alphaeff.setVal(0.0_rt);
        sim().turbulence_model().update_alphaeff(alphaeff);
        return alphaeff;
    }

    //! alpha_lam + mu_t / Pr_t(Rt) of the default update_alphaeff: Rt is
    //! the one of the eddy viscosity, which the neutral switch sets to 0
    //! (ABL.surface_temp_flux = 0 below ABL.meso_sponge_start), so Pr_t = 1
    [[nodiscard]] amrex::Real model_alpha(const int i, const int j, const int k)
    {
        return m_mu_lam + mu(i, j, k);
    }

    //! alpha_lam + mu_t / Pr_t(Rt) of KLAxell_coeffs.legacy_alphaeff, Rt
    //! recomputed from the length scale for the uniform stratification of
    //! init_stable_temperature
    [[nodiscard]] amrex::Real
    legacy_alpha(const int i, const int j, const int k)
    {
        const amrex::Real N2 = 9.81_rt * m_beta * m_dTdz;
        const amrex::Real tke =
            utils::field_probe(sim().repo().get_field("tke"), 0, i, j, k);
        const amrex::Real l = utils::field_probe(
            sim().repo().get_field("turb_lscale"), 0, i, j, k);
        const amrex::Real eps =
            kynema_sgf::utils::powi(m_Cmu, 3) * std::pow(tke, 1.5_rt) / l;
        const amrex::Real Rt = kynema_sgf::utils::powi(tke / eps, 2) * N2;
        const amrex::Real prt =
            (1.0_rt + (0.193_rt * Rt)) / (1.0_rt + (0.0302_rt * Rt));
        return m_mu_lam + (mu(i, j, k) / prt);
    }

    //! DragTempForcing source of a cell with the current inputs
    kynema_sgf::Field& drag_temp_source(const std::string& name)
    {
        auto& src = sim().repo().declare_cc_field(name, 1, 0);
        src.setVal(0.0_rt);
        const kynema_sgf::pde::temperature::DragTempForcing forcing(sim());
        const int nlevels = sim().repo().num_active_levels();
        for (int lev = 0; lev < nlevels; ++lev) {
            forcing(lev, kynema_sgf::FieldState::New, src(lev));
        }
        return src;
    }

    //! Explicit drag rate of a blanked cell at rest, min(Cd / (dz |u|), 10 /
    //! dz) = 10 / dz
    [[nodiscard]] amrex::Real blank_drag_rate() const { return 10.0_rt / m_dz; }

    //! u of the shear flow at the center of level k
    [[nodiscard]] amrex::Real u_shear(const int k) const
    {
        return m_shear * (k + 0.5_rt) * m_dz;
    }

    //! setup() must abort with a message that contains msg
    void expect_abort(const std::string& msg)
    {
        try {
            setup();
            ADD_FAILURE() << "setup() did not abort";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(msg), std::string::npos)
                << e.what();
        }
    }

    std::unique_ptr<kynema_sgf::terraindrag::TerrainDrag> m_terrain;
    bool m_with_terrain{true};
    //! amr.max_grid_size when > 0
    int m_max_grid_size{0};
    amrex::Real m_plateau_height{100.0_rt};
    const amrex::Real m_dz{32.0_rt};
    const amrex::Real m_dt{0.5_rt};
    const amrex::Real m_rho0{1.2_rt};
    const amrex::Real m_z0{0.1_rt};
    const amrex::Real m_shear{0.05_rt};
    const amrex::Real m_vspan{2.0_rt};
    const amrex::Real m_tke{0.1_rt};
    const amrex::Real m_Cmu{0.556_rt};
    const amrex::Real m_mu_lam{1.0e-5_rt};
    const amrex::Real m_beta{1.0_rt / 300.0_rt};
    const amrex::Real m_L{200.0_rt};
    const amrex::Real m_dTdz{0.01_rt};
    const amrex::Real m_soil_temperature{290.0_rt};
    const amrex::Real m_tol{
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt};
};

// The wall is the face of the blanked cell: the drag cell on the plateau
// (15, 10, 3) has blanked cells below, and the linear shear flow gives the
// exact shear with the one-sided stencil whatever the blanked velocity. The
// flat-ground cells next to the plateau sides (13 and 18, k = 1) see a wall
// normal to x, through which u must vanish: du/dx = -+ 4 u / (3 dx) = -+ 2 s,
// so sqrt(2 (du/dx)^2 + (du/dz)^2) = 3 s. Far from the terrain the strain
// rate is unchanged.
TEST_F(KLAxellTerrainTest, wall_stencil_uses_the_flat_ground_stencil)
{
    set_bool("KLAxell", "terrain_wall_stencil", true);
    setup();
    for (const amrex::Real us : {-3.0_rt, 5.0_rt}) {
        set_blank_velocity(
            sim().repo().get_field("velocity"),
            sim().repo().get_int_field("terrain_blank"), us);
        update_viscosity();
        EXPECT_NEAR(strain(15, 10, 3), m_shear, m_tol * m_shear);
        EXPECT_NEAR(strain(13, 10, 1), 3.0_rt * m_shear, m_tol * m_shear);
        EXPECT_NEAR(strain(18, 10, 1), 3.0_rt * m_shear, m_tol * m_shear);
        EXPECT_NEAR(strain(5, 5, 8), m_shear, m_tol * m_shear);
    }
}

// The plateau (h = 100 m) blanks the cells up to k = 2, whose top face is at
// 96 m. With the option the mixing length is measured from that face: 48 m
// above it at k = 4 (144 m), not 44 m above the terrain height. The drag cell
// keeps the floor dz / 2 and flat ground (h = 0) is unchanged.
TEST_F(KLAxellTerrainTest, blanked_face_length_measures_from_the_face)
{
    set_bool("KLAxell", "terrain_blanked_face_length", true);
    setup();
    update_viscosity();
    EXPECT_NEAR(mu(15, 10, 4), mu_rans(48.0_rt), m_tol * mu_rans(48.0_rt));
    EXPECT_NEAR(mu(15, 10, 5), mu_rans(80.0_rt), m_tol * mu_rans(80.0_rt));
    EXPECT_NEAR(mu(15, 10, 3), mu_rans(16.0_rt), m_tol * mu_rans(16.0_rt));
    EXPECT_NEAR(mu(5, 5, 8), mu_rans(272.0_rt), m_tol * mu_rans(272.0_rt));
}

// A 120 m plateau lies in the upper half of its cell: the cells up to k = 3
// (center 112 m) are blanked and the face is at 128 m, which the rounding to
// the nearest face gives and truncation (96 m) does not. The cell at k = 5
// (176 m) is 48 m above the face, not 56 m above the terrain height; the
// drag cell k = 4 keeps the floor dz / 2.
TEST_F(KLAxellTerrainTest, blanked_face_length_rounds_to_the_nearest_face)
{
    m_plateau_height = 120.0_rt;
    set_bool("KLAxell", "terrain_blanked_face_length", true);
    setup();
    ASSERT_EQ(
        utils::field_probe(
            sim().repo().get_int_field("terrain_blank"), 0, 15, 10, 3),
        1);
    ASSERT_EQ(
        utils::field_probe(
            sim().repo().get_int_field("terrain_drag"), 0, 15, 10, 4),
        1);
    update_viscosity();
    EXPECT_NEAR(mu(15, 10, 5), mu_rans(48.0_rt), m_tol * mu_rans(48.0_rt));
    EXPECT_NEAR(mu(15, 10, 6), mu_rans(80.0_rt), m_tol * mu_rans(80.0_rt));
    EXPECT_NEAR(mu(15, 10, 4), mu_rans(16.0_rt), m_tol * mu_rans(16.0_rt));
}

// The drag cell (15, 10, 3) takes the viscosity for which the face above it,
// the mean of the two cells, carries u*^2 of the DragForcing wall law from
// the speed of the cell above (1.5 dz from the wall) over the velocity
// difference s dz of the shear flow. The cells above and flat ground are
// unchanged.
TEST_F(KLAxellTerrainTest, face_stress_sets_the_drag_cell_viscosity)
{
    set_bool("KLAxell", "terrain_face_stress", true);
    setup();
    update_viscosity();
    check_face_stress(0.0_rt);
}

// Under mol the friction velocity includes psi_m(1.5 dz / L), -5 zeta for a
// stable L.
TEST_F(KLAxellTerrainTest, face_stress_uses_the_stability_function)
{
    set_bool("KLAxell", "terrain_face_stress", true);
    const amrex::Real L = 200.0_rt;
    {
        amrex::ParmParse pp("ABL");
        pp.add("wall_het_model", (std::string) "mol");
        pp.add("monin_obukhov_length", L);
    }
    setup();
    update_viscosity();
    check_face_stress(-5.0_rt * 1.5_rt * m_dz / L);
}

// Under mol the drag cell takes the heat diffusivity for which the face above
// it carries the MOST heat flux q = -u* theta*, theta* = T u*^2 / (kappa g L),
// over the temperature difference of the stable profile (-dTdz dz, down the
// gradient of q < 0). The cell above and flat ground keep the model value.
TEST_F(KLAxellTerrainTest, face_heat_flux_sets_the_drag_cell_diffusivity)
{
    set_bool("KLAxell", "terrain_face_heat_flux", true);
    set_stable_mol();
    setup();
    init_stable_temperature();
    update_viscosity();
    const auto& alpha = update_alpha();
    const auto a = [&](const int i, const int j, const int k) {
        return utils::field_probe(alpha, 0, i, j, k);
    };
    const amrex::Real psi = -5.0_rt * 1.5_rt * m_dz / m_L;
    const amrex::Real mref =
        std::sqrt((u_shear(4) * u_shear(4)) + (m_vspan * m_vspan));
    const amrex::Real us =
        0.41_rt * mref / (std::log(1.5_rt * m_dz / m_z0) - psi);
    const amrex::Real T = 300.0_rt + (m_dTdz * 3.5_rt * m_dz);
    const amrex::Real q = -us * T * us * us / (0.41_rt * 9.81_rt * m_L);
    const amrex::Real a_face = m_rho0 * q * m_dz / (-m_dTdz * m_dz);
    const amrex::Real expected = (2.0_rt * a_face) - a(15, 10, 4);
    EXPECT_GT(expected, m_mu_lam);
    EXPECT_NEAR(a(15, 10, 3), expected, m_tol * expected);
    EXPECT_NEAR(a(15, 10, 4), model_alpha(15, 10, 4), m_tol * a(15, 10, 4));
    EXPECT_NEAR(a(5, 5, 8), model_alpha(5, 5, 8), m_tol * a(5, 5, 8));
}

// Without mol there is no Obukhov length: the option turned on by the
// improved treatment does nothing (set on its own it aborts).
TEST_F(KLAxellTerrainTest, face_heat_flux_needs_mol)
{
    set_string("TerrainDrag", "wall_treatment", "improved");
    setup();
    init_stable_temperature();
    update_viscosity();
    const auto& alpha = update_alpha();
    const amrex::Real a3 = utils::field_probe(alpha, 0, 15, 10, 3);
    EXPECT_NEAR(a3, model_alpha(15, 10, 3), m_tol * a3);
}

// Asked for on its own without mol, the face heat flux aborts.
TEST_F(KLAxellTerrainTest, explicit_face_heat_flux_without_mol_aborts)
{
    set_bool("KLAxell", "terrain_face_heat_flux", true);
    expect_abort("terrain_face_heat_flux needs ABL.wall_het_model");
}

// The options read the TerrainDrag fields: without TerrainDrag they abort.
TEST_F(KLAxellTerrainTest, options_without_terrain_drag_abort)
{
    m_with_terrain = false;
    set_bool("KLAxell", "terrain_wall_stencil", true);
    expect_abort("need TerrainDrag in incflo.physics");
}

// Unstable mol (L < 0) with the potential temperature falling with height:
// the upward flux q = -u* theta* > 0 runs down the gradient of the profile,
// and psi_m is the unstable branch.
TEST_F(KLAxellTerrainTest, face_heat_flux_unstable)
{
    const amrex::Real L = -200.0_rt;
    set_bool("KLAxell", "terrain_face_heat_flux", true);
    {
        amrex::ParmParse pp("ABL");
        pp.add("wall_het_model", (std::string) "mol");
        pp.add("monin_obukhov_length", L);
    }
    setup();
    init_temperature(sim().repo().get_field("temperature"), 300.0_rt, -m_dTdz);
    update_viscosity();
    const auto& alpha = update_alpha();
    const auto a = [&](const int i, const int j, const int k) {
        return utils::field_probe(alpha, 0, i, j, k);
    };
    // Default ABL.mo_beta_m = 16, ABL.mo_gamma_m = 5
    const amrex::Real psi = kynema_sgf::surface_stability::psi_m(
        1.5_rt * m_dz / L, 16.0_rt, 5.0_rt);
    EXPECT_GT(psi, 0.0_rt);
    const amrex::Real mref =
        std::sqrt((u_shear(4) * u_shear(4)) + (m_vspan * m_vspan));
    const amrex::Real us =
        0.41_rt * mref / (std::log(1.5_rt * m_dz / m_z0) - psi);
    const amrex::Real T = 300.0_rt - (m_dTdz * 3.5_rt * m_dz);
    const amrex::Real q = -us * T * us * us / (0.41_rt * 9.81_rt * L);
    EXPECT_GT(q, 0.0_rt);
    const amrex::Real a_face = m_rho0 * q * m_dz / (m_dTdz * m_dz);
    const amrex::Real expected = (2.0_rt * a_face) - a(15, 10, 4);
    EXPECT_GT(expected, m_mu_lam);
    EXPECT_NEAR(a(15, 10, 3), expected, m_tol * expected);
    EXPECT_NEAR(a(15, 10, 4), model_alpha(15, 10, 4), m_tol * a(15, 10, 4));
}

// A temperature difference against the flux (stable L, unstable profile)
// or below 1e-4 K (uniform temperature) leaves the model value.
TEST_F(KLAxellTerrainTest, face_heat_flux_skips_counter_gradient_and_uniform)
{
    set_bool("KLAxell", "terrain_face_heat_flux", true);
    set_stable_mol();
    setup();
    for (const amrex::Real dTdz : {-m_dTdz, 0.0_rt}) {
        init_temperature(sim().repo().get_field("temperature"), 300.0_rt, dTdz);
        update_viscosity();
        const amrex::Real a3 = utils::field_probe(update_alpha(), 0, 15, 10, 3);
        EXPECT_NEAR(a3, model_alpha(15, 10, 3), m_tol * a3);
    }
}

// The blanked cells relax toward the temperature of the cell above, at the
// exactly integrated rate (1 - exp(-C dt)) / dt, in place of the soil
// temperature; every other cell keeps the source of the default forcing.
TEST_F(KLAxellTerrainTest, blank_follow_fluid_relaxes_toward_the_cell_above)
{
    set_stable_mol();
    setup();
    init_stable_temperature();
    set_blank_velocity(
        sim().repo().get_field("velocity"),
        sim().repo().get_int_field("terrain_blank"), 0.0_rt);
    const auto& src_soil = drag_temp_source("src_soil");
    set_bool("DragTempForcing", "blank_follow_fluid", true);
    const auto& src_follow = drag_temp_source("src_follow");
    const auto T = [&](const int k) {
        return 300.0_rt + (m_dTdz * (k + 0.5_rt) * m_dz);
    };
    const amrex::Real C = blank_drag_rate();
    const amrex::Real C_e = (1.0_rt - std::exp(-C * m_dt)) / m_dt;
    for (const int k : {0, 1, 2}) {
        const amrex::Real expected = -C_e * (T(k) - T(k + 1));
        EXPECT_NEAR(
            utils::field_probe(src_follow, 0, 15, 10, k), expected,
            m_tol * std::abs(expected));
    }
    for (const auto& ijk :
         {amrex::IntVect(15, 10, 3), amrex::IntVect(5, 5, 0),
          amrex::IntVect(5, 5, 8)}) {
        EXPECT_EQ(
            utils::field_probe(src_follow, 0, ijk[0], ijk[1], ijk[2]),
            utils::field_probe(src_soil, 0, ijk[0], ijk[1], ijk[2]));
    }
}

// With every option at its default the model runs the unchanged TerrainDrag
// path: at each cell an option changes, the result is the legacy value.
TEST_F(KLAxellTerrainTest, defaults_leave_the_legacy_path_unchanged)
{
    // Stable mol surface layer, under which every option acts
    set_stable_mol();
    setup();
    check_original_path();
}

// With the improved treatment and all five options turned off, the result is
// that of the original treatment at every cell an option changes.
TEST_F(KLAxellTerrainTest, improved_wall_treatment_with_every_option_off)
{
    set_string("TerrainDrag", "wall_treatment", "improved");
    for (const char* key :
         {"terrain_wall_stencil", "terrain_blanked_face_length",
          "terrain_face_stress", "terrain_face_heat_flux"}) {
        set_bool("KLAxell", key, false);
    }
    set_bool("DragTempForcing", "blank_follow_fluid", false);
    set_stable_mol();
    setup();
    check_original_path();
}

// KLAxell_coeffs.legacy_alphaeff recomputes Rt in update_alphaeff; the
// face heat flux is applied on that path too, and the cells it does not
// touch keep the legacy value.
TEST_F(KLAxellTerrainTest, face_heat_flux_on_the_legacy_alphaeff_path)
{
    set_bool("KLAxell", "terrain_face_heat_flux", true);
    set_bool("KLAxell_coeffs", "legacy_alphaeff", true);
    set_stable_mol();
    setup();
    init_stable_temperature();
    update_viscosity();
    const auto& alpha = update_alpha();
    const auto a = [&](const int i, const int j, const int k) {
        return utils::field_probe(alpha, 0, i, j, k);
    };
    const amrex::Real psi = -5.0_rt * 1.5_rt * m_dz / m_L;
    const amrex::Real mref =
        std::sqrt((u_shear(4) * u_shear(4)) + (m_vspan * m_vspan));
    const amrex::Real us =
        0.41_rt * mref / (std::log(1.5_rt * m_dz / m_z0) - psi);
    const amrex::Real T = 300.0_rt + (m_dTdz * 3.5_rt * m_dz);
    const amrex::Real q = -us * T * us * us / (0.41_rt * 9.81_rt * m_L);
    const amrex::Real a_face = m_rho0 * q * m_dz / (-m_dTdz * m_dz);
    EXPECT_NE(legacy_alpha(15, 10, 4), model_alpha(15, 10, 4));
    EXPECT_NEAR(a(15, 10, 4), legacy_alpha(15, 10, 4), m_tol * a(15, 10, 4));
    const amrex::Real expected = (2.0_rt * a_face) - a(15, 10, 4);
    EXPECT_NEAR(a(15, 10, 3), expected, m_tol * expected);
    EXPECT_NEAR(a(5, 5, 8), legacy_alpha(5, 5, 8), m_tol * a(5, 5, 8));
}

// Boxes of 4 cells: the drag cell (15, 10, 3) is the top cell of its box and
// the cell above it, whose viscosity and diffusivity the face stress and
// heat flux read, is a ghost cell. Both are applied after the fill, so the
// result is the one of a single box.
TEST_F(KLAxellTerrainTest, face_corrections_across_a_box_boundary)
{
    m_max_grid_size = 4;
    set_bool("KLAxell", "terrain_face_stress", true);
    set_bool("KLAxell", "terrain_face_heat_flux", true);
    set_stable_mol();
    setup();
    ASSERT_GT(sim().repo().mesh().boxArray(0).size(), 1);
    init_stable_temperature();
    update_viscosity();
    check_face_stress(-5.0_rt * 1.5_rt * m_dz / m_L);
    const auto& alpha = update_alpha();
    const amrex::Real psi = -5.0_rt * 1.5_rt * m_dz / m_L;
    const amrex::Real mref =
        std::sqrt((u_shear(4) * u_shear(4)) + (m_vspan * m_vspan));
    const amrex::Real us =
        0.41_rt * mref / (std::log(1.5_rt * m_dz / m_z0) - psi);
    const amrex::Real T = 300.0_rt + (m_dTdz * 3.5_rt * m_dz);
    const amrex::Real q = -us * T * us * us / (0.41_rt * 9.81_rt * m_L);
    const amrex::Real a_face = m_rho0 * q * m_dz / (-m_dTdz * m_dz);
    const amrex::Real expected = (2.0_rt * a_face) - model_alpha(15, 10, 4);
    EXPECT_NEAR(
        utils::field_probe(alpha, 0, 15, 10, 3), expected, m_tol * expected);
}

// TerrainDrag.wall_treatment is original unless set to improved.
TEST_F(KLAxellTerrainTest, wall_treatment_defaults_to_original)
{
    EXPECT_FALSE(kynema_sgf::terraindrag::improved_wall_treatment());
    set_string("TerrainDrag", "wall_treatment", "original");
    EXPECT_FALSE(kynema_sgf::terraindrag::improved_wall_treatment());
    set_string("TerrainDrag", "wall_treatment", "improved");
    EXPECT_TRUE(kynema_sgf::terraindrag::improved_wall_treatment());
}

// TerrainDrag.wall_treatment = improved turns on the five options together,
// each acting as in its own test: the flat-ground stencil beside the plateau,
// the mixing length from the blanked face, the drag-cell viscosity and heat
// diffusivity from the face stress and heat flux (stable mol), and the
// blanked cells following the air above them.
TEST_F(KLAxellTerrainTest, improved_wall_treatment_turns_on_every_option)
{
    set_string("TerrainDrag", "wall_treatment", "improved");
    set_stable_mol();
    setup();
    set_blank_velocity(
        sim().repo().get_field("velocity"),
        sim().repo().get_int_field("terrain_blank"), -3.0_rt);
    update_viscosity();
    // terrain_wall_stencil
    EXPECT_NEAR(strain(13, 10, 1), 3.0_rt * m_shear, m_tol * m_shear);
    EXPECT_NEAR(strain(18, 10, 1), 3.0_rt * m_shear, m_tol * m_shear);
    // terrain_blanked_face_length
    const amrex::Real mu_above = mu_rans(48.0_rt);
    EXPECT_NEAR(mu(15, 10, 4), mu_above, m_tol * mu_above);
    // terrain_face_stress, with the mixing length above from the face
    const amrex::Real psi = -5.0_rt * 1.5_rt * m_dz / m_L;
    const amrex::Real mref =
        std::sqrt((u_shear(4) * u_shear(4)) + (m_vspan * m_vspan));
    const amrex::Real us =
        0.41_rt * mref / (std::log(1.5_rt * m_dz / m_z0) - psi);
    const amrex::Real mu_face = m_rho0 * us * us / m_shear;
    EXPECT_NEAR(mu(15, 10, 3), (2.0_rt * mu_face) - mu_above, m_tol * mu_face);
    // terrain_face_heat_flux
    init_stable_temperature();
    update_viscosity();
    const auto& alpha = update_alpha();
    const amrex::Real T = 300.0_rt + (m_dTdz * 3.5_rt * m_dz);
    const amrex::Real q = -us * T * us * us / (0.41_rt * 9.81_rt * m_L);
    const amrex::Real a_face = m_rho0 * q * m_dz / (-m_dTdz * m_dz);
    const amrex::Real a_expected =
        (2.0_rt * a_face) - utils::field_probe(alpha, 0, 15, 10, 4);
    EXPECT_NEAR(
        utils::field_probe(alpha, 0, 15, 10, 3), a_expected,
        m_tol * a_expected);
    // blank_follow_fluid
    set_blank_velocity(
        sim().repo().get_field("velocity"),
        sim().repo().get_int_field("terrain_blank"), 0.0_rt);
    const amrex::Real C = blank_drag_rate();
    const amrex::Real C_e = (1.0_rt - std::exp(-C * m_dt)) / m_dt;
    const amrex::Real follow = C_e * m_dTdz * m_dz;
    EXPECT_NEAR(
        utils::field_probe(drag_temp_source("src"), 0, 15, 10, 1), follow,
        m_tol * follow);
}

// With the improved treatment each option can still be turned off: the drag
// cell keeps the model viscosity of the floor length dz / 2 while the length
// above it is measured from the face, and the blanked cells relax toward the
// soil temperature.
TEST_F(KLAxellTerrainTest, improved_wall_treatment_options_can_be_turned_off)
{
    set_string("TerrainDrag", "wall_treatment", "improved");
    set_bool("KLAxell", "terrain_face_stress", false);
    set_bool("DragTempForcing", "blank_follow_fluid", false);
    setup();
    update_viscosity();
    EXPECT_NEAR(mu(15, 10, 3), mu_rans(16.0_rt), m_tol * mu_rans(16.0_rt));
    EXPECT_NEAR(mu(15, 10, 4), mu_rans(48.0_rt), m_tol * mu_rans(48.0_rt));
    set_blank_velocity(
        sim().repo().get_field("velocity"),
        sim().repo().get_int_field("terrain_blank"), 0.0_rt);
    const amrex::Real soil =
        -blank_drag_rate() * (300.0_rt - m_soil_temperature);
    EXPECT_NEAR(
        utils::field_probe(drag_temp_source("src"), 0, 15, 10, 1), soil,
        m_tol * std::abs(soil));
}

// The blanked-cell count matches the blanking test of TerrainDrag also when
// h is at or one ulp below a cell center, where the rounded estimate is one
// off in round-off (for example dz = 0.1, h at or below the cell-8 center).
TEST(KLAxellTerrainCount, blanked_cells_match_the_blanking_test)
{
    const std::vector<amrex::Real> zlos{0.0_rt, -10.0_rt, 3.7_rt};
    const std::vector<amrex::Real> dzs{
        0.1_rt, 7.3_rt, 1.0_rt / 3.0_rt, 400.0_rt / 48.0_rt, 32.0_rt};
    const int nz = 64;
    int n_estimate_off = 0;
    for (const auto zlo : zlos) {
        for (const auto dz : dzs) {
            for (int m = 0; m < nz - 4; ++m) {
                const amrex::Real c = zlo + ((m + 0.5_rt) * dz);
                for (const amrex::Real h :
                     {c, std::nextafter(c, -1.0e30_rt),
                      std::nextafter(c, 1.0e30_rt), zlo + (m * dz)}) {
                    int count = 0;
                    for (int k = 0; k < nz; ++k) {
                        if (zlo + ((k + 0.5_rt) * dz) <= h) {
                            ++count;
                        }
                    }
                    EXPECT_EQ(
                        kynema_sgf::turbulence::terrain_blanked_cells(
                            h, zlo, dz),
                        count)
                        << "zlo " << zlo << " dz " << dz << " h " << h;
                    const int estimate = amrex::max(
                        0, static_cast<int>(
                               std::floor(((h - zlo) / dz) + 0.5_rt)));
                    n_estimate_off += static_cast<int>(estimate != count);
                }
            }
        }
    }
    // The sweep reaches the round-off cases the correction is for
    EXPECT_GT(n_estimate_off, 0);
}

} // namespace kynema_sgf_tests
