#include "ks_test_utils/MeshTest.H"
#include "ks_test_utils/test_utils.H"
#include "src/physics/TerrainDrag.H"
#include "src/turbulence/TurbulenceModel.H"
#include "src/utilities/math_ops.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include <limits>

using namespace amrex::literals;

namespace {
// 100 m plateau for x in [449, 576], flat ground elsewhere
void write_terrain(const std::string& fname)
{
    std::ofstream os(fname);
    os << "6\n2\n";
    os << "0.0\n448.0\n449.0\n576.0\n577.0\n1024.0\n";
    os << "0.0\n1024.0\n";
    os << "0.0\n0.0\n0.0\n0.0\n100.0\n100.0\n100.0\n100.0\n0.0\n0.0\n0.0\n0."
          "0\n";
}

// z u v T tke, read by KransAxell for the mesoscale sponge
void write_rans_profile(const std::string& fname)
{
    std::ofstream os(fname);
    os << "0 8 0 300 0.1\n1000 8 0 300 0.1\n";
}

void set_bool(const std::string& prefix, const char* key, const bool v)
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
            pp.add("viscosity", 1.0e-5_rt);
            pp.add("reference_temperature", 300.0_rt);
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
            pp.add("terrain_file", (std::string) "terrain.amrwind");
            pp.add("uniform_roughness", m_z0);
        }
    }

    void setup()
    {
        write_terrain("terrain.amrwind");
        write_rans_profile("rans_1d.info");
        populate_parameters();
        initialize_mesh();
        auto& pde_mgr = sim().pde_manager();
        pde_mgr.register_icns();
        sim().init_physics();
        sim().create_transport_model();
        m_terrain =
            std::make_unique<kynema_sgf::terraindrag::TerrainDrag>(sim());
        const int nlevels = sim().repo().num_active_levels();
        for (int lev = 0; lev < nlevels; ++lev) {
            m_terrain->initialize_fields(lev, sim().repo().mesh().Geom(lev));
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

    //! u of the shear flow at the center of level k
    [[nodiscard]] amrex::Real u_shear(const int k) const
    {
        return m_shear * (k + 0.5_rt) * m_dz;
    }

    std::unique_ptr<kynema_sgf::terraindrag::TerrainDrag> m_terrain;
    const amrex::Real m_dz{32.0_rt};
    const amrex::Real m_dt{0.5_rt};
    const amrex::Real m_rho0{1.2_rt};
    const amrex::Real m_z0{0.1_rt};
    const amrex::Real m_shear{0.05_rt};
    const amrex::Real m_vspan{2.0_rt};
    const amrex::Real m_tke{0.1_rt};
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

// With every option at its default the model runs the unchanged TerrainDrag
// path: at each cell an option changes, the result is the legacy value.
TEST_F(KLAxellTerrainTest, defaults_leave_the_legacy_path_unchanged)
{
    setup();
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
}

} // namespace kynema_sgf_tests
