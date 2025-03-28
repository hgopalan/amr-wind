#include "gtest/gtest.h"
#include "aw_test_utils/MeshTest.H"
#include "amr-wind/turbulence/TurbulenceModel.H"
#include "aw_test_utils/test_utils.H"

namespace amr_wind_tests {

namespace {

void init_strain_field(amr_wind::Field& fld, amrex::Real srate)
{
    const auto& mesh = fld.repo().mesh();
    const int nlevels = fld.repo().num_active_levels();
    amrex::Real offset =
        (fld.field_location() == amr_wind::FieldLoc::CELL) ? 0.5 : 0.0;
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& dx = mesh.Geom(lev).CellSizeArray();
        const auto& problo = mesh.Geom(lev).ProbLoArray();
        const auto& farrs = fld(lev).arrays();

        amrex::ParallelFor(
            fld(lev), fld.num_grow(),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                const amrex::Real x = problo[0] + (i + offset) * dx[0];
                const amrex::Real y = problo[1] + (j + offset) * dx[1];
                const amrex::Real z = problo[2] + (k + offset) * dx[2];

                farrs[nbx](i, j, k, 0) = x / sqrt(6.0) * srate;
                farrs[nbx](i, j, k, 1) = y / sqrt(6.0) * srate;
                farrs[nbx](i, j, k, 2) = z / sqrt(6.0) * srate;
            });
    }
    amrex::Gpu::streamSynchronize();
}

void init_temperature_field(amr_wind::Field& fld, amrex::Real tgrad)
{
    const auto& mesh = fld.repo().mesh();
    const int nlevels = fld.repo().num_active_levels();

    amrex::Real offset =
        (fld.field_location() == amr_wind::FieldLoc::CELL) ? 0.5 : 0.0;

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& dx = mesh.Geom(lev).CellSizeArray();
        const auto& problo = mesh.Geom(lev).ProbLoArray();
        const auto& farrs = fld(lev).arrays();

        amrex::ParallelFor(
            fld(lev), fld.num_grow(),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                const amrex::Real z = problo[2] + (k + offset) * dx[2];

                farrs[nbx](i, j, k, 0) = z * tgrad;
            });
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace

class TurbRANSTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{10, 10, 64}};
            pp.addarr("n_cell", ncell);
            pp.add("blocking_factor", 2);
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0, 0.0, 0.0}};
            amrex::Vector<amrex::Real> probhi{{1024.0, 1024.0, 1024.0}};
            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
        }
    }
};

TEST_F(TurbRANSTest, test_1eqKrans_setup_calc)
{
    // Parser inputs for turbulence model
    const amrex::Real Tref = 265;
    const amrex::Real gravz = 10.0;
    const amrex::Real rho0 = 1.2;
    {
        amrex::ParmParse pp("turbulence");
        pp.add("model", (std::string) "KLAxell");
    }
    {
        amrex::ParmParse pp("incflo");
        amrex::Vector<std::string> physics{"ABL"};
        pp.addarr("physics", physics);
        pp.add("density", rho0);
        amrex::Vector<amrex::Real> vvec{8.0, 0.0, 0.0};
        pp.addarr("velocity", vvec);
        amrex::Vector<amrex::Real> gvec{0.0, 0.0, -gravz};
        pp.addarr("gravity", gvec);
    }
    {
        amrex::ParmParse pp("ABL");
        pp.add("surface_temp_rate", 0.0);
        pp.add("initial_wind_profile", true);
        amrex::Vector<amrex::Real> t_hts{0.0, 100.0, 4000.0};
        pp.addarr("temperature_heights", t_hts);
        pp.addarr("wind_heights", t_hts);
        amrex::Vector<amrex::Real> t_vals{265.0, 265.0, 265.0};
        pp.addarr("temperature_values", t_vals);
        amrex::Vector<amrex::Real> u_vals{8.0, 8.0, 8.0};
        pp.addarr("u_values", u_vals);
        amrex::Vector<amrex::Real> v_vals{0.0, 0.0, 0.0};
        pp.addarr("v_values", v_vals);
        amrex::Vector<amrex::Real> tke_vals{0.1, 0.1, 0.1};
        pp.addarr("tke_values", tke_vals);
        pp.add("surface_temp_flux", 0.0);
    }
    // Transport
    {
        amrex::ParmParse pp("transport");
        pp.add("reference_temperature", Tref);
    }

    // Initialize necessary parts of solver
    populate_parameters();
    initialize_mesh();
    auto& pde_mgr = sim().pde_manager();
    pde_mgr.register_icns();
    sim().init_physics();

    // Create turbulence model
    sim().create_turbulence_model();
    // Get turbulence model
    auto& tmodel = sim().turbulence_model();

    // Get coefficients
    auto model_dict = tmodel.model_coeffs();

    // Constants for fields
    const amrex::Real srate = 20.0;
    const amrex::Real Tgz = 0.0;
    const amrex::Real lambda = 30.0;
    const amrex::Real kappa = 0.41;
    const amrex::Real x3 = 1016;
    const amrex::Real lscale_s = (lambda * kappa * x3) / (lambda + kappa * x3);
    const amrex::Real tlscale_val = lscale_s;
    const amrex::Real tke_val = 0.1;
    // Set up velocity field with constant strainrate
    auto& vel = sim().repo().get_field("velocity");
    init_strain_field(vel, srate);
    // Set up uniform unity density field
    auto& dens = sim().repo().get_field("density");
    dens.setVal(rho0);
    // Set up temperature field with constant gradient in z
    auto& temp = sim().repo().get_field("temperature");
    init_temperature_field(temp, Tgz);
    // Give values to tlscale and tke arrays
    auto& tlscale = sim().repo().get_field("turb_lscale");
    tlscale.setVal(tlscale_val);
    auto& tke = sim().repo().get_field("tke");
    tke.setVal(tke_val);

    // Update turbulent viscosity directly
    tmodel.update_turbulent_viscosity(
        amr_wind::FieldState::New, DiffusionType::Crank_Nicolson);
    const auto& muturb = sim().repo().get_field("mu_turb");

    // Check values of turbulent viscosity
    const auto max_val = utils::field_max(muturb);
    const amrex::Real Cmu = 0.556;
    const amrex::Real epsilon =
        std::pow(Cmu, 3) * std::pow(tke_val, 1.5) / (tlscale_val + 1e-3);
    const amrex::Real stratification = 0.0;
    const amrex::Real Rt = std::pow(tke_val / epsilon, 2) * stratification;
    const amrex::Real Cmu_Rt =
        (0.556 + 0.108 * Rt) / (1 + 0.308 * Rt + 0.00837 * std::pow(Rt, 2));
    const amrex::Real tol = 0.12;
    const amrex::Real nut_max = rho0 * Cmu_Rt * tlscale_val * sqrt(tke_val);
    EXPECT_NEAR(max_val, nut_max, tol);
}

} // namespace amr_wind_tests
