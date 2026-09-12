#include "gtest/gtest.h"
#include "ks_test_utils/MeshTest.H"
#include "src/equation_systems/PDEBase.H"
#include "src/turbulence/TurbulenceModel.H"
#include "AMReX_ParReduce.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {

const amrex::Real tol = std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

//! Largest absolute value over valid cells
amrex::Real max_abs(const amrex::MultiFab& a)
{
    const auto& a_arrs = a.const_arrays();
    return amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{},
        a, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept
            -> amrex::GpuTuple<amrex::Real> {
            return {std::abs(a_arrs[nbx](i, j, k))};
        });
}

//! Largest absolute value of a - b over valid cells
amrex::Real max_diff(const amrex::MultiFab& a, const amrex::MultiFab& b)
{
    const auto& a_arrs = a.const_arrays();
    const auto& b_arrs = b.const_arrays();
    return amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{},
        a, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept
            -> amrex::GpuTuple<amrex::Real> {
            return {std::abs(a_arrs[nbx](i, j, k) - b_arrs[nbx](i, j, k))};
        });
}

//! Largest absolute value of a - b - c over valid cells
amrex::Real max_residual(
    const amrex::MultiFab& a,
    const amrex::MultiFab& b,
    const amrex::MultiFab& c)
{
    const auto& a_arrs = a.const_arrays();
    const auto& b_arrs = b.const_arrays();
    const auto& c_arrs = c.const_arrays();
    return amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{},
        a, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept
            -> amrex::GpuTuple<amrex::Real> {
            return {std::abs(
                a_arrs[nbx](i, j, k) - b_arrs[nbx](i, j, k) -
                c_arrs[nbx](i, j, k))};
        });
}

} // namespace

/** Checks the source terms of the k-omega SST models
 *
 *  The Godunov predictor uses the FieldState::Old source as forcing, so it must
 *  hold the full explicit balance of each equation. The right-hand side of the
 *  update (FieldState::NPH for the predictor, FieldState::New for the MOL
 *  corrector) minus the diagonal term lhs_src_term * field / dt must equal that
 *  balance too, otherwise the converged solution depends on the diffusion type
 *  and the time step.
 */
class TurbSSTSourceTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> probhi{{1.0_rt, 1.0_rt, 1.0_rt}};
            pp.addarr("prob_hi", probhi);
        }
        {
            amrex::ParmParse pp("time");
            pp.add("fixed_dt", m_dt);
        }
        {
            amrex::ParmParse pp("transport");
            pp.add("viscosity", 1.0e-5_rt);
        }
        // No TKE.source_terms or SDR.source_terms: the model injects them
    }

    void setup_model(const std::string& model_name)
    {
        {
            amrex::ParmParse pp("turbulence");
            pp.add("model", model_name);
        }
        {
            amrex::ParmParse pp(model_name + "_coeffs");
            pp.add("tke_amb", m_tke_amb);
            pp.add("sdr_amb", m_sdr_amb);
        }
        populate_parameters();
        initialize_mesh();

        auto& pde_mgr = sim().pde_manager();
        pde_mgr.register_icns();
        sim().init_physics();
        sim().create_turbulence_model();
        sim().turbulence_model().post_init_actions();
        pde_mgr.icns().initialize();
        for (auto& eqn : pde_mgr.scalar_eqns()) {
            eqn->initialize();
        }
    }

    /** Set density, velocity, tke, sdr and wall distance
     *
     *  \param uniform uniform tke and sdr with F1 = 1, otherwise smooth
     *  periodic fields that give cross diffusion of both signs
     */
    void init_fields(const bool uniform)
    {
        auto& repo = sim().repo();
        auto& density = repo.get_field("density");
        density.setVal(m_rho);
        density.state(kynema_sgf::FieldState::Old).setVal(m_rho);
        density.state(kynema_sgf::FieldState::NPH).setVal(m_rho);

        auto& walldist = repo.get_field("wall_dist");
        const auto& geom = sim().mesh().Geom(0);
        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        const amrex::Real twopi = 2.0_rt * std::numbers::pi_v<amrex::Real>;
        const amrex::Real srate = m_srate;

        // Old and New states differ so that a source evaluated on the wrong
        // state is detected
        for (const auto& [fstate, scale] :
             {std::pair{kynema_sgf::FieldState::Old, 1.0_rt},
              std::pair{kynema_sgf::FieldState::New, 1.5_rt}}) {
            auto& tke = repo.get_field("tke").state(fstate)(0);
            auto& sdr = repo.get_field("sdr").state(fstate)(0);
            auto& vel = repo.get_field("velocity").state(fstate)(0);
            const auto& tke_arrs = tke.arrays();
            const auto& sdr_arrs = sdr.arrays();
            const auto& vel_arrs = vel.arrays();
            const auto& wd_arrs = walldist(0).arrays();
            const amrex::Real tke0 = scale * m_tke0;
            const amrex::Real sdr0 = scale * m_sdr0;
            const amrex::Real amp = uniform ? 0.0_rt : 0.5_rt;
            const amrex::Real wd_const = uniform ? 1.0e-5_rt : 0.0_rt;
            amrex::ParallelFor(
                tke, tke.nGrowVect(),
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                    const amrex::Real x = problo[0] + ((i + 0.5_rt) * dx[0]);
                    const amrex::Real y = problo[1] + ((j + 0.5_rt) * dx[1]);
                    const amrex::Real z = problo[2] + ((k + 0.5_rt) * dx[2]);
                    tke_arrs[nbx](i, j, k) =
                        tke0 * (1.0_rt + (amp * std::sin(twopi * x) *
                                          std::cos(twopi * y)));
                    sdr_arrs[nbx](i, j, k) =
                        sdr0 * (1.0_rt + (amp * std::cos(twopi * x) *
                                          std::sin(twopi * z)));
                    vel_arrs[nbx](i, j, k, 0) = x / std::sqrt(6.0_rt) * srate;
                    vel_arrs[nbx](i, j, k, 1) = y / std::sqrt(6.0_rt) * srate;
                    vel_arrs[nbx](i, j, k, 2) = z / std::sqrt(6.0_rt) * srate;
                    wd_arrs[nbx](i, j, k) =
                        (amp > 0.0_rt) ? (0.5_rt + z) : wd_const;
                });
        }
        amrex::Gpu::streamSynchronize();
    }

    kynema_sgf::pde::PDEBase& equation(const std::string& name)
    {
        for (auto& eqn : sim().pde_manager().scalar_eqns()) {
            if (eqn->fields().field.name() == name) {
                return *eqn;
            }
        }
        amrex::Abort("No equation for " + name);
        return *sim().pde_manager().scalar_eqns().front();
    }

    //! Copy of the source term of an equation evaluated for fstate
    std::unique_ptr<kynema_sgf::ScratchField>
    source(const std::string& name, const kynema_sgf::FieldState fstate)
    {
        auto& eqn = equation(name);
        eqn.compute_source_term(fstate);
        auto out = sim().repo().create_scratch_field(1, 0);
        amrex::MultiFab::Copy((*out)(0), eqn.fields().src_term(0), 0, 0, 1, 0);
        return out;
    }

    //! lhs_src_term * field / dt with the field at fstate
    std::unique_ptr<kynema_sgf::ScratchField>
    diagonal_term(const std::string& name, const kynema_sgf::FieldState fstate)
    {
        auto& repo = sim().repo();
        const auto& lhs_arrs =
            repo.get_field(name + "_lhs_src_term")(0).const_arrays();
        const auto& phi_arrs =
            repo.get_field(name).state(fstate)(0).const_arrays();
        auto out = repo.create_scratch_field(1, 0);
        const auto& out_arrs = (*out)(0).arrays();
        const amrex::Real dt = sim().time().delta_t();
        amrex::ParallelFor(
            (*out)(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                out_arrs[nbx](i, j, k) =
                    lhs_arrs[nbx](i, j, k) * phi_arrs[nbx](i, j, k) / dt;
            });
        amrex::Gpu::streamSynchronize();
        return out;
    }

    //! Old-state forcing is the same for every diffusion type
    void check_forcing_independent_of_diffusion()
    {
        auto& tmodel = sim().turbulence_model();
        for (const auto& name : {"tke", "sdr"}) {
            tmodel.update_turbulent_viscosity(
                kynema_sgf::FieldState::Old, DiffusionType::Explicit);
            const auto ref = source(name, kynema_sgf::FieldState::Old);
            const amrex::Real scale = max_abs((*ref)(0));
            ASSERT_GT(scale, 0.0_rt);
            for (const auto dtype :
                 {DiffusionType::Crank_Nicolson, DiffusionType::Implicit}) {
                tmodel.update_turbulent_viscosity(
                    kynema_sgf::FieldState::Old, dtype);
                const auto src = source(name, kynema_sgf::FieldState::Old);
                EXPECT_LE(max_diff((*src)(0), (*ref)(0)), tol * scale)
                    << name << " diffusion type " << static_cast<int>(dtype);
            }
        }
    }

    //! Update source minus diagonal term equals the explicit balance
    void check_steady_state_consistency()
    {
        auto& tmodel = sim().turbulence_model();
        for (const auto fstate :
             {kynema_sgf::FieldState::NPH, kynema_sgf::FieldState::New}) {
            // The model is evaluated on Old for the predictor and on New for
            // the MOL corrector
            const auto model_state = (fstate == kynema_sgf::FieldState::New)
                                         ? kynema_sgf::FieldState::New
                                         : kynema_sgf::FieldState::Old;
            for (const auto dtype :
                 {DiffusionType::Explicit, DiffusionType::Crank_Nicolson,
                  DiffusionType::Implicit}) {
                tmodel.update_turbulent_viscosity(model_state, dtype);
                for (const auto& name : {"tke", "sdr"}) {
                    const auto balance =
                        source(name, kynema_sgf::FieldState::Old);
                    const auto rhs = source(name, fstate);
                    const auto diag = diagonal_term(name, model_state);
                    const amrex::Real diag_max = max_abs((*diag)(0));
                    if (dtype == DiffusionType::Explicit) {
                        EXPECT_EQ(diag_max, 0.0_rt) << name;
                    } else {
                        EXPECT_GT(diag_max, 0.0_rt) << name;
                    }
                    const amrex::Real scale =
                        std::max(max_abs((*balance)(0)), diag_max);
                    EXPECT_LE(
                        max_residual((*rhs)(0), (*diag)(0), (*balance)(0)),
                        tol * scale)
                        << name << " state " << static_cast<int>(fstate)
                        << " diffusion type " << static_cast<int>(dtype);
                }
            }
        }
    }

    const amrex::Real m_dt{0.05_rt};
    const amrex::Real m_tke0{0.5_rt};
    const amrex::Real m_sdr0{10.0_rt};
    const amrex::Real m_srate{2.0_rt};
    const amrex::Real m_tke_amb{0.1_rt};
    const amrex::Real m_sdr_amb{2.0_rt};
    amrex::Real m_rho{1.0_rt};
};

TEST_F(TurbSSTSourceTest, sst_explicit_balance_values)
{
    setup_model("KOmegaSST");

    auto& tmodel = sim().turbulence_model();
    auto coeffs = tmodel.model_coeffs();
    const amrex::Real beta_star = coeffs["beta_star"];
    const amrex::Real alpha1 = coeffs["alpha1"];
    const amrex::Real beta1 = coeffs["beta1"];

    // The equation source (after multiplication by density) must be density
    // weighted once, for unit and non-unit density
    for (const auto dtype :
         {DiffusionType::Explicit, DiffusionType::Crank_Nicolson,
          DiffusionType::Implicit}) {
        for (const amrex::Real rho : {1.0_rt, 1.2_rt}) {
            m_rho = rho;
            init_fields(true);

            // Uniform tke and sdr with F1 = 1 (no cross diffusion) and a strain
            // rate small enough that neither the viscosity limiter nor the
            // production limiters are active
            const amrex::Real prod_k =
                rho * (m_tke0 / m_sdr0) * m_srate * m_srate;
            const amrex::Real expected_tke =
                prod_k - (beta_star * rho * m_tke0 * m_sdr0) +
                (beta_star * rho * m_sdr_amb * m_tke_amb);
            const amrex::Real expected_sdr =
                (rho * alpha1 * m_srate * m_srate) -
                (beta1 * rho * m_sdr0 * m_sdr0) +
                (beta1 * rho * m_sdr_amb * m_sdr_amb);

            tmodel.update_turbulent_viscosity(
                kynema_sgf::FieldState::Old, dtype);
            for (const auto& [name, expected] :
                 {std::pair{"tke", expected_tke},
                  std::pair{"sdr", expected_sdr}}) {
                const auto forcing = source(name, kynema_sgf::FieldState::Old);
                const auto rhs = source(name, kynema_sgf::FieldState::NPH);
                const auto diag =
                    diagonal_term(name, kynema_sgf::FieldState::Old);
                const auto& src_arrs = (*forcing)(0).const_arrays();
                const auto& rhs_arrs = (*rhs)(0).const_arrays();
                const auto& diag_arrs = (*diag)(0).const_arrays();
                const auto err = amrex::ParReduce(
                    amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpMax>{},
                    amrex::TypeList<amrex::Real, amrex::Real>{}, (*forcing)(0),
                    amrex::IntVect(0),
                    [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept
                        -> amrex::GpuTuple<amrex::Real, amrex::Real> {
                        return {
                            std::abs(src_arrs[nbx](i, j, k) - expected),
                            std::abs(
                                rhs_arrs[nbx](i, j, k) -
                                diag_arrs[nbx](i, j, k) - expected)};
                    });
                const amrex::Real scale = std::abs(expected);
                EXPECT_LE(amrex::get<0>(err), tol * scale)
                    << name << " forcing, diffusion type "
                    << static_cast<int>(dtype);
                EXPECT_LE(amrex::get<1>(err), tol * scale)
                    << name << " right-hand side, diffusion type "
                    << static_cast<int>(dtype);
            }
        }
    }
}

TEST_F(TurbSSTSourceTest, sst_forcing_independent_of_diffusion)
{
    m_rho = 1.2_rt;
    setup_model("KOmegaSST");
    init_fields(false);
    check_forcing_independent_of_diffusion();
}

TEST_F(TurbSSTSourceTest, sst_steady_state_consistency)
{
    m_rho = 1.2_rt;
    setup_model("KOmegaSST");
    init_fields(false);
    check_steady_state_consistency();
}

TEST_F(TurbSSTSourceTest, sstiddes_forcing_independent_of_diffusion)
{
    m_rho = 1.2_rt;
    setup_model("KOmegaSSTIDDES");
    init_fields(false);
    check_forcing_independent_of_diffusion();
}

TEST_F(TurbSSTSourceTest, sstiddes_steady_state_consistency)
{
    m_rho = 1.2_rt;
    setup_model("KOmegaSSTIDDES");
    init_fields(false);
    check_steady_state_consistency();
}

} // namespace kynema_sgf_tests
