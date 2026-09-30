#include "gtest/gtest.h"
#include "ks_test_utils/MeshTest.H"
#include "src/boundary_conditions/BCInterface.H"
#include "src/boundary_conditions/scalar_bcs.H"
#include "src/core/FieldRepo.H"
#include "src/physics/udfs/TabulatedProfile.H"

#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include <fstream>
#include <limits>

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {

//! Write a profile file that a test can point the boundary condition at
void write_profile(const std::string& fname, const std::string& contents)
{
    std::ofstream outfile(fname);
    outfile << contents;
    outfile.close();
}

/** Check that building the profile aborts, and for the expected reason
 *
 *  Several checks can reject the same broken file, so a bare EXPECT_THROW
 *  would pass on whichever fires first. Matching the message pins the test to
 *  the check it is named after.
 *
 *  \param field Field whose boundary profile is built
 *  \param message Part of the abort message the check under test gives
 */
void expect_abort_with(
    const kynema_sgf::Field& field, const std::string& message)
{
    try {
        const kynema_sgf::udf::TabulatedProfile profile(field);
        ADD_FAILURE() << "expected an abort mentioning: " << message;
    } catch (const amrex::RuntimeError& err) {
        const std::string what = err.what();
        EXPECT_NE(what.find(message), std::string::npos)
            << "aborted for another reason: " << what;
    }
}

/** Set the normal velocity through xlo to enter below a given height and
 *  leave above it, ghost cells included, as a veering inflow does
 *
 *  \param vel Velocity field to fill
 *  \param kmid First cell index, counted up from the bottom, that flows out
 */
void set_inflow_outflow_velocity(kynema_sgf::Field& vel, const int kmid)
{
    auto& mfab = vel(0);
    for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
        const auto& gbx = mfi.growntilebox();
        const auto& arr = mfab.array(mfi);
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            arr(i, j, k, 0) = (k < kmid) ? 1.0_rt : -1.0_rt;
            arr(i, j, k, 1) = 0.0_rt;
            arr(i, j, k, 2) = 0.0_rt;
        });
    }
}

/** Largest departure of the xlo ghost cells from the profile where the flow
 *  enters and from the adjacent interior value where it leaves
 *
 *  \param field Scalar field whose ghost cells are checked
 *  \param kmid First cell index, counted up from the bottom, that flows out
 *  \param slope Rate at which the tabulated profile increases with height
 *  \param interior Value held in every interior cell
 */
amrex::Real xlo_ghost_error(
    kynema_sgf::Field& field,
    const int kmid,
    const amrex::Real slope,
    const amrex::Real interior)
{
    const auto& domain = field.repo().mesh().Geom(0).Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);
    auto error = amrex::ReduceMax(
        field(0), 1,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx,
            amrex::Array4<amrex::Real const> const& arr) -> amrex::Real {
            amrex::Real err = 0.0_rt;
            amrex::Loop(bx, [=, &err](int i, int j, int k) {
                if ((i != dlo.x - 1) || (j < dlo.y) || (j > dhi.y) ||
                    (k < dlo.z) || (k > dhi.z)) {
                    return;
                }
                // The grid is 1 m, so the cell center height is k + 0.5
                const auto expected =
                    (k < kmid) ? (slope * (k + 0.5_rt)) : interior;
                err = amrex::max(err, std::abs(arr(i, j, k) - expected));
            });
            return err;
        });
    amrex::ParallelDescriptor::ReduceRealMax(error);
    return error;
}

/** Fill a field using the tabulated profile and return the largest departure
 *  from the expected linear variation with height
 *
 *  The profile operator only needs the cell index to determine the height, so
 *  it can be exercised over the interior of the domain rather than in the
 *  ghost cells alone.
 */
amrex::Real max_error(
    kynema_sgf::Field& field,
    const amrex::Geometry& geom,
    const kynema_sgf::udf::TabulatedProfile& profile,
    const amrex::Orientation ori,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& slope,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& intercept,
    const amrex::Real zground = 0.0_rt)
{
    const int lev = 0;
    const int ncomp = field.num_comp();
    const auto op = profile.device_instance();
    const auto geomdata = geom.data();
    auto& mfab = field(lev);

    for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        const auto& arr = mfab.array(mfi);
        amrex::ParallelFor(
            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                op(amrex::IntVect{i, j, k}, arr, geomdata, 0.0_rt, ori, n, 0,
                   0);
            });
    }

    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    auto error = amrex::ReduceMax(
        mfab, 0,
        [=] AMREX_GPU_HOST_DEVICE(
            amrex::Box const& bx,
            amrex::Array4<amrex::Real const> const& arr) -> amrex::Real {
            amrex::Real err = 0.0_rt;
            amrex::Loop(bx, [=, &err](int i, int j, int k) {
                const auto zco = problo[2] + ((k + 0.5_rt) * dx[2]);
                for (int n = 0; n < ncomp; ++n) {
                    // Below the ground the lowest tabulated value is held
                    const auto zex = amrex::max(zco, zground);
                    const auto expected = intercept[n] + (slope[n] * zex);
                    err = amrex::max(err, std::abs(arr(i, j, k, n) - expected));
                }
            });
            return err;
        });
    amrex::ParallelDescriptor::ReduceRealMax(error);
    return error;
}

//! Write a flat grid file whose ground rises linearly across the domain
void write_terrain(
    const std::string& fname,
    const amrex::Real z_at_ylo,
    const amrex::Real z_at_yhi)
{
    std::ofstream outfile(fname);
    const amrex::Vector<amrex::Real> xs{{0.0_rt, 4.0_rt, 8.0_rt}};
    const amrex::Vector<amrex::Real> ys{{0.0_rt, 4.0_rt, 8.0_rt}};
    outfile << "3\n"
               "3\n";
    for (const auto& x : xs) {
        outfile << x << "\n";
    }
    for (const auto& y : ys) {
        outfile << y << "\n";
    }
    // Indexed [i * ny + j], so x varies slowest
    for (int i = 0; i < 3; ++i) {
        for (const auto& y : ys) {
            outfile << z_at_ylo + ((z_at_yhi - z_at_ylo) * y / 8.0_rt) << "\n";
        }
    }
    outfile.close();
}

//! Write a flat grid file whose ground varies along y only, knot by knot
void write_terrain_along_y(
    const std::string& fname,
    const amrex::Vector<amrex::Real>& ys,
    const amrex::Vector<amrex::Real>& zs)
{
    std::ofstream outfile(fname);
    const amrex::Vector<amrex::Real> xs{{0.0_rt, 8.0_rt}};
    outfile << xs.size() << "\n" << ys.size() << "\n";
    for (const auto& x : xs) {
        outfile << x << "\n";
    }
    for (const auto& y : ys) {
        outfile << y << "\n";
    }
    // Indexed [i * ny + j], so x varies slowest
    for (int i = 0; i < xs.size(); ++i) {
        for (const auto& z : zs) {
            outfile << z << "\n";
        }
    }
    outfile.close();
}

//! Write a flat grid file whose ground rises linearly along x only
void write_terrain_along_x(
    const std::string& fname,
    const amrex::Real z_at_xlo,
    const amrex::Real z_at_xhi)
{
    std::ofstream outfile(fname);
    const amrex::Vector<amrex::Real> xs{{0.0_rt, 8.0_rt}};
    const amrex::Vector<amrex::Real> ys{{0.0_rt, 8.0_rt}};
    outfile << "2\n"
               "2\n";
    for (const auto& x : xs) {
        outfile << x << "\n";
    }
    for (const auto& y : ys) {
        outfile << y << "\n";
    }
    // Indexed [i * ny + j], so x varies slowest
    for (const auto& x : xs) {
        for (int j = 0; j < ys.size(); ++j) {
            outfile << z_at_xlo + ((z_at_xhi - z_at_xlo) * x / 8.0_rt) << "\n";
        }
    }
    outfile.close();
}

} // namespace

class TabulatedProfileTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();
        amrex::ParmParse pp("geometry");
        amrex::Vector<int> periodic{{0, 0, 0}};
        pp.addarr("is_periodic", periodic);
    }

    //! Declare a field and mark the given faces as inflow
    kynema_sgf::Field& inflow_field(
        const std::string& name,
        const int ncomp,
        const amrex::Vector<amrex::Orientation>& inflow_faces)
    {
        auto& frepo = mesh().field_repo();
        auto& fld = frepo.declare_field(name, ncomp, 1, 1);
        fld.setVal(0.0_rt);
        for (const auto& ori : inflow_faces) {
            fld.bc_type()[ori] = BC::mass_inflow;
        }
        return fld;
    }

    // The default mesh is 8 cells over [0, 8], so heights are 0.5 ... 7.5
    const amrex::Real m_tol = 1.0e-12_rt;
    const amrex::Orientation m_xlo{0, amrex::Orientation::low};
    const amrex::Orientation m_ylo{1, amrex::Orientation::low};
};

// A note that starts with z is not the header, before or after the real one.
// The header swaps u and v, so only the real header gives u = 2z, v = 3 - z
TEST_F(TabulatedProfileTest, a_note_before_the_header_is_not_the_header)
{
    populate_parameters();
    write_profile(
        "tp_note1.txt",
        "# z is the height above ground in meters\n"
        "# z v u T\n"
        "0.0  3.0  0.0  300.0\n"
        "8.0 -5.0 16.0  308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_note1.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);
    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {2.0_rt, -1.0_rt, 0.0_rt},
        {0.0_rt, 3.0_rt, 0.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, a_note_after_the_header_is_not_the_header)
{
    populate_parameters();
    write_profile(
        "tp_note2.txt",
        "# z v u T\n"
        "# z values are in meters\n"
        "0.0  3.0  0.0  300.0\n"
        "8.0 -5.0 16.0  308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_note2.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);
    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {2.0_rt, -1.0_rt, 0.0_rt},
        {0.0_rt, 3.0_rt, 0.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, velocity_from_header)
{
    populate_parameters();
    write_profile(
        "tp_header.txt",
        "# z u v T tke\n"
        "0.0  0.0  3.0  300.0  0.0\n"
        "8.0 16.0 -5.0  308.0  0.8\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_header.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);

    // u = 2z, v = 3 - z, and w has no column so it is zero
    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {2.0_rt, -1.0_rt, 0.0_rt},
        {0.0_rt, 3.0_rt, 0.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, temperature_from_headerless_file)
{
    populate_parameters();
    write_profile(
        "tp_plain.txt",
        "0.0  0.0  3.0  300.0\n"
        "8.0 16.0 -5.0  308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_plain.txt"));
    initialize_mesh();

    auto& temp = inflow_field("temperature", 1, {m_xlo});
    const kynema_sgf::udf::TabulatedProfile profile(temp);

    // The fourth column of a headerless file is temperature
    const auto err = max_error(
        temp, mesh().Geom(0), profile, m_xlo, {1.0_rt, 0.0_rt, 0.0_rt},
        {300.0_rt, 0.0_rt, 0.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, tke_column_is_found_by_field_name)
{
    populate_parameters();
    write_profile(
        "tp_tke.txt",
        "0.0  0.0  3.0  300.0  0.0\n"
        "8.0 16.0 -5.0  308.0  0.8\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_tke.txt"));
    initialize_mesh();

    auto& tke = inflow_field("tke", 1, {m_xlo});
    const kynema_sgf::udf::TabulatedProfile profile(tke);

    const auto err = max_error(
        tke, mesh().Geom(0), profile, m_xlo, {0.1_rt, 0.0_rt, 0.0_rt},
        {0.0_rt, 0.0_rt, 0.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, each_face_keeps_its_own_profile)
{
    populate_parameters();
    write_profile(
        "tp_x.txt",
        "# z u v T\n"
        "0.0  0.0  3.0  300.0\n"
        "8.0 16.0 -5.0  308.0\n");
    write_profile(
        "tp_y.txt",
        "# z u v T\n"
        "0.0  1.0  0.0  300.0\n"
        "8.0  1.0  8.0  308.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_x.txt"));
    }
    {
        amrex::ParmParse pp("ylo");
        pp.add("tabulated_profile_file", std::string("tp_y.txt"));
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo, m_ylo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);

    const auto err_x = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {2.0_rt, -1.0_rt, 0.0_rt},
        {0.0_rt, 3.0_rt, 0.0_rt});
    EXPECT_NEAR(err_x, 0.0_rt, m_tol);

    // The same operator returns the other profile on the other face
    const auto err_y = max_error(
        vel, mesh().Geom(0), profile, m_ylo, {0.0_rt, 1.0_rt, 0.0_rt},
        {1.0_rt, 0.0_rt, 0.0_rt});
    EXPECT_NEAR(err_y, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, face_without_a_profile_uses_the_constant)
{
    populate_parameters();
    write_profile(
        "tp_x_only.txt",
        "# z u v T\n"
        "0.0  0.0  3.0  300.0\n"
        "8.0 16.0 -5.0  308.0\n");
    {
        amrex::ParmParse pp("xlo");
        pp.add("tabulated_profile_file", std::string("tp_x_only.txt"));
    }
    {
        amrex::ParmParse pp("ylo");
        amrex::Vector<amrex::Real> uvw{{7.0_rt, 8.0_rt, 9.0_rt}};
        pp.addarr("velocity", uvw);
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo, m_ylo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);

    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_ylo, {0.0_rt, 0.0_rt, 0.0_rt},
        {7.0_rt, 8.0_rt, 9.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, ambiguous_column_count_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_bad.txt",
        "0.0  0.0  3.0\n"
        "8.0 16.0 -5.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_bad.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_THROW(kynema_sgf::udf::TabulatedProfile{vel}, amrex::RuntimeError);
}

// The tke boundary condition needs a tke column wherever tke is profiled
TEST_F(TabulatedProfileTest, tke_without_a_tke_column_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_no_tke.txt",
        "0.0  0.0  3.0  300.0\n"
        "8.0 16.0 -5.0  308.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_no_tke.txt"));
    }
    {
        amrex::ParmParse pp("turbulence");
        pp.add("model", std::string("KLAxell"));
    }
    initialize_mesh();

    auto& tke = inflow_field("tke", 1, {m_xlo});
    expect_abort_with(tke, "tp_no_tke.txt has no tke column");
}

// With KLAxell, velocity may be profiled from a file without tke, for
// example when tke is held at a constant on the inflow face
TEST_F(TabulatedProfileTest, klaxell_velocity_needs_no_tke_column)
{
    populate_parameters();
    write_profile(
        "tp_no_tke2.txt",
        "0.0  0.0  3.0  300.0\n"
        "8.0 16.0 -5.0  308.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_no_tke2.txt"));
    }
    {
        amrex::ParmParse pp("turbulence");
        pp.add("model", std::string("KLAxell"));
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_NO_THROW(kynema_sgf::udf::TabulatedProfile{vel});
}

TEST_F(TabulatedProfileTest, non_monotonic_heights_are_rejected)
{
    populate_parameters();
    write_profile(
        "tp_unsorted.txt",
        "0.0  0.0  3.0  300.0\n"
        "8.0 16.0 -5.0  308.0\n"
        "4.0  8.0 -1.0  304.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_unsorted.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_THROW(kynema_sgf::udf::TabulatedProfile{vel}, amrex::RuntimeError);
}

TEST_F(TabulatedProfileTest, reversing_normal_velocity_needs_inflow_outflow)
{
    populate_parameters();
    // u enters through xlo low down and leaves higher up, as it does under veer
    write_profile(
        "tp_veer.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0  -4.0  0.0  308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_veer.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel,
        "changes sign over the column, so part of the xlo boundary is an "
        "outflow");
}

TEST_F(TabulatedProfileTest, reversing_normal_velocity_is_allowed_on_mixed_face)
{
    populate_parameters();
    write_profile(
        "tp_veer_mio.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0  -4.0  0.0  308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_veer_mio.txt"));
    initialize_mesh();

    auto& frepo = mesh().field_repo();
    auto& vel = frepo.declare_field("velocity", 3, 1, 1);
    vel.setVal(0.0_rt);
    vel.bc_type()[m_xlo] = BC::mass_inflow_outflow;

    const kynema_sgf::udf::TabulatedProfile profile(vel);
    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {-1.0_rt, 0.0_rt, 0.0_rt},
        {4.0_rt, 0.0_rt, 0.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, outflow_part_of_a_scalar_face_is_extrapolated)
{
    populate_parameters();
    // A transported scalar other than temperature or tke, profiled on an
    // inflow-outflow face
    write_profile(
        "tp_tracer.txt",
        "# z u v T tracer\n"
        "0.0  1.0  0.0  300.0  0.0\n"
        "8.0  1.0  0.0  308.0  16.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_tracer.txt"));
    }
    for (const auto& face : {"ylo", "zlo", "xhi", "yhi", "zhi"}) {
        amrex::ParmParse pp(face);
        pp.add("type", std::string("slip_wall"));
    }
    {
        amrex::ParmParse pp("xlo");
        pp.add("type", std::string("mass_inflow_outflow"));
        pp.add("tracer.inflow_outflow_type", std::string("TabulatedProfile"));
    }
    initialize_mesh();

    // Flow enters through the lower half of xlo and leaves through the upper
    const int kmid = 4;
    const amrex::Real interior = 100.0_rt;
    auto& frepo = mesh().field_repo();
    auto& vel = frepo.declare_field("velocity", 3, 1, 1);
    set_inflow_outflow_velocity(vel, kmid);

    auto& tracer = frepo.declare_field("tracer", 1, 1, 1);
    tracer.setVal(interior);
    kynema_sgf::BCScalar bc(tracer);
    bc(0.0_rt);
    kynema_sgf::scalar_bc::register_scalar_dirichlet(
        tracer, mesh(), time(), bc.get_dirichlet_udfs());

    tracer.fillphysbc(0.0_rt);
    tracer.apply_bc_funcs(kynema_sgf::FieldState::New);

    // The profile is held where the flow enters and the interior value is
    // extrapolated where it leaves
    const auto err = xlo_ghost_error(tracer, kmid, 2.0_rt, interior);
    constexpr amrex::Real tol =
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;
    EXPECT_NEAR(err, 0.0_rt, tol);
}

// A scalar set by another UDF keeps that UDF's value on the whole face,
// where the flow leaves too
TEST_F(TabulatedProfileTest, outflow_of_a_scalar_set_by_another_udf_is_kept)
{
    populate_parameters();
    for (const auto& face : {"ylo", "zlo", "xhi", "yhi", "zhi"}) {
        amrex::ParmParse pp(face);
        pp.add("type", std::string("slip_wall"));
    }
    {
        amrex::ParmParse pp("xlo");
        pp.add("type", std::string("mass_inflow_outflow"));
        pp.add("tracer.inflow_outflow_type", std::string("CustomScalar"));
    }
    initialize_mesh();

    // Flow enters through the lower half of xlo and leaves through the upper
    const int kmid = 4;
    const amrex::Real interior = 100.0_rt;
    auto& frepo = mesh().field_repo();
    auto& vel = frepo.declare_field("velocity", 3, 1, 1);
    set_inflow_outflow_velocity(vel, kmid);

    // Ghost cells hold what the UDF would put there, 1, and the interior 100.
    // CustomScalar itself is a template to be filled in by the user, so only
    // the outflow treatment is exercised: it must leave the ghosts alone
    auto& tracer = frepo.declare_field("tracer", 1, 1, 1);
    tracer.setVal(1.0_rt);
    tracer(0).setVal(interior, 0, 1, 0);
    kynema_sgf::BCScalar bc(tracer);
    bc(0.0_rt);
    tracer.apply_bc_funcs(kynema_sgf::FieldState::New);

    const auto err = xlo_ghost_error(tracer, 0, 0.0_rt, 1.0_rt);
    constexpr amrex::Real tol =
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;
    EXPECT_NEAR(err, 0.0_rt, tol);
}

TEST_F(TabulatedProfileTest, outflow_everywhere_on_an_inflow_face_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_backwards.txt",
        "# z u v T\n"
        "0.0  -4.0  0.0  300.0\n"
        "8.0  -4.0  0.0  308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_backwards.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "is directed out of the domain everywhere on xlo");
}

TEST_F(TabulatedProfileTest, zoffset_lifts_the_profile_to_the_ground)
{
    populate_parameters();
    // Tabulated above ground, on a boundary whose ground sits at z = 2
    write_profile(
        "tp_lift.txt",
        "# z u v T\n"
        "0.0   0.0  1.0  300.0\n"
        "8.0  16.0  1.0  308.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_lift.txt"));
        pp.add("zoffset", 2.0_rt);
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);

    // u = 2(z - 2), so the shift moves the whole profile up by the ground
    // height; below the ground the lowest tabulated value is held
    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {2.0_rt, 0.0_rt, 0.0_rt},
        {-4.0_rt, 1.0_rt, 0.0_rt}, 2.0_rt);
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, each_face_can_sit_on_its_own_ground)
{
    populate_parameters();
    write_profile(
        "tp_ground.txt",
        "# z u v T\n"
        "0.0   0.0  1.0  300.0\n"
        "8.0  16.0  1.0  308.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_ground.txt"));
        pp.add("zoffset", 2.0_rt);
    }
    {
        amrex::ParmParse pp("ylo");
        pp.add("tabulated_profile_zoffset", 0.0_rt);
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo, m_ylo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);

    const auto err_x = max_error(
        vel, mesh().Geom(0), profile, m_xlo, {2.0_rt, 0.0_rt, 0.0_rt},
        {-4.0_rt, 1.0_rt, 0.0_rt}, 2.0_rt);
    EXPECT_NEAR(err_x, 0.0_rt, m_tol);

    // The face that overrides the offset back to zero is unshifted
    const auto err_y = max_error(
        vel, mesh().Geom(0), profile, m_ylo, {2.0_rt, 0.0_rt, 0.0_rt},
        {0.0_rt, 1.0_rt, 0.0_rt});
    EXPECT_NEAR(err_y, 0.0_rt, m_tol);
}

TEST_F(TabulatedProfileTest, a_reversal_the_domain_never_reaches_is_allowed)
{
    populate_parameters();
    // u only turns around above z = 40, far above this 8 m tall domain
    write_profile(
        "tp_high_reversal.txt",
        "# z u v T\n"
        "0.0    4.0  0.0  300.0\n"
        "40.0   4.0  0.0  340.0\n"
        "80.0  -4.0  0.0  380.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_high_reversal.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_NO_THROW(kynema_sgf::udf::TabulatedProfile{vel});
}

TEST_F(TabulatedProfileTest, only_the_profile_inside_the_domain_is_checked)
{
    populate_parameters();
    // u only turns around at z = 40, so over this 8 m tall domain it falls
    // from 4 to 3.2 and enters everywhere, although the knot above the domain
    // points out of it
    write_profile(
        "tp_far_knot.txt",
        "# z u v T\n"
        "0.0    4.0  0.0  300.0\n"
        "80.0  -4.0  0.0  380.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_far_knot.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_NO_THROW(kynema_sgf::udf::TabulatedProfile{vel});
}

TEST_F(TabulatedProfileTest, a_reversal_between_two_knots_is_found)
{
    populate_parameters();
    // Neither knot lies strictly inside the domain, but u changes sign at
    // z = 5, so only evaluating the ends of the domain catches it
    write_profile(
        "tp_between_knots.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "10.0 -4.0  0.0  310.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_between_knots.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel,
        "changes sign over the column, so part of the xlo boundary is an "
        "outflow");
}

TEST_F(TabulatedProfileTest, offset_must_match_the_ground_it_stands_on)
{
    populate_parameters();
    write_profile(
        "tp_g1.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0   4.0  0.0  308.0\n");
    write_terrain("tp_terrain_flat.amrwind", 3.0_rt, 3.0_rt);
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_g1.txt"));
    }
    {
        amrex::ParmParse pp("TerrainDrag");
        pp.add("terrain_file", std::string("tp_terrain_flat.amrwind"));
    }
    initialize_mesh();

    // The ground is at 3 but no offset was given
    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_THROW(kynema_sgf::udf::TabulatedProfile{vel}, amrex::RuntimeError);
}

TEST_F(TabulatedProfileTest, offset_matching_the_ground_is_accepted)
{
    populate_parameters();
    write_profile(
        "tp_g2.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0   4.0  0.0  308.0\n");
    write_terrain("tp_terrain_flat2.amrwind", 3.0_rt, 3.0_rt);
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_g2.txt"));
        pp.add("zoffset", 3.0_rt);
    }
    {
        amrex::ParmParse pp("TerrainDrag");
        pp.add("terrain_file", std::string("tp_terrain_flat2.amrwind"));
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_NO_THROW(kynema_sgf::udf::TabulatedProfile{vel});
}

TEST_F(TabulatedProfileTest, ground_varying_along_the_face_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_g3.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0   4.0  0.0  308.0\n");
    // Rising across the span, so the xlo face does not stand on level ground
    write_terrain("tp_terrain_slope.amrwind", 0.0_rt, 8.0_rt);
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_g3.txt"));
        pp.add("zoffset", 4.0_rt);
    }
    {
        amrex::ParmParse pp("TerrainDrag");
        pp.add("terrain_file", std::string("tp_terrain_slope.amrwind"));
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_THROW(kynema_sgf::udf::TabulatedProfile{vel}, amrex::RuntimeError);
}

// The ground checks concern profiles only: a constant ylo face may stand on
// ground that rises along it while the profiled xlo face stands level
TEST_F(TabulatedProfileTest, a_constant_face_may_stand_on_varying_ground)
{
    populate_parameters();
    write_profile(
        "tp_g6.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0   4.0  0.0  308.0\n");
    write_terrain_along_x("tp_terrain_xslope.amrwind", 0.0_rt, 4.0_rt);
    {
        amrex::ParmParse pp("xlo");
        pp.add("tabulated_profile_file", std::string("tp_g6.txt"));
    }
    {
        amrex::ParmParse pp("ylo");
        amrex::Vector<amrex::Real> uvw{{7.0_rt, 8.0_rt, 9.0_rt}};
        pp.addarr("velocity", uvw);
    }
    {
        amrex::ParmParse pp("TerrainDrag");
        pp.add("terrain_file", std::string("tp_terrain_xslope.amrwind"));
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo, m_ylo});
    const kynema_sgf::udf::TabulatedProfile profile(vel);

    const auto err = max_error(
        vel, mesh().Geom(0), profile, m_ylo, {0.0_rt, 0.0_rt, 0.0_rt},
        {7.0_rt, 8.0_rt, 9.0_rt});
    EXPECT_NEAR(err, 0.0_rt, m_tol);
}

// A bump between two cell centers (3.5 and 4.5) is still ground that varies
TEST_F(TabulatedProfileTest, ground_varying_between_cell_centers_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_g5.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0   4.0  0.0  308.0\n");
    write_terrain_along_y(
        "tp_terrain_bump.amrwind", {0.0_rt, 3.5_rt, 4.0_rt, 4.5_rt, 8.0_rt},
        {0.0_rt, 0.0_rt, 8.0_rt, 0.0_rt, 0.0_rt});
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_g5.txt"));
    }
    {
        amrex::ParmParse pp("TerrainDrag");
        pp.add("terrain_file", std::string("tp_terrain_bump.amrwind"));
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "the ground along xlo varies between");
}

TEST_F(TabulatedProfileTest, an_offset_conflicts_with_an_unaligned_interior)
{
    populate_parameters();
    write_profile(
        "tp_g4.txt",
        "# z u v T\n"
        "0.0   4.0  0.0  300.0\n"
        "8.0   4.0  0.0  308.0\n");
    {
        amrex::ParmParse pp("TabulatedProfile");
        pp.add("filename", std::string("tp_g4.txt"));
        pp.add("zoffset", 3.0_rt);
    }
    {
        // The interior profile is measured from the bottom of the domain
        amrex::ParmParse pp("ABL");
        pp.add("initial_wind_profile", true);
        pp.add("terrain_aligned_profile", false);
    }
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    EXPECT_THROW(kynema_sgf::udf::TabulatedProfile{vel}, amrex::RuntimeError);
}

TEST_F(TabulatedProfileTest, a_trailing_word_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e1.txt",
        "# z u v T\n"
        "0.0 1.0 2.0 300.0 junk\n"
        "8.0 3.0 4.0 308.0 junk\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e1.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel, "tp_e1.txt line 2, column 5: 'junk' is not a number");
}

TEST_F(TabulatedProfileTest, a_word_where_a_number_belongs_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e2.txt",
        "# z u v T\n"
        "0.0 abc 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e2.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "tp_e2.txt line 2, column 2: 'abc' is not a number");
}

TEST_F(TabulatedProfileTest, a_nan_in_the_file_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e3.txt",
        "# z u v T\n"
        "0.0 nan 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e3.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel, "tp_e3.txt line 2, column 2: 'nan' is not a finite value");
}

TEST_F(TabulatedProfileTest, an_infinity_in_the_file_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e4.txt",
        "# z u v T\n"
        "0.0 inf 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e4.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel, "tp_e4.txt line 2, column 2: 'inf' is not a finite value");
}

TEST_F(TabulatedProfileTest, an_unrepresentable_number_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e5.txt",
        "# z u v T\n"
        "0.0 1e400 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e5.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel,
        "tp_e5.txt line 2, column 2: '1e400' is too large or too small to "
        "represent");
}

TEST_F(TabulatedProfileTest, rows_of_different_widths_are_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e6.txt",
        "# z u v T\n"
        "0.0 1.0 2.0 300.0\n"
        "8.0 3.0 4.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e6.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(
        vel, "tp_e6.txt line 3 has 3 columns but tp_e6.txt line 2 has 4");
}

TEST_F(TabulatedProfileTest, a_file_with_no_data_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e7.txt",
        "# z u v T\n"
        "\n"
        "# nothing follows\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e7.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "tp_e7.txt holds no profile data");
}

TEST_F(TabulatedProfileTest, a_single_height_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e8.txt",
        "# z u v T\n"
        "0.0 1.0 2.0 300.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e8.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "tp_e8.txt must tabulate at least two heights");
}

TEST_F(TabulatedProfileTest, a_repeated_column_name_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e9.txt",
        "# z u u T\n"
        "0.0 1.0 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e9.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "the header of tp_e9.txt names 'u' more than once");
}

TEST_F(TabulatedProfileTest, a_height_with_no_values_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e10.txt",
        "# z u v T\n"
        "0.0\n"
        "8.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e10.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "tp_e10.txt line 2 holds a height and no values");
}

TEST_F(TabulatedProfileTest, a_repeated_height_name_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e11.txt",
        "# z z u v T\n"
        "0.0 0.0 1.0 2.0 300.0\n"
        "8.0 8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e11.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "the header of tp_e11.txt names 'z' more than once");
}

// 1e40 and 1e-40 are finite doubles but overflow and underflow a float
TEST_F(TabulatedProfileTest, a_number_too_large_for_real_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e12.txt",
        "# z u v T\n"
        "0.0 1e40 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e12.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    if (std::numeric_limits<amrex::Real>::max() < 1.0e40) {
        expect_abort_with(
            vel,
            "tp_e12.txt line 2, column 2: '1e40' is too large or too small to "
            "represent");
    } else {
        EXPECT_NO_THROW(kynema_sgf::udf::TabulatedProfile{vel});
    }
}

TEST_F(TabulatedProfileTest, a_number_too_small_for_real_is_rejected)
{
    populate_parameters();
    write_profile(
        "tp_e13.txt",
        "# z u v T\n"
        "0.0 1e-40 2.0 300.0\n"
        "8.0 3.0 4.0 308.0\n");
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_e13.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    if (std::numeric_limits<amrex::Real>::min() > 1.0e-40) {
        expect_abort_with(
            vel,
            "tp_e13.txt line 2, column 2: '1e-40' is too large or too small "
            "to represent");
    } else {
        EXPECT_NO_THROW(kynema_sgf::udf::TabulatedProfile{vel});
    }
}

// The same file under two spellings of its name is still the RANS file
TEST_F(TabulatedProfileTest, the_rans_profile_file_needs_a_header)
{
    populate_parameters();
    write_profile(
        "tp_rans.txt",
        "0.0 1.0 2.0 0.0 0.5\n"
        "8.0 3.0 4.0 0.0 0.4\n");
    {
        amrex::ParmParse pp("ABL");
        pp.add("rans_1dprofile_file", std::string("./tp_rans.txt"));
    }
    amrex::ParmParse pp("TabulatedProfile");
    pp.add("filename", std::string("tp_rans.txt"));
    initialize_mesh();

    auto& vel = inflow_field("velocity", 3, {m_xlo});
    expect_abort_with(vel, "is also used as ABL.rans_1dprofile_file");
}

} // namespace kynema_sgf_tests
