#include <fstream>
#include "ks_test_utils/AmrexTest.H"
#include "src/utilities/io_utils.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

// Five columns per line, z u v w tke; comments, blank lines and CRLF line
// ends are skipped. The earlier reader took the stream five numbers at a
// time, so it read the same values only when every line held five numbers.
TEST(Rans1DProfile, reads_five_columns)
{
    const std::string fname = "rans_1d_reader_test.info";
    {
        std::ofstream os(fname);
        os << "# z u v w tke\n";
        os << "0 3.5 0.5 0 0.9\r\n";
        os << "\n";
        os << "\r\n";
        os << "  500 15 -1 0.25 1e-15  \n";
        os << "500 15 -1 0.25 2e-15\n";
        os << "1000 15 0 0 1.5e-08\n";
    }
    const auto prof = kynema_sgf::ioutils::read_rans_1d_profile(fname);
    ASSERT_EQ(prof.z.size(), 4u);
    const amrex::Vector<amrex::Real> z{0.0_rt, 500.0_rt, 500.0_rt, 1000.0_rt};
    const amrex::Vector<amrex::Real> u{3.5_rt, 15.0_rt, 15.0_rt, 15.0_rt};
    const amrex::Vector<amrex::Real> v{0.5_rt, -1.0_rt, -1.0_rt, 0.0_rt};
    const amrex::Vector<amrex::Real> w{0.0_rt, 0.25_rt, 0.25_rt, 0.0_rt};
    const amrex::Vector<amrex::Real> tke{0.9_rt, 1e-15_rt, 2e-15_rt, 1.5e-8_rt};
    for (int n = 0; n < 4; ++n) {
        EXPECT_EQ(prof.z[n], z[n]);
        EXPECT_EQ(prof.u[n], u[n]);
        EXPECT_EQ(prof.v[n], v[n]);
        EXPECT_EQ(prof.w[n], w[n]);
        EXPECT_EQ(prof.tke[n], tke[n]);
    }
    std::remove(fname.c_str());
}

} // namespace kynema_sgf_tests
