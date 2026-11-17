#include <fstream>

#include "src/utilities/io_utils.H"

namespace kynema_sgf::ioutils {

void goto_next_line(std::istream& is)
{
    constexpr std::streamsize bl_ignore_max{100000};
    is.ignore(bl_ignore_max, '\n');
}

void read_flat_grid_file(
    const std::string& fname,
    amrex::Vector<amrex::Real>& xs,
    amrex::Vector<amrex::Real>& ys,
    amrex::Vector<amrex::Real>& zs)
{
    std::ifstream file(fname, std::ios::in);

    if (!file.good()) {
        amrex::Abort("Cannot find file");
    }

    size_t nx = 0;
    size_t ny = 0;
    if (!(file >> nx)) {
        amrex::Abort("Failed to read grid dimension nx");
    }
    if (!(file >> ny)) {
        amrex::Abort("Failed to read grid dimension ny");
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nx > 0, "nx must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ny > 0, "ny must be > 0");

    xs.resize(nx);
    ys.resize(ny);
    zs.resize(nx * ny);

    for (size_t n = 0; n < nx; n++) {
        if (!(file >> xs[n])) {
            amrex::Abort("Failed to read xs[" + std::to_string(n) + "]");
        }
    }
    for (size_t n = 0; n < ny; n++) {
        if (!(file >> ys[n])) {
            amrex::Abort("Failed to read ys[" + std::to_string(n) + "]");
        }
    }
    for (size_t n = 0; n < nx * ny; n++) {
        if (!(file >> zs[n])) {
            amrex::Abort("Failed to read zs[" + std::to_string(n) + "]");
        }
    }

    file.close();
}
Rans1DProfile read_rans_1d_profile(const std::string& fname)
{
    std::ifstream ifh(fname, std::ios::in);
    if (!ifh.good()) {
        amrex::Abort("Cannot find 1-D RANS profile file " + fname);
    }
    Rans1DProfile prof;
    std::string line;
    int lineno = 0;
    while (std::getline(ifh, line)) {
        ++lineno;
        const auto first = line.find_first_not_of(" \t\r");
        if ((first == std::string::npos) || (line[first] == '#')) {
            continue;
        }
        std::istringstream iss(line);
        amrex::Vector<amrex::Real> vals;
        amrex::Real val;
        while (iss >> val) {
            vals.push_back(val);
        }
        const bool bad_token = !iss.eof();
        if (bad_token || (vals.size() != 5)) {
            amrex::Abort(
                "ABL.rans_1dprofile_file: line " + std::to_string(lineno) +
                " of " + fname +
                " must hold exactly five numbers, z u v w tke; found " +
                (bad_token ? std::string("a non-numeric entry")
                           : std::to_string(vals.size()) + " numbers"));
        }
        if (!prof.z.empty() && (vals[0] < prof.z.back())) {
            amrex::Abort(
                "ABL.rans_1dprofile_file: heights in " + fname +
                " must not decrease (line " + std::to_string(lineno) + ")");
        }
        prof.z.push_back(vals[0]);
        prof.u.push_back(vals[1]);
        prof.v.push_back(vals[2]);
        prof.w.push_back(vals[3]);
        prof.tke.push_back(vals[4]);
    }
    if (prof.z.empty()) {
        amrex::Abort("ABL.rans_1dprofile_file: no rows in " + fname);
    }
    return prof;
}

} // namespace kynema_sgf::ioutils
