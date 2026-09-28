#include <numbers>

#include "ks_test_utils/MeshTest.H"

#include "src/wind_energy/actuator/FLLC.H"
#include "src/wind_energy/actuator/turbine/ActSrcDiskOp_Turbine.H"
#include "src/wind_energy/actuator/turbine/turbine_types.H"
#include "src/core/Slice.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {
namespace {

namespace act = kynema_sgf::actuator;
namespace vs = kynema_sgf::vs;
namespace utils = kynema_sgf::utils;

struct DiskTurbine : public act::TurbineType
{
    using InfoType = act::TurbineInfo;
    using GridType = act::ActGrid;
    using MetaType = act::TurbineBaseData;
    using DataType = act::ActDataHolder<DiskTurbine>;

    static std::string identifier() { return "TestDiskTurbine"; }
};

constexpr int num_blades = 3;
constexpr int num_pts_blade = 8;
constexpr int num_pts_tower = 4;
// Radial spacing of the blade points (larger than 1 m so that dR and dR^2
// differ)
constexpr amrex::Real blade_dr = 3.0_rt;
const vs::Vector rotor_center{32.0_rt, 32.0_rt, 32.0_rt};

// Populate the actuator grid with the same layout as the external turbine
// models (hub, blades, tower) and set up the component views into it.
void init_disk_turbine(DiskTurbine::DataType& data)
{
    auto& info = data.info();
    auto& grid = data.grid();
    auto& meta = data.meta();

    info.bound_box =
        amrex::RealBox({0.0_rt, 0.0_rt, 0.0_rt}, {64.0_rt, 64.0_rt, 64.0_rt});

    meta.num_blades = num_blades;
    meta.num_pts_blade = num_pts_blade;
    meta.num_vel_pts_blade = num_pts_blade;
    meta.num_pts_tower = num_pts_tower;
    meta.rot_center = rotor_center;
    meta.rotor_frame = vs::Tensor::identity();

    const int npts = 1 + (num_blades * num_pts_blade) + num_pts_tower;
    grid.resize(npts);
    for (int ip = 0; ip < npts; ++ip) {
        grid.epsilon[ip] = vs::Vector(3.0_rt, 3.0_rt, 3.0_rt);
        grid.orientation[ip] = vs::Tensor::identity();
        grid.force[ip] = vs::Vector::zero();
    }

    grid.pos[0] = rotor_center;
    meta.hub.pos = utils::slice(grid.pos, 0, 1);
    meta.hub.force = utils::slice(grid.force, 0, 1);
    meta.hub.epsilon = utils::slice(grid.epsilon, 0, 1);
    meta.hub.orientation = utils::slice(grid.orientation, 0, 1);

    for (int ib = 0; ib < num_blades; ++ib) {
        const amrex::Real phi =
            2.0_rt * std::numbers::pi_v<amrex::Real> * ib / num_blades;
        const int start = 1 + (ib * num_pts_blade);
        for (int ip = 0; ip < num_pts_blade; ++ip) {
            const amrex::Real r = blade_dr * (ip + 1);
            grid.pos[start + ip] =
                rotor_center +
                vs::Vector(0.0_rt, r * std::cos(phi), r * std::sin(phi));
        }

        act::ComponentView cv;
        cv.pos = utils::slice(grid.pos, start, num_pts_blade);
        cv.force = utils::slice(grid.force, start, num_pts_blade);
        cv.epsilon = utils::slice(grid.epsilon, start, num_pts_blade);
        cv.orientation = utils::slice(grid.orientation, start, num_pts_blade);
        meta.blades.emplace_back(cv);
    }

    const int tstart = 1 + (num_blades * num_pts_blade);
    for (int ip = 0; ip < num_pts_tower; ++ip) {
        grid.pos[tstart + ip] =
            vs::Vector(36.0_rt, 32.0_rt, 16.0_rt + (3.0_rt * ip));
    }
    meta.tower.pos = utils::slice(grid.pos, tstart, num_pts_tower);
    meta.tower.force = utils::slice(grid.force, tstart, num_pts_tower);
    meta.tower.epsilon = utils::slice(grid.epsilon, tstart, num_pts_tower);
    meta.tower.orientation =
        utils::slice(grid.orientation, tstart, num_pts_tower);
}

class TurbineDiskSrcTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{32, 32, 32}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", 16);
            pp.addarr("n_cell", ncell);
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0_rt, 0.0_rt, 0.0_rt}};
            amrex::Vector<amrex::Real> probhi{{64.0_rt, 64.0_rt, 64.0_rt}};

            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
        }
    }
};

void compute_source(DiskTurbine::DataType& data, kynema_sgf::Field& src)
{
    act::ops::ActSrcOp<DiskTurbine, act::ActSrcDisk> op(data);
    op.initialize();
    op.setup_op();

    src.setVal(0.0_rt);
    const auto& geom = data.sim().mesh().Geom(0);
    for (amrex::MFIter mfi(src(0)); mfi.isValid(); ++mfi) {
        op(0, mfi, geom);
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace

TEST_F(TurbineDiskSrcTest, conserves_total_force)
{
    initialize_mesh();
    auto& src = sim().repo().declare_field("actuator_src_term", 3, 0);

    DiskTurbine::DataType data(sim(), "disk", 0);
    init_disk_turbine(data);

    auto& grid = data.grid();
    vs::Vector total_force = vs::Vector::zero();
    for (int ip = 0; ip < grid.force.size(); ++ip) {
        grid.force[ip] =
            vs::Vector(1.0_rt + (0.1_rt * ip), 0.2_rt, -0.1_rt * (ip % 3));
        total_force = total_force + grid.force[ip];
    }

    compute_source(data, src);

    const auto& dx = sim().mesh().Geom(0).CellSizeArray();
    const amrex::Real vol = dx[0] * dx[1] * dx[2];
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        const amrex::Real integral = src(0).sum(n) * vol;
        EXPECT_NEAR(integral, total_force[n], 5.0e-3_rt * vs::mag(total_force))
            << "component " << n;
    }
}

TEST_F(TurbineDiskSrcTest, blade_point_radial_support)
{
    initialize_mesh();
    auto& src = sim().repo().declare_field("actuator_src_term", 3, 0);

    DiskTurbine::DataType data(sim(), "disk", 0);
    init_disk_turbine(data);

    // Load a single blade point
    const int ipt = 1 + 3;
    const amrex::Real rpt = blade_dr * 4;
    data.grid().force[ipt] = vs::Vector(1.0_rt, 0.0_rt, 0.0_rt);

    compute_source(data, src);

    amrex::MultiFab hsrc(
        src(0).boxArray(), src(0).DistributionMap(), 3, 0,
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(hsrc, src(0));

    const auto& geom = sim().mesh().Geom(0);
    const auto& problo = geom.ProbLoArray();
    const auto& dx = geom.CellSizeArray();
    int ninside = 0;
    int noutside = 0;
    for (amrex::MFIter mfi(hsrc); mfi.isValid(); ++mfi) {
        const auto& sarr = hsrc.const_array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
            const amrex::Real y = problo[1] + ((j + 0.5_rt) * dx[1]);
            const amrex::Real z = problo[2] + ((k + 0.5_rt) * dx[2]);
            const amrex::Real r =
                std::hypot(y - rotor_center.y(), z - rotor_center.z());
            if (std::abs(r - rpt) > blade_dr + 1.0e-6_rt) {
                if (sarr(i, j, k, 0) != 0.0_rt) {
                    ++noutside;
                }
            } else if (sarr(i, j, k, 0) > 0.0_rt) {
                ++ninside;
            }
        });
    }
    // The force spreads to cells within one point spacing of the point only
    EXPECT_GT(ninside, 0);
    EXPECT_EQ(noutside, 0);
}

} // namespace kynema_sgf_tests
