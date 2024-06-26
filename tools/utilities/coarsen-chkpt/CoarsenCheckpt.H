#ifndef COARSENCHECKPT_H
#define COARSENCHECKPT_H

#include "amr-wind/incflo.H"

namespace amr_wind {
namespace tools {

// Adapted from RefineCheckpt tool
// additional code from amr-wind/utilities/io.cpp and
// amr-wind/utilities/IOManager.cpp

class CoarsenCheckpt : public incflo
{
public:
    CoarsenCheckpt();

    virtual ~CoarsenCheckpt() = default;

    void read_chkpt_add_baselevel();

    void coarsen_chkpt_file();

    void run_utility();

    void read_checkpoint_fields_offset(
        const std::string& restart_file,
        const amrex::Vector<amrex::BoxArray>& ba_chk,
        const amrex::Vector<amrex::DistributionMapping>& dm_chk,
        const amrex::IntVect& rep);

    void average_down_all_fields();

private:
    amrex::Vector<amrex::BoxArray> m_orig_ba;
};

} // namespace tools
} // namespace amr_wind

#endif /* COARSENCHECKPT_H */
