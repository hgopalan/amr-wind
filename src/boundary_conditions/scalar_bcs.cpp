#include "src/boundary_conditions/scalar_bcs.H"

namespace kynema_sgf::scalar_bc {
void register_scalar_dirichlet(
    Field& field,
    const amrex::AmrCore& mesh,
    const SimTime& time,
    const amrex::Array<const std::string, 3>& udfs)
{
    const std::string& inflow_udf = udfs[0];
    const std::string& inflow_outflow_udf = udfs[1];
    const std::string& wall_udf = udfs[2];

    if ((inflow_udf == "ConstDirichlet") &&
        (inflow_outflow_udf == "ConstDirichlet") &&
        (wall_udf == "ConstDirichlet")) {
        return;
    }

    if (wall_udf != "ConstDirichlet") {
        amrex::Abort(
            "Scalar BC: Only constant dirichlet supported for Wall BC");
    }

    // One fill operator serves every inflow and inflow-outflow face, so the
    // two kinds of face cannot use different UDFs; when they use the same one
    // it is registered once
    if ((inflow_udf != "ConstDirichlet") &&
        (inflow_outflow_udf != "ConstDirichlet") &&
        (inflow_udf != inflow_outflow_udf)) {
        amrex::Abort(
            "Scalar BC: " + field.name() + ".inflow_type = " + inflow_udf +
            " (mass_inflow faces) and " + field.name() +
            ".inflow_outflow_type = " + inflow_outflow_udf +
            " (mass_inflow_outflow faces) differ; one UDF fills every "
            "inflow face, so use the same one on both");
    }
    if (inflow_udf != "ConstDirichlet") {
        register_inflow_scalar_dirichlet<ConstDirichlet>(
            field, inflow_udf, mesh, time);
    } else if (inflow_outflow_udf != "ConstDirichlet") {
        register_inflow_scalar_dirichlet<ConstDirichlet>(
            field, inflow_outflow_udf, mesh, time);
    }
}
} // namespace kynema_sgf::scalar_bc
