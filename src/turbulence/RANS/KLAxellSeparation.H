#ifndef KLAXELLSEPARATION_H
#define KLAXELLSEPARATION_H

#include <memory>
#include <string>
#include "src/turbulence/RANS/KLAxell.H"

namespace kynema_sgf::turbulence {

namespace klaxell_separation {
//! Geometry component: height of the cell center above the surface
inline constexpr int geom_height = 0;
//! Geometry component: fluid weight of the cell (0 inside the terrain)
inline constexpr int geom_fluid_weight = 1;
//! Number of geometry components
inline constexpr int geom_ncomp = 2;
} // namespace klaxell_separation

/** KLAxell one-equation RANS model with treatments for separating flow
 *  \ingroup turb_model
 *
 *  The closure is evaluated in a single kernel. The height above the
 *  surface and the fluid weight of each cell are filled beforehand by one
 *  geometry kernel per terrain treatment, so a new terrain treatment adds a
 *  geometry kernel instead of another copy of the closure. With every
 *  optional treatment disabled the model reproduces KLAxell.
 *
 *  Each treatment is switched on from the KLAxellSeparation input section.
 *  The pressure-gradient sensor is a diagnostic stored in the field
 *  pressure_gradient_sensor; the treatments that change the solution act
 *  only where the sensor exceeds its threshold.
 */
template <typename Transport>
class KLAxellSeparation : public KLAxell<Transport>
{
public:
    static std::string identifier()
    {
        return "KLAxellSeparation-" + Transport::identifier();
    }

    explicit KLAxellSeparation(CFDSim& sim);

    ~KLAxellSeparation() override;

    [[nodiscard]] std::string model_name() const override
    {
        return "KLAxellSeparation";
    }

    //! Update the turbulent viscosity field
    void update_turbulent_viscosity(
        FieldState fstate, DiffusionType /*unused*/) override;

    //! Actions to perform post initialization
    void post_init_actions() override;

    //! Actions to perform post regrid
    void post_regrid_actions() override;

    //! Parse turbulence model coefficients
    void parse_model_coeffs() override;

    //! Return turbulence model coefficients
    [[nodiscard]] TurbulenceModel::CoeffsDictType model_coeffs() const override;

    /** Fill the geometry fields for flat terrain
     *
     *  \param lev AMR level
     */
    void flat_geometry(int lev);

    /** Fill the geometry fields from the TerrainDrag fields
     *
     *  \param lev AMR level
     */
    void terrain_drag_geometry(int lev);

    /** Evaluate the closure on one level from the geometry fields
     *
     *  \param lev AMR level
     *  \param fstate Field state
     *  \param beta Thermal expansion coefficient
     */
    void closure(int lev, FieldState fstate, const ScratchField& beta);

    /** Compute the pressure-gradient sensor on one level
     *
     *  The sensor is the pressure gradient along the local flow direction,
     *  (u / |u|) . grad(p), divided by rho (k + c_u |u|^2) / L. Inside the
     *  boundary layer the TKE sets the scale; above it, where the TKE
     *  vanishes, the velocity term keeps the sensor bounded.
     *
     *  \param lev AMR level
     *  \param fstate Field state
     */
    void pressure_gradient_sensor(int lev, FieldState fstate);

    /** Apply the realizable Cmu limiter on one level
     *
     *  The eddy viscosity and the shear and buoyancy production are divided
     *  by 1 + c_s g max(0, Sigma / Cmu - 1), with Sigma = L S / sqrt(k) and
     *  the gate g = clamp((s - s_T) / s_T, 0, 1) of the pressure-gradient
     *  sensor s. The gate is exactly 0 below s_T, so those cells keep the
     *  KLAxell values. Sigma equals Cmu in an equilibrium log layer, and for
     *  large Sigma the eddy viscosity tends to rho Cmu(Rt) Cmu k / (c_s g S).
     *
     *  \param lev AMR level
     */
    void realizable_cmu(int lev);

protected:
    //! Height above the surface and fluid weight of each cell
    std::unique_ptr<ScratchField> m_geometry;

    //! Compute the pressure-gradient sensor
    bool m_use_pressure_gradient_sensor{false};

    //! Weight c_u of the velocity term in the sensor scale
    amrex::Real m_sensor_velocity_weight{0.05_rt};

    //! Sensor value s_T where the treatments start; their weight reaches 1 at
    //! twice this value
    amrex::Real m_sensor_threshold{0.1_rt};

    //! Pressure-gradient sensor, declared only when the sensor is enabled
    Field* m_pressure_gradient_sensor{nullptr};

    //! Apply the realizable Cmu limiter
    bool m_use_realizable_cmu{false};

    //! Strength c_s of the realizable Cmu limiter
    amrex::Real m_realizable_cmu_strength{1.0_rt};

    //! Strain rate magnitude, kept for the realizable Cmu limiter
    std::unique_ptr<ScratchField> m_strain;
};

} // namespace kynema_sgf::turbulence

#endif /* KLAXELLSEPARATION_H */
