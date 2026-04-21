// DragTempForcing.H

#ifndef DRAGTEMPFORCING_H
#define DRAGTEMPFORCING_H

#include <ParmParse.H>
#include <some_other_dependencies>

class DragTempForcing {
public:
    DragTempForcing() : m_surface_heat_flux(0.0), m_rho_cp(1.225 * 1004.0), m_use_soil_temperature(true) {
        ParmParse p;
        p.query("surface_heat_flux", m_surface_heat_flux);
        p.query("rho_cp", m_rho_cp);
        p.query("use_soil_temperature", m_use_soil_temperature);
    }

private:
    Real m_surface_heat_flux;
    Real m_rho_cp;
    bool m_use_soil_temperature;
};

#endif // DRAGTEMPFORCING_H
