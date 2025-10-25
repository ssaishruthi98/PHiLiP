#ifndef __PARAMETERS_EULER_H__
#define __PARAMETERS_EULER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the linear solver
class EulerParam
{
public:
    double ref_length; ///< Reference length.
    double mach_inf; ///< Mach number at infinity.
    double gamma_gas; ///< Adiabatic index of the fluid.
    /// Input file provides in degrees, but the value stored here is in radians
    double angle_of_attack;
    /// Input file provides in degrees, but the value stored here is in radians
    double side_slip_angle;

    /// Custom boundary values
    /** These boundary conditions can only be used in Euler so max length is max nstate = dim + 2 = 5 **/
    dealii::Tensor<1, 5, double> custom_boundary_for_each_state;

    /// Flag for using custom heat capacities for species
    bool use_custom_heat_capacities;

    /// Custom Cp and Cv values
    /** These values are only used in multispecies calorically perfect gas and varies depending on number of species in case **/
    std::vector<double> custom_Cp;
    std::vector<double> custom_Cv;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
