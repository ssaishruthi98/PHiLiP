#include <deal.II/base/function.h>
#include "exact_solution.h"

namespace PHiLiP {

// ========================================================
// ZERO -- Returns zero everywhere; used a placeholder when no exact solution is defined.
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_Zero<dim,nstate,real>
::ExactSolutionFunction_Zero(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_Zero<dim,nstate,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 0;
    return value;
}

// ========================================================
// 1D SINE -- Exact solution for advection_explicit_time_study
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_1DSine<dim,nstate,real>
::ExactSolutionFunction_1DSine (double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_1DSine<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    double x_adv_speed = 1.0;

    real value = 0;
    real pi = dealii::numbers::PI;
    if(point[0] >= 0.0 && point[0] <= 2.0){
        value = sin(2*pi*(point[0] - x_adv_speed * t)/2.0);
    }
    return value;
}

// ========================================================
// Inviscid Isentropic Vortex 
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::ExactSolutionFunction_IsentropicVortex(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Setting constants
    const double L = 10.0; // half-width of domain
    const double pi = dealii::numbers::PI;
    const double gam = 1.4;
    const double M_infty = sqrt(2/gam);
    const double R = 1;
    const double sigma = 1;
    const double beta = M_infty * 5 * sqrt(2.0)/4.0/pi * exp(1.0/2.0);
    const double alpha = pi/4; //rad

    // Centre of the vortex  at t
    const double x_travel = M_infty * t * cos(alpha);
    const double x0 = 0.0 + x_travel;
    const double y_travel = M_infty * t * sin(alpha);
    const double y0 = 0.0 + y_travel;
    const double x = std::fmod(point[0] - x0-L, 2*L)+L;
    const double y = std::fmod(point[1] - y0-L, 2*L)+L;

    const double Omega = beta * exp(-0.5/sigma/sigma* (x/R * x/R + y/R * y/R));
    const double delta_Ux = -y/R * Omega;
    const double delta_Uy =  x/R * Omega;
    const double delta_T  = -(gam-1.0)/2.0 * Omega * Omega;

    // Primitive
    const double rho = pow((1 + delta_T), 1.0/(gam-1.0));
    const double Ux = M_infty * cos(alpha) + delta_Ux;
    const double Uy = M_infty * sin(alpha) + delta_Uy;
    const double Uz = 0;
    const double p = 1.0/gam*pow(1+delta_T, gam/(gam-1.0));

    //Convert to conservative variables
    if (istate == 0)      return rho;       //density 
    else if (istate == nstate-1) return p/(gam-1.0) + 0.5 * rho * (Ux*Ux + Uy*Uy + Uz*Uz);   //total energy
    else if (istate == 1) return rho * Ux;  //x-momentum
    else if (istate == 2) return rho * Uy;  //y-momentum
    else if (istate == 3) return rho * Uz;  //z-momentum
    else return 0;

}


// ========================================================
// Sod Shock Tube
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_SodShockTube<dim,nstate,real>
::ExactSolutionFunction_SodShockTube(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_SodShockTube<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Setting constants
    const double gam = 1.4;
    const double gam_p1 = gam + 1.0;
    const double gam_m1 = gam - 1.0;
    const double c_1 = sqrt(gam * 1.0 / 1.0);
    const double x_i = 0.5;
    const double t_final = 0.2;
    
    double rho_1 = 1.0;
    double Ux_1 = 0.0;
    double p_1 = 1.0;

    double rho = 0.0;
    double Ux = 0.0;
    double p = 0.0;

    if(point[0] < 0.26335680867601535) {
        rho = rho_1;
        Ux = Ux_1;
        p = p_1;
    } else if (point[0] < 0.4859454374877634) {
        Ux = 2.0 / gam_p1 * (c_1 + (point[0] - x_i) / t_final);

        double fact = 1 - 0.5 * gam_m1 * Ux / c_1;

        rho = rho_1 * pow(fact , (2.0 / gam_m1));
        p = p_1 * pow(fact , (2.0 * gam / gam_m1));
    } else if(point[0] < 0.6854905240097902) {
        rho = 0.42631942817849544;
        Ux = 0.92745262004895057;
        p = 0.30313017805064707;
    } else if(point[0] < 0.8504311464060357) {
        rho = 0.26557371170530725;
        Ux = 0.92745262004895057;
        p = 0.30313017805064707;
    } else {
        rho = 0.125;
        Ux = 0.0;
        p = 0.1;
    }

    //Convert to conservative variables
    if (istate == 0)      return rho;       //density 
    else if (istate == nstate-1) return p/(gam-1.0) + 0.5 * rho * (Ux*Ux);   //total energy
    else if (istate == 1) return rho * Ux;  //x-momentum
    
    else return 0;

}

//=========================================================
// FLOW SOLVER -- Exact Solution Base Class + Factory
//=========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction<dim,nstate,real>
::ExactSolutionFunction ()
    : dealii::Function<dim,real>(nstate)
{
    //do nothing
}

template <int dim, int nstate, typename real>
std::shared_ptr<ExactSolutionFunction<dim, nstate, real>>
ExactSolutionFactory<dim,nstate, real>::create_ExactSolutionFunction(
        const Parameters::FlowSolverParam& flow_solver_parameters, 
        const double time_compare)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = flow_solver_parameters.flow_case_type;
    if (flow_type == FlowCaseEnum::periodic_1D_unsteady){
        if constexpr (dim==1 && nstate==dim)  return std::make_shared<ExactSolutionFunction_1DSine<dim,nstate,real> > (time_compare);
    } else if (flow_type == FlowCaseEnum::isentropic_vortex){
        if constexpr (dim>1 && nstate==dim+2)  return std::make_shared<ExactSolutionFunction_IsentropicVortex<dim,nstate,real> > (time_compare);
    } else if (flow_type == FlowCaseEnum::sod_shock_tube){
        if constexpr (dim==1 && nstate==dim+2)  return std::make_shared<ExactSolutionFunction_SodShockTube<dim,nstate,real> > (time_compare);
    } else {
        // Select zero function if there is no exact solution defined
        return std::make_shared<ExactSolutionFunction_Zero<dim,nstate,real>> (time_compare);
    }
    return nullptr;
}

template class ExactSolutionFunction <PHILIP_DIM,PHILIP_DIM, double>;
template class ExactSolutionFunction <PHILIP_DIM,PHILIP_DIM+2, double>;
template class ExactSolutionFactory <PHILIP_DIM, PHILIP_DIM+2, double>;
template class ExactSolutionFactory <PHILIP_DIM, PHILIP_DIM, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,1, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,2, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,3, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,4, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,5, double>;

#if PHILIP_DIM>1
template class ExactSolutionFunction_IsentropicVortex <PHILIP_DIM,PHILIP_DIM+2, double>;
#endif
} // PHiLiP namespace
