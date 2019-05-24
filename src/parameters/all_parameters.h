#ifndef __ALL_PARAMETERS_H__
#define __ALL_PARAMETERS_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters.h"
#include "parameters/parameters_ode_solver.h"
#include "parameters/parameters_linear_solver.h"
#include "parameters/parameters_manufactured_convergence_study.h"


namespace Parameters
{
    using namespace dealii;

    /// Main parameter class that contains the various other sub-parameter classes.
    class AllParameters
    {
    public:
        /// Constructor
        AllParameters();

        /// Contains parameters for manufactured convergence study
        ManufacturedConvergenceStudyParam manufactured_convergence_study_param;
        /// Contains parameters for ODE solver
        ODESolverParam ode_solver_param;
        /// Contains parameters for linear solver
        LinearSolverParam linear_solver_param;

        /// Number of dimensions. Note that it has to match the executable PHiLiP_xD
        unsigned int dimension;


        /// Number of state variables. Will depend on PDE
        int nstate;

        /// Currently allows to solve advection, diffusion, convection-diffusion
        enum PartialDifferentialEquation { 
            advection,
            diffusion,
            convection_diffusion,
            advection_vector};
        /// Store the PDE type to be solved
        PartialDifferentialEquation pde_type;

        /// Currently only Lax-Friedrichs can be used as an input parameter
        enum ConvectiveNumericalFlux { lax_friedrichs };
        /// Store convective flux type
        ConvectiveNumericalFlux conv_num_flux_type;

        /// Currently only symmetric internal penalty can be used as an input parameter
        enum DissipativeNumericalFlux { symm_internal_penalty };
        /// Store diffusive flux type
        DissipativeNumericalFlux diss_num_flux_type;

        /// Declare parameters that can be set as inputs and set up the default options
        /** This subroutine should call the sub-parameter classes static declare_parameters()
          * such that each sub-parameter class is responsible to declare their own parameters.
          */
        static void declare_parameters (ParameterHandler &prm);

        /// Retrieve parameters from ParameterHandler
        /** This subroutine should call the sub-parameter classes static parse_parameters()
          * such that each sub-parameter class is responsible to parse their own parameters.
          */
        void parse_parameters (ParameterHandler &prm);

        //FunctionParser<dim> initial_conditions;
        //BoundaryConditions  boundary_conditions[max_n_boundaries];
    };  
}

#endif
