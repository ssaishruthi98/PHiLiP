#include "limiter_convergence_tests.h"
#include <iostream>
#include <stdlib.h>
#include "mesh/gmsh_reader.hpp"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include "mesh/grids/straight_periodic_cube.hpp"


namespace PHiLiP{
namespace FlowSolver{

template <int dim, int nstate>
LimiterConvergenceTests<dim, nstate>::LimiterConvergenceTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
std::shared_ptr<Triangulation> LimiterConvergenceTests<dim,nstate>::generate_grid() const
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

    double left = this->all_param.flow_solver_param.grid_left_bound;
    double right = this->all_param.flow_solver_param.grid_right_bound;
    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;

    PHiLiP::Grids::straight_periodic_cube<dim, Triangulation>(grid, left, right, pow(2.0, number_of_refinements));

    std::cout << "Grid generated and refined" << std::endl;

    return grid;
}

template <int dim, int nstate>
double LimiterConvergenceTests<dim, nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim, double>> dg) const
{
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = this->all_param.flow_solver_param.flow_case_type;

    double left = this->all_param.flow_solver_param.grid_left_bound;
    double right = this->all_param.flow_solver_param.grid_right_bound;

    const unsigned int n_global_active_cells = dg->triangulation->n_global_active_cells();
    const unsigned int n_dofs_cfl = dg->dof_handler.n_dofs() / nstate;
    double delta_x = (PHILIP_DIM == 2) ? (right - left) / pow(n_global_active_cells, (1.0 / dim)) : (right - left) / pow(n_dofs_cfl, (1.0 / dim));
    double time_step = 1e-5;

    /**********************************
    * These values for the time step are chosen to show dominant spatial accuracy in the OOA results for P2
    * For >=P3 timestep values  refer to: 
    * Zhang, Xiangxiong, and Chi-Wang Shu. 
    * "On maximum-principle-satisfying high order schemes for scalar conservation laws." 
    * Journal of Computational Physics 229.9 (2010): 3091-3120.
    **********************************/
   
    if(flow_case == Parameters::FlowSolverParam::FlowCaseType::advection_limiter)
        time_step = (PHILIP_DIM == 2) ? (1.0 / 14.0) * delta_x : (1.0 / 3.0) * pow(delta_x, 2.0);
    
    if(flow_case == Parameters::FlowSolverParam::FlowCaseType::burgers_limiter)
        time_step = (PHILIP_DIM == 2) ? (1.0 / 14.0) * delta_x : (1.0 / 24.0) * delta_x;

    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density_2d)
        time_step = (1.0 / 50.0) * pow(delta_x , 2.0);

    return time_step;
}

template <int dim, int nstate>
double LimiterConvergenceTests<dim, nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim, double>> dg)
{
    // compute time step for each case such that results show dominant spatial accuracy
    const double time_step = get_adaptive_time_step(dg);

    return time_step;
}

template <int dim, int nstate>
void LimiterConvergenceTests<dim, nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

#if PHILIP_DIM==1
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM>;
#elif PHILIP_DIM==2
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM>;
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM+2>;
    template class LimiterConvergenceTests<PHILIP_DIM, 1>;
#endif

}
}