#include "positivity_preserving_tests.h"
#include "mesh/grids/positivity_preserving_tests_grid.h"
#include <deal.II/grid/grid_generator.h>
#include "physics/physics_factory.h"
#include "mesh/gmsh_reader.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
PositivityPreservingTests<dim, nstate>::PositivityPreservingTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : CubeFlow_UniformGrid<dim, nstate>(parameters_input)
    , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    //create the Physics object
    this->pde_physics = std::dynamic_pointer_cast<Physics::PhysicsBase<dim,nstate,double>>(
                Physics::PhysicsFactory<dim,nstate,double>::create_Physics(parameters_input));
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> PositivityPreservingTests<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );
    
    if(dim >= 1) {
        if(this->all_param.flow_solver_param.grid_xmax == this->all_param.flow_solver_param.grid_xmin) {
            std::cout << "Error: xmax and xmin need to be provided as parameters - Aborting... " << std::endl << std::flush;
            std::abort();
        }
    }

    if(dim >= 2) {
        if(this->all_param.flow_solver_param.grid_ymax == this->all_param.flow_solver_param.grid_ymin) {
            std::cout << "Error: ymax and ymin need to be provided as parameters - Aborting... " << std::endl << std::flush;
            std::abort();
        }
    }

    if(dim == 3) {
        if(this->all_param.flow_solver_param.grid_zmax == this->all_param.flow_solver_param.grid_zmin) {
            std::cout << "Error: zmax and zmin need to be provided as parameters - Aborting... " << std::endl << std::flush;
            std::abort();
        }
    }
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if (dim==1 && (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube
        || flow_case_type == flow_case_enum::shu_osher_problem)) {
        Grids::shock_tube_1D_grid<dim>(*grid, &this->all_param);
    }
    else if (flow_case_type == flow_case_enum::sod_shock_tube) {
        Grids::explosion_problem_grid<dim>(*grid, &this->all_param);
    }
    else if ((dim == 2 || dim == 3) && flow_case_type == flow_case_enum::explosion_problem) {
        Grids::explosion_problem_grid<dim>(*grid, &this->all_param);
    }
    else if (dim == 2 && flow_case_type == flow_case_enum::leblanc_shock_tube) {
        Grids::explosion_problem_grid<dim>(*grid, &this->all_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::sedov_blast_wave) {
        Grids::sedov_blast_wave_grid<dim>(*grid, &this->all_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::mach_3_wind_tunnel) {
        Grids::mach_3_wind_tunnel_grid<dim>(*grid, &this->all_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::shock_diffraction) {
        Grids::shock_diffraction_grid<dim>(*grid, &this->all_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::astrophysical_jet) {
        Grids::astrophysical_jet_grid<dim>(*grid, &this->all_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::daru_tenaud) {
        Grids::daru_tenaud_grid<dim>(*grid, &this->all_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::double_mach_reflection) {
        if(this->all_param.flow_solver_param.use_gmsh_mesh) {
            const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
            const bool use_mesh_smoothing = false;
            std::shared_ptr<HighOrderGrid<dim,double>> dmr_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
            return dmr_mesh->triangulation;
        } else {
            Grids::double_mach_reflection_grid<dim>(*grid, &this->all_param);
        }
    }

    return grid;
}

template <int dim, int nstate>
void PositivityPreservingTests<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

template<int dim, int nstate>
void PositivityPreservingTests<dim, nstate>::check_positivity_density(DGBase<dim, double>& dg)
{
    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = dg.max_grid_degree;
    const unsigned int poly_degree = this->all_param.flow_solver_param.poly_degree;
    //Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, double> soln_basis(1, poly_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, double> soln_basis_projection_oper(1, dg.max_degree, init_grid_degree);


    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);

    for (auto soln_cell = dg.dof_handler.begin_active(); soln_cell != dg.dof_handler.end(); ++soln_cell) {
        if (!soln_cell->is_locally_owned()) continue;


        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = dg.fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<double>, nstate> soln_coeff;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local max and min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = dg.solution[current_dofs_indices[idof]];
        }

        const unsigned int n_quad_pts = dg.volume_quadrature_collection[poly_degree].size();

        std::array<std::vector<double>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate], soln_basis.oneD_vol_operator);
        }

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            // Verify that positivity of density is preserved
            if (soln_at_q[0][iquad] < -1e-13) {
                std::cout << "Error: Density is negative - Aborting... " << std::endl << std::flush;
                std::abort();
            }

            if ((isnan(soln_at_q[0][iquad])) ) {
                std::cout << "Error: Density is NaN - Aborting... " << std::endl << std::flush;
                std::abort();
            }
        }
    }
}

template <int dim, int nstate>
void PositivityPreservingTests<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table,
    const bool do_write_unsteady_data_table_file)
{
    //unpack current iteration and current time from ode solver
    const unsigned int current_iteration = ode_solver->current_iteration;
    const double current_time = ode_solver->current_time;

    // Update maximum local wave speed for adaptive time_step
    if(this->all_param.flow_solver_param.adaptive_time_step) this->update_maximum_local_wave_speed(*dg);

    this->check_positivity_density(*dg);
    if (this->mpi_rank == 0) {

        unsteady_data_table->add_value("iteration", current_iteration);
        // Add values to data table
        this->add_value_to_data_table(current_time, "time", unsteady_data_table);


        // Write to file
        if(do_write_unsteady_data_table_file){
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);    
        }
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
            << "    Time: " << current_time;

        this->pcout << std::endl;
    }
}

template class PositivityPreservingTests<PHILIP_DIM, PHILIP_DIM+2>;

} // FlowSolver namespace
} // PHiLiP namespace