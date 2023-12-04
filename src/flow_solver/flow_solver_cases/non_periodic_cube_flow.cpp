#include "non_periodic_cube_flow.h"
#include "mesh/grids/non_periodic_cube.h"
#include <deal.II/grid/grid_generator.h>
#include "physics/physics_factory.h"
#include "mesh/gmsh_reader.hpp"

namespace PHiLiP {
    namespace FlowSolver {

        template <int dim, int nstate>
        NonPeriodicCubeFlow<dim, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters* const parameters_input)
            : FlowSolverCaseBase<dim, nstate>(parameters_input)
            , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename + ".txt")
        {
            //create the Physics object
            this->pde_physics = std::dynamic_pointer_cast<Physics::PhysicsBase<dim, nstate, double>>(
                Physics::PhysicsFactory<dim, nstate, double>::create_Physics(parameters_input));
        }

        template <int dim, int nstate>
        std::shared_ptr<Triangulation> NonPeriodicCubeFlow<dim, nstate>::generate_grid() const
        {
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
                this->mpi_communicator
#endif
                );

            bool use_number_mesh_refinements = false;
            if (this->all_param.flow_solver_param.number_of_mesh_refinements > 0)
                use_number_mesh_refinements = true;

            const unsigned int number_of_refinements = use_number_mesh_refinements ? this->all_param.flow_solver_param.number_of_mesh_refinements
                : log2(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension);

            const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
            const double domain_right = this->all_param.flow_solver_param.grid_right_bound;

            const double n_subdivisions_0 = this->all_param.flow_solver_param.n_subdivisions_0;
            const double n_subdivisions_1 = this->all_param.flow_solver_param.n_subdivisions_1;

            const bool colorize = true;

            int left_boundary_id = 9999;
            using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
            flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

            if (flow_case_type == flow_case_enum::sod_shock_tube
                || flow_case_type == flow_case_enum::leblanc_shock_tube
                || flow_case_type == flow_case_enum::mach_3_wind_tunnel) {
                left_boundary_id = 1001;
            }
            else if (flow_case_type == flow_case_enum::shu_osher_problem) {
                left_boundary_id = 1004;
            }


            Grids::non_periodic_cube<dim>(*grid, domain_left, domain_right, colorize, left_boundary_id, n_subdivisions_0, n_subdivisions_1);
            if (dim == 1)
                grid->refine_global(number_of_refinements);

            return grid;
        }

        template <int dim, int nstate>
        void NonPeriodicCubeFlow<dim, nstate>::display_additional_flow_case_specific_parameters() const
        {
            this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
        }

        template <int dim, int nstate>
        double NonPeriodicCubeFlow<dim, nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim, double>> dg) const
        {
            // compute time step based on advection speed (i.e. maximum local wave speed)
            const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs() / nstate;
            const double approximate_grid_spacing = (this->all_param.flow_solver_param.grid_right_bound - this->all_param.flow_solver_param.grid_left_bound) / pow(number_of_degrees_of_freedom_per_state, (1.0 / dim));
            const double cfl_number = this->all_param.flow_solver_param.courant_friedrichs_lewy_number;
            const double time_step = cfl_number * approximate_grid_spacing / this->maximum_local_wave_speed;

            return time_step;
        }

        template <int dim, int nstate>
        double NonPeriodicCubeFlow<dim, nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim, double>> dg)
        {
            // initialize the maximum local wave speed
            update_maximum_local_wave_speed(*dg);
            // compute time step based on advection speed (i.e. maximum local wave speed)
            const double time_step = get_adaptive_time_step(dg);
            return time_step;
        }

        template<int dim, int nstate>
        void NonPeriodicCubeFlow<dim, nstate>::update_maximum_local_wave_speed(DGBase<dim, double>& dg)
        {
            // Initialize the maximum local wave speed to zero
            this->maximum_local_wave_speed = 0.0;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg.max_degree + 1 + overintegrate);
            dealii::FEValues<dim, dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double, nstate> soln_at_q;

            std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
            for (auto cell = dg.dof_handler.begin_active(); cell != dg.dof_handler.end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
                fe_values_extra.reinit(cell);
                cell->get_dof_indices(dofs_indices);

                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                    for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    // Update the maximum local wave speed (i.e. convective eigenvalue)
                    const double local_wave_speed = this->pde_physics->max_convective_eigenvalue(soln_at_q);
                    if (local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
                }
            }
            this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);
        }

        template<int dim, int nstate>
        std::array<double, 2> NonPeriodicCubeFlow<dim, nstate>::compute_max_density(DGBase<dim, double>& dg)
        {
            double max_density = 0.0;
            double x_location = 0.0;
            auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
            for (auto soln_cell = dg.dof_handler.begin_active(); soln_cell != dg.dof_handler.end(); ++soln_cell, ++metric_cell) {
                if (!soln_cell->is_locally_owned()) continue;

                std::vector<dealii::types::global_dof_index> current_dofs_indices;
                // Current reference element related to this physical cell
                const int i_fele = soln_cell->active_fe_index();
                const int poly_degree = i_fele;

                const dealii::FESystem<dim, dim>& current_fe_ref = dg.fe_collection[poly_degree];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
                const unsigned int n_metric_dofs = dg.high_order_grid->fe_system.dofs_per_cell;
                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                soln_cell->get_dof_indices(current_dofs_indices);

                // setup metric cell
                std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
                metric_cell->get_dof_indices(metric_dof_indices);

                // Allocate solution dofs and set global max and min
                for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                    const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
                    if (istate == 0) {
                        if (dg.solution[current_dofs_indices[idof]] > max_density) {
                            max_density = dg.solution[current_dofs_indices[idof]];
                            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                                double x = dg.high_order_grid->volume_nodes[metric_dof_indices[n_metric_dofs - 1]];
                                if (x > x_location) {
                                    x_location = x;
                                }
                            }
                        }
                    }
                }
            }

            return { {max_density,x_location} };
        }

        template <int dim, int nstate>
        void NonPeriodicCubeFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
        {
            std::array<double, 2> max_density = this->compute_max_density(*dg);

            if (current_iteration % 1 == 0) {
                if (this->mpi_rank == 0) {

                    unsteady_data_table->add_value("iteration", current_iteration);
                    // Add values to data table
                    this->add_value_to_data_table(current_time, "time", unsteady_data_table);
                    this->add_value_to_data_table(max_density[0], "max_density", unsteady_data_table);
                    this->add_value_to_data_table(max_density[1], "max_density_x", unsteady_data_table);

                    // Write to file
                    std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
                    unsteady_data_table->write_text(unsteady_data_table_file);
                }
                // Print to console
                this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << current_time
                    << "    Max Density: " << max_density[0]
                    << "    Max Density Location: " << max_density[1];

                this->pcout << std::endl;
            }
        }

#if PHILIP_DIM==2
        template class NonPeriodicCubeFlow<PHILIP_DIM, 1>;
        template class NonPeriodicCubeFlow<PHILIP_DIM, PHILIP_DIM + 2>;
#else
        template class NonPeriodicCubeFlow <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
    } // FlowSolver namespace
} // PHiLiP namespace
