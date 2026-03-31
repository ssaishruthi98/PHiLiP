#include "multispecies_tests.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_periodic_cube.hpp"
#include "mesh/grids/positivity_preserving_tests_grid.h"
#include "mesh/gmsh_reader.hpp"
#include "physics/physics_factory.h"

namespace PHiLiP {

namespace FlowSolver {
//==========================================================================
// FLOW SOLVER CASE FOR TESTS INVOLVING MULTISPECIES AND RELATED PARAMETERS
//==========================================================================
template <int dim, int nspecies, int nstate>
MultispeciesTests<dim, nspecies, nstate>::MultispeciesTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : CubeFlow_UniformGrid<dim, nspecies, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ 
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_param.flow_solver_param.flow_case_type;

    // Flow case identifiers
    this->is_taylor_green_vortex = (flow_type == FlowCaseEnum::taylor_green_vortex);
    this->is_viscous_flow = (this->all_param.pde_type == Parameters::AllParameters::PartialDifferentialEquation::navier_stokes_real_gas);
    
    // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = PDE_enum::navier_stokes_real_gas;
    this->ns_real_gas_physics = std::dynamic_pointer_cast<Physics::NavierStokes_RealGas<dim,nspecies,dim+nspecies+1,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+nspecies+1,double>::create_Physics(&parameters_navier_stokes));

    /* Initialize integrated quantities as NAN; 
       done as a precaution in the case compute_integrated_quantities() is not called
       before a member function of kind get_integrated_quantity() is called
     */
    std::fill(this->integrated_quantities.begin(), this->integrated_quantities.end(), NAN);

    // Initialize the integrated kinetic energy as NAN
    this->integrated_kinetic_energy_at_previous_time_step = NAN;
}

template <int dim, int nspecies, int nstate>
std::shared_ptr<Triangulation> MultispeciesTests<dim,nspecies,nstate>::generate_grid() const
{
    
    this->pcout << "- Generating grid using dealii GridGenerator" << std::endl;
    
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
        this->mpi_communicator
    #endif
    );
        
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if(dim==1 && flow_case_type == flow_case_enum::multi_species_sod_shock_tube) {
        Grids::shock_tube_1D_grid<dim>(*grid, &this->all_param.flow_solver_param);
    } else if (dim==2 && flow_case_type == flow_case_enum::multi_species_hydrogen_injection) {
        Grids::hydrogen_injection_grid<dim>(*grid, &this->all_param.flow_solver_param);        
    } else {
        Grids::straight_periodic_cube<dim, Triangulation>(grid, domain_left, domain_right,
                                                            number_of_cells_per_direction);
    }
    return grid;

}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim,nspecies,nstate>::display_grid_parameters() const
{
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    std::string grid_type_string = "";
    if(flow_case_type == flow_case_enum::multi_species_sod_shock_tube) {
        grid_type_string = "1d_shock_tube";
    } else if (flow_case_type == flow_case_enum::multi_species_hydrogen_injection) {
        grid_type_string = "hydrogen_injection_chamber";
    } else {
        grid_type_string = "straight_periodic_cube";
    }
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << this->domain_left << std::endl;
    this->pcout << "- - Domain right: " << this->domain_right << std::endl;

    int cells_in_each_dir = 0;
    if(this->number_of_cells_per_direction > this->all_param.flow_solver_param.number_of_grid_elements_x)
        cells_in_each_dir = this->number_of_cells_per_direction;
    else
        cells_in_each_dir = this->all_param.flow_solver_param.number_of_grid_elements_x;

    this->pcout << "- - Number of cells in each direction: " << cells_in_each_dir << std::endl;
    if constexpr(dim==1) this->pcout << "- - Domain length: " << this->domain_size << std::endl;
    if constexpr(dim==2) this->pcout << "- - Domain area: " << this->domain_size << std::endl;
    if constexpr(dim==3) this->pcout << "- - Domain volume: " << this->domain_size << std::endl;
}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim,nspecies,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->display_grid_parameters();
}

template<int dim, int nspecies, int nstate>
void MultispeciesTests<dim, nspecies, nstate>::compute_and_update_integrated_quantities(DGBase<dim, nspecies, double> &dg)
{
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integral_values;
    std::fill(integral_values.begin(), integral_values.end(), 0.0);
    
    // Initialize the maximum local wave speed to zero; only used for adaptive time step
    if(this->all_param.flow_solver_param.adaptive_time_step == true || this->all_param.flow_solver_param.error_adaptive_time_step == true) this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 0;

    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::QGauss<1> quad_extra_1D(dg.max_degree+1+overintegrate);

    const unsigned int n_quad_pts = quad_extra.size();
    const unsigned int grid_degree = dg.high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = dg.max_degree;
    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, grid_degree);
    // Build basis function volume operator and gradient operator from 1D finite element for 1 state.
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    soln_basis.build_1D_gradient_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    // Build mapping shape functions operators using the oneD high_ordeR_grid finite element
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg.high_order_grid->oneD_fe_system, dg.high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg.high_order_grid->oneD_fe_system, quad_extra_1D, dg.oneD_face_quadrature);
    // Construct and build projection operator
    OPERATOR::vol_projection_operator<dim,2*dim,double> soln_basis_projection_oper(1, poly_degree, grid_degree);
    soln_basis_projection_oper.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    const std::vector<double> &quad_weights = quad_extra.get_weights();
    // If in the future we need the physical quadrature node location, turn these flags to true and the constructor will
    // automatically compute it for you. Currently set to false as to not compute extra unused terms.
    bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position

    // if(this->do_compute_angular_momentum) store_vol_flux_nodes = true;
    
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    const unsigned int n_dofs = dg.fe_collection[poly_degree].n_dofs_per_cell();
    const unsigned int n_shape_fns = n_dofs / nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs);
    auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = dg.dof_handler.begin_active(); cell!= dg.dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg.high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg.high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, store_vol_flux_nodes, store_surf_flux_nodes);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg.all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
         
            soln_coeff[istate][ishape] = dg.solution(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q_vect;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
            // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                soln_grad_at_q_vect[istate][idim].resize(n_quad_pts);
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.
            soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        soln_grad_at_q_vect[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                  * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                  / metric_oper.det_Jac_vol[iquad];
                    }
                }
            }
        }

        std::array<std::vector<double>,3> vorticity_at_q_vect;// putting nstate as 3 for the 3 vorticity components
        // Resize for n_quad_pts
        for(int istate=0; istate<3; istate++){
            vorticity_at_q_vect[istate].resize(n_quad_pts);
        }
        // Store vorticity at quadrature points
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            // Extract solution and gradient in a way that the physics can use them.
            std::array<double,nstate> soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_at_q[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                }
            }
            dealii::Tensor<1,3,double> vorticity_at_q = this->ns_real_gas_physics->compute_vorticity(soln_at_q,soln_grad_at_q);
            for(int istate=0; istate<3; istate++){
                vorticity_at_q_vect[istate][iquad] = vorticity_at_q[istate];
            }
        }
        
        // Now compute and store the gradient of vorticity:
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<dealii::Tensor<1,dim,std::vector<double>>,3> vorticity_grad_at_q_vect;
        for(int istate=0; istate<3; istate++){
            std::vector<double> vorticity_coeff(n_shape_fns);
            soln_basis_projection_oper.matrix_vector_mult_1D(vorticity_at_q_vect[istate], vorticity_coeff,
                                                              soln_basis_projection_oper.oneD_vol_operator);
            // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                vorticity_grad_at_q_vect[istate][idim].resize(n_quad_pts);
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.
            soln_basis.gradient_matrix_vector_mult_1D(vorticity_coeff, ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        vorticity_grad_at_q_vect[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                  * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                  / metric_oper.det_Jac_vol[iquad];
                    }
                }
            }
        }


        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;
            dealii::Tensor<1,3,double> vorticity_at_q;
            std::array<dealii::Tensor<1,dim,double>,3> vorticity_grad_at_q;
            // Extract solution and gradient in a way that the physics can use them.
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
                if(istate<3) vorticity_at_q[istate] = vorticity_at_q_vect[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_at_q[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                    if(istate<3) vorticity_grad_at_q[istate][idim] = vorticity_grad_at_q_vect[istate][idim][iquad];
                }
            }
            // dealii::Point<dim> qpoint;
            // if(this->do_compute_angular_momentum){
            //     for(int idim=0; idim<dim; idim++){
            //         qpoint[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            //     }
            // }

            // If you want to include the commented quantities, make sure the associated functions are implemented in ns_real_gas
            // Then add the integrated quantity to the enum in the header file
            std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrand_values;
            std::fill(integrand_values.begin(), integrand_values.end(), 0.0);
            integrand_values[IntegratedQuantitiesEnum::kinetic_energy] = this->ns_real_gas_physics->compute_kinetic_energy_from_conservative_solution(soln_at_q);
            integrand_values[IntegratedQuantitiesEnum::enstrophy] = this->ns_real_gas_physics->compute_enstrophy(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::pressure_dilatation] = this->ns_real_gas_physics->compute_pressure_dilatation(soln_at_q,soln_grad_at_q);
            // integrand_values[IntegratedQuantitiesEnum::viscosity_times_deviatoric_strain_rate_tensor_magnitude_sqr] = this->ns_real_gas_physics->compute_viscosity_times_deviatoric_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);
            // integrand_values[IntegratedQuantitiesEnum::viscosity_times_strain_rate_tensor_magnitude_sqr] = this->ns_real_gas_physics->compute_viscosity_times_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::incompressible_kinetic_energy] = this->ns_real_gas_physics->compute_incompressible_kinetic_energy_from_conservative_solution(soln_at_q);
            integrand_values[IntegratedQuantitiesEnum::incompressible_enstrophy] = this->ns_real_gas_physics->compute_incompressible_enstrophy(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::incompressible_palinstrophy] = this->ns_real_gas_physics->compute_incompressible_palinstrophy(soln_at_q,vorticity_grad_at_q);
            // if(this->do_compute_angular_momentum) integrand_values[IntegratedQuantitiesEnum::angular_momentum] = this->compute_angular_momentum(qpoint,vorticity_at_q);
            // else integrand_values[IntegratedQuantitiesEnum::angular_momentum] = 0.0;
            for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
                integral_values[i_quantity] += integrand_values[i_quantity] * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue) if using an adaptive time step
            if(this->all_param.flow_solver_param.adaptive_time_step == true || this->all_param.flow_solver_param.error_adaptive_time_step == true) {
                const double local_wave_speed = this->ns_real_gas_physics->max_convective_eigenvalue(soln_at_q);
                if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
            }
        }
    }
    if(this->all_param.flow_solver_param.adaptive_time_step == true) {
        this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);
    }
    // update integrated quantities
    for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
        this->integrated_quantities[i_quantity] = dealii::Utilities::MPI::sum(integral_values[i_quantity], this->mpi_communicator);
        this->integrated_quantities[i_quantity] /= this->domain_size; // divide by total domain volume
    }
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_integrated_kinetic_energy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::kinetic_energy];
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_integrated_enstrophy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_integrated_incompressible_kinetic_energy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::incompressible_kinetic_energy];
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_integrated_incompressible_enstrophy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::incompressible_enstrophy];
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_integrated_incompressible_palinstrophy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::incompressible_palinstrophy];
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_vorticity_based_dissipation_rate() const
{
    const double integrated_enstrophy = this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
    double vorticity_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        vorticity_based_dissipation_rate = this->ns_real_gas_physics->compute_vorticity_based_dissipation_rate_from_integrated_enstrophy(integrated_enstrophy);
    }
    return vorticity_based_dissipation_rate;
}

template <int dim, int nspecies, int nstate>
double MultispeciesTests<dim, nspecies, nstate>::get_pressure_dilatation_based_dissipation_rate() const
{
    const double integrated_pressure_dilatation = this->integrated_quantities[IntegratedQuantitiesEnum::pressure_dilatation];
    return (-1.0*integrated_pressure_dilatation); // See reference in PeriodicTurbulence Header file
}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim, nspecies, nstate>::compute_unsteady_data_and_write_to_table(
    const std::shared_ptr<ODE::ODESolverBase<dim, nspecies, double>> ode_solver,
    const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    //unpack current iteration and current time from ode solver
    const unsigned int current_iteration = ode_solver->current_iteration;
    const double current_time = ode_solver->current_time;

    // Compute and update integrated quantities
    if(this->is_viscous_flow) {
        this->compute_and_update_integrated_quantities(*dg);
        // Get computed quantities
        const double integrated_kinetic_energy = this->get_integrated_kinetic_energy();
        const double integrated_enstrophy = this->get_integrated_enstrophy();
        const double vorticity_based_dissipation_rate = this->get_vorticity_based_dissipation_rate();
        const double pressure_dilatation_based_dissipation_rate = this->get_pressure_dilatation_based_dissipation_rate();
        // const double deviatoric_strain_rate_tensor_based_dissipation_rate = this->get_deviatoric_strain_rate_tensor_based_dissipation_rate();
        // const double strain_rate_tensor_based_dissipation_rate = this->get_strain_rate_tensor_based_dissipation_rate();
        const double integrated_incompressible_kinetic_energy = this->get_integrated_incompressible_kinetic_energy();
        const double integrated_incompressible_enstrophy = this->get_integrated_incompressible_enstrophy();
        const double integrated_incompressible_palinstrophy = this->get_integrated_incompressible_palinstrophy();
        // double integrated_angular_momentum = 0.0;
        // if(this->do_compute_angular_momentum) integrated_angular_momentum = this->get_integrated_angular_momentum();

        if(this->mpi_rank==0) {
            // Add values to data table
            this->add_value_to_data_table(current_time,"time",unsteady_data_table);
            this->add_value_to_data_table(integrated_kinetic_energy,"kinetic_energy",unsteady_data_table);
            this->add_value_to_data_table(integrated_enstrophy,"enstrophy",unsteady_data_table);
            if(is_viscous_flow) this->add_value_to_data_table(vorticity_based_dissipation_rate,"eps_vorticity",unsteady_data_table);
            this->add_value_to_data_table(pressure_dilatation_based_dissipation_rate,"eps_pressure",unsteady_data_table);
            // if(is_viscous_flow) this->add_value_to_data_table(strain_rate_tensor_based_dissipation_rate,"eps_strain",unsteady_data_table);
            // if(is_viscous_flow) this->add_value_to_data_table(deviatoric_strain_rate_tensor_based_dissipation_rate,"eps_dev_strain",unsteady_data_table);
            this->add_value_to_data_table(integrated_incompressible_kinetic_energy,"incompressible_kinetic_energy",unsteady_data_table);
            this->add_value_to_data_table(integrated_incompressible_enstrophy,"incompressible_enstrophy",unsteady_data_table);
            this->add_value_to_data_table(integrated_incompressible_palinstrophy,"incompressible_palinstrophy",unsteady_data_table);
            // if(this->do_compute_angular_momentum) this->add_value_to_data_table(integrated_angular_momentum,"angular_momentum",unsteady_data_table);
            // Write to file
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    } else {

        if (this->mpi_rank == 0) {

            unsteady_data_table->add_value("iteration", current_iteration);
            // Add values to data table
            this->add_value_to_data_table(current_time, "time", unsteady_data_table);
            // Write to file
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << std::setprecision(16) << current_time;

        this->pcout << std::endl;
    }

    // Update local maximum wave speed before calculating next time step
    update_maximum_local_wave_speed(*dg);
}

template class MultispeciesTests <PHILIP_DIM, PHILIP_SPECIES,PHILIP_DIM+PHILIP_SPECIES+1>;
} // FlowSolver namespace
} // PHiLiP namespace

