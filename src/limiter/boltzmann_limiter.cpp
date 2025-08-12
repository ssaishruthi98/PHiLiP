#include "boltzmann_limiter.h"
#include "positivity_preserving_limiter.h"
#include "tvb_limiter.h"
#include <eigen/unsupported/Eigen/Polynomials>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
/**********************************
*
* Boltzmann Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
BoltzmannLimiter<dim, nstate, real>::BoltzmannLimiter(
    const Parameters::AllParameters* const parameters_input)
    : PositivityPreservingLimiter<dim,nstate,real>::PositivityPreservingLimiter(parameters_input)
    , flow_solver_param(parameters_input->flow_solver_param)
    , dx((flow_solver_param.grid_xmax-flow_solver_param.grid_xmin)/flow_solver_param.number_of_grid_elements_x)
    , dy((flow_solver_param.grid_ymax-flow_solver_param.grid_ymin)/flow_solver_param.number_of_grid_elements_y)
    , dz((flow_solver_param.grid_zmax-flow_solver_param.grid_zmin)/flow_solver_param.number_of_grid_elements_z)
    , resolution(flow_solver_param.resolution)
    , first_run(true)
{
    // Create pointer to Euler Physics to compute pressure if pde_type==euler
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PDE_enum pde_type = parameters_input->pde_type;

    std::shared_ptr< ManufacturedSolutionFunction<dim, real> >  manufactured_solution_function
        = ManufacturedSolutionFactory<dim, real>::create_ManufacturedSolution(parameters_input, nstate);

    if (pde_type == PDE_enum::euler && nstate == dim + 2) {
        euler_physics = std::make_shared < Physics::Euler<dim, nstate, real> >(
            parameters_input,
            parameters_input->euler_param.ref_length,
            parameters_input->euler_param.gamma_gas,
            parameters_input->euler_param.mach_inf,
            parameters_input->euler_param.angle_of_attack,
            parameters_input->euler_param.side_slip_angle,
            manufactured_solution_function,
            parameters_input->two_point_num_flux_type);
    }
    else {
        std::cout << "Error: Positivity-Preserving Limiter can only be applied for pde_type==euler" << std::endl;
        std::abort();
    }

    // Create pointer to TVB Limiter class if use_tvb_limiter==true && dim == 1
    if (parameters_input->limiter_param.use_tvb_limiter) {
        if (dim == 1) {
            tvbLimiter = std::make_shared < TVBLimiter<dim, nstate, real> >(parameters_input);
        }
        else {
            std::cout << "Error: Cannot create TVB limiter for dim > 1" << std::endl;
            std::abort();
        }
    }

    if(dim >= 2 && (flow_solver_param.number_of_grid_elements_x == 1 || flow_solver_param.number_of_grid_elements_y == 1)) {
        std::cout << "Error: number_of_grid_elements must be passed for all directions to use PPL Limiter." << std::endl;
        std::abort();
    }

    if(dim == 3 && flow_solver_param.number_of_grid_elements_z == 1) {
        std::cout << "Error: number_of_grid_elements must be passed for all directions to use PPL Limiter." << std::endl;
        std::abort();
    }
}

template <int dim, int nstate, typename real>
std::vector<real> BoltzmannLimiter<dim, nstate, real>::get_integrating_domain(
    const std::array<std::vector<real>, nstate>&    soln_at_q,
    const unsigned int                              n_quad_pts,
    const double                                    k)
        // from Dzanic, Martinelli 2025 3.7: k=4 bounds relative error by approximately 6e-5 and k=8 bound relative error by approximately 1e-15
{
    std::vector<real> bounds(2, 0.0);                   // lower and upper bounds of the microscopic velocity distribution function domain for integration
    std::array<real, nstate> soln_at_iquad;

    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_at_iquad[istate] = soln_at_q[istate][iquad];
        }
        
        // Did not account for non-Euler style situations as in the get_theta2_Wang2012 function
        real U = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[1];   // hard-coded for 1D. For multidimensional, need to generalize

        real density = soln_at_iquad[0];
        real pressure = euler_physics->compute_pressure(soln_at_iquad);
        if(pressure > 1e9) {
            std::cout << density << "   " << soln_at_iquad[1] << "    " << soln_at_iquad[2] << std::endl;
        }
        real theta = pressure / density;
        
        real pot_lower_bound = U - k * sqrt(theta);
        real pot_upper_bound = U + k * sqrt(theta);

        bounds[0] = std::min(pot_lower_bound, bounds[0]);
        bounds[1] = std::max(pot_upper_bound, bounds[1]);
    }

    return bounds;
}

template <int dim, int nstate, typename real>
std::vector< std::vector<real> >  BoltzmannLimiter<dim, nstate, real>::get_boltzmann_distribution(
    const std::array<std::vector<real>, nstate>&                                                soln_at_q_dim,   // _dim added just to differentiate from soln_at_q which is passed in as soln_at_q[0]
    const unsigned int                                                                          n_quad_pts,
    const double                                                                                resolution,
    const double                                                                                lower_distribution_limit,
    const double                                                                                upper_distribution_limit,
    const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>>            mapping_field,
    dealii::QGaussLobatto<dim>                                                                  quad_for_l2_norm,
    const dealii::hp::FECollection<dim>&                                                        fe_collection,
    const int                                                                                   poly_degree)
{
    const int num_u = static_cast<int>((upper_distribution_limit - lower_distribution_limit) / resolution) + 1;
    if(num_u < 0) {
        std::cout << "Error: Integrating limits are diverging from nonphysical values....Aborting" << std::endl;
        std::cout << "upper_distribution_limit:   " << upper_distribution_limit << "    lower_distribution_limit:   " << lower_distribution_limit << std::endl;
        std::abort();
    }
    std::vector< std::vector<real> > output_points(3, std::vector<real>(num_u));

    real pi = std::acos(-1.0);
    
    
    std::vector<real> f_min(num_u, std::numeric_limits<real>::max());
    std::vector<real> f_max(num_u, std::numeric_limits<real>::lowest());
    std::vector< std::vector<real> > g(num_u, std::vector<real>(n_quad_pts, 0.0));

    dealii::FEValues<dim, dim> fe_values(*mapping_field, fe_collection[poly_degree], quad_for_l2_norm,
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    for (int i = 0; i < num_u; ++i) {

        real u = lower_distribution_limit + i * resolution;
        std::array<real, nstate> soln_at_iquad;                                // creates fixed-size array for working with state vectors
        real l2_squared = 0.0;
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            for (unsigned int istate = 0; istate < nstate; ++istate) {          // iterates through each state variable (ρ, m, E)
                soln_at_iquad[istate] = soln_at_q_dim[istate][iquad];               // sets state vector do be manipulated in the loop
            }

            real U = 0.0;
            if (nstate == dim + 2)                                            // checks if it is a NS or Euler problem
                U = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[1];

            l2_squared += pow(u - U, 2.0) * fe_values.JxW(iquad);
        }

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            real pressure = 0.0;
            real density = soln_at_iquad[0];

            if (nstate == dim + 2)                                          // checks if it is a NS or Euler problem
                pressure = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[2];
            
            real theta = pressure/density;
            
            g[i][iquad] = (density/(pow(2*pi*theta, dim/2.0)))*exp(-l2_squared/(2*theta));
            //pow(density, dim / 2.0 + 1.0) / (pow(2 * pi * pressure, dim / 2.0)) * exp(-density / (2 * pressure) * l2_squared);

            f_min[i] = std::min(f_min[i], g[i][iquad]);
            f_max[i] = std::max(f_max[i], g[i][iquad]);
        }

        output_points[0][i] = u;
        output_points[1][i] = f_min[i];
        output_points[2][i] = f_max[i];
    }

    return output_points;
}

template <int dim, int nstate, typename real>
// lower bounds are in [0][ ], upper bounds are in [1][ ]
// density bounds - [][0], momentum bounds - [][1]:[][dim], energy bounds - [][dim+1]
std::vector< std::vector<real>> BoltzmannLimiter<dim, nstate, real>::boltzmann_limits(
    const std::vector<real>&            u_values,
    const std::vector<real>&            f_min_values,
    const std::vector<real>&            f_max_values)
{
    std::vector<std::vector<real>> limits(2, std::vector<real>(dim + 1));       // match the limiter to the size of the state vector based on # of dimensions

    const std::size_t N = u_values.size();

    // define density, momentum, and energy limiters
    real rho_min = 0.0;
    real rho_max = 0.0;
    real momentum_min = 0.0;
    real momentum_max = 0.0;
    real E_min = 0.0;
    real E_max = 0.0;

    for (std::size_t i = 0; i < N - 1; ++i) {
        double du = u_values[i + 1] - u_values[i];          // made generic to the first stepsize assuming constant stepsize
        real u_ave = 0.5 * (u_values[i + 1] + u_values[i]);
        real f_min_ave = 0.5 * (f_min_values[i] + f_min_values[i + 1]);
        real f_max_ave = 0.5 * (f_max_values[i] + f_max_values[i + 1]);

        rho_min += f_min_ave * du;
        rho_max += f_max_ave * du;
        if(u_ave > 0) {
            momentum_min += f_max_ave * u_ave * du;
            momentum_max += f_min_ave * u_ave * du;
        } else {
            momentum_min += f_min_ave * u_ave * du;
            momentum_max += f_max_ave * u_ave * du;
        }
        E_min += 0.5 * u_ave * u_ave * f_min_ave * du;
        E_max += 0.5 * u_ave * u_ave * f_max_ave * du;
    }

    limits[0][0] = rho_min;
    limits[1][0] = rho_max;

    // replace this part with a for loop when implementing for multiple dims!!
    limits[0][1] = momentum_min;
    limits[1][1] = momentum_max;


    limits[0][nstate-1] = E_min;
    limits[1][nstate-1] = E_max;

    // std::cout << "density-min: " << rho_min << ", density-max: " << rho_max << ", momentum-min: " << momentum_min << ", momentum-max: " 
    //    << momentum_max << ", energy-min: " << E_min << ", energy-max: " << E_max << std::endl;
    
    return limits;
}

template <int dim, int nstate, typename real>
real BoltzmannLimiter<dim, nstate, real>::get_alpha(
    const std::array<std::vector<real>, nstate>&    soln_at_q_dim,
    const unsigned int                              n_quad_pts,
    const std::array<real, nstate>&                 soln_cell_avg,
    const std::vector<real>&                        soln_cell_min,
    const std::vector<real>&                        soln_cell_max)
{
    real alpha = 1.0;

    // finds max and min deviations of a quadrature point's state vector from the cell-averaged state vector
    // std::vector<real> min_values(nstate, 1e12);
        // arbitrary high value for starting minimization function
    // std::vector<real> max_values(nstate,-1e12);
        // arbitrary low value for starting minimization function (because momentum could be negative)

    // min and max expressions from Eq. (37)
    std::vector<real> min_state_values(nstate);
    std::vector<real> max_state_values(nstate);

    std::vector<real> min_denominators(nstate);
    std::vector<real> max_denominators(nstate);    


    for (int istate = 0; istate < nstate; ++istate) {
        
        // initialize values using the first quad point
        min_state_values[istate] = soln_at_q_dim[istate][0];
        max_state_values[istate] = soln_at_q_dim[istate][0];
        
        // iterate through the rest of the quad points and obtain minimum and maximum values
        for (unsigned int iquad = 1; iquad < n_quad_pts; ++iquad){

            // replace minimum value if lower than previous minimum
            if (soln_at_q_dim[istate][iquad] < min_state_values[istate])
                min_state_values[istate] = soln_at_q_dim[istate][iquad];

            if(min_state_values[istate] == 1e9)
                std::cout << "the solution at the quadrature point for state  " << istate << "  is  " << soln_at_q_dim[istate][iquad] << std::endl;
            
            // replace maximum value if greater than previous minimum
            if (soln_at_q_dim[istate][iquad] > max_state_values[istate])
                max_state_values[istate] = soln_at_q_dim[istate][iquad];

        }

        min_denominators[istate] = min_state_values[istate] - soln_cell_avg[istate];
        max_denominators[istate] = max_state_values[istate] - soln_cell_avg[istate];
        
    }

    for (int istate = 0; istate < nstate; ++istate) {
        real max_term = std::abs((soln_cell_max[istate] - soln_cell_avg[istate]) / max_denominators[istate]);
        real min_term = std::abs((soln_cell_min[istate] - soln_cell_avg[istate]) / min_denominators[istate]);
        alpha = std::min(max_term, alpha);
        alpha = std::min(min_term, alpha);
        // std::cout << "\t istate: " << istate << ", entry " << 1 + 2 * istate << ": " << max_term <<
        //     ", entry " << 2 + 2 * istate << ": " << min_term << std::endl;
    }

    return alpha;
}


template <int dim, int nstate, typename real>
void BoltzmannLimiter<dim, nstate, real>::write_limited_solution(
    dealii::LinearAlgebra::distributed::Vector<double>& solution,
    const std::array<std::vector<real>, nstate>& soln_coeff,
    const unsigned int                                      n_shape_fns,
    const std::vector<dealii::types::global_dof_index>& current_dofs_indices)
{
    // Write limited solution dofs to the global solution vector.
    for (int istate = 0; istate < nstate; istate++) {
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            const unsigned int idof = istate * n_shape_fns + ishape;
            solution[current_dofs_indices[idof]] = soln_coeff[istate][ishape]; //

            // Verify that positivity of density is preserved after application of theta2 limiter
            if (istate == 0 && solution[current_dofs_indices[idof]] < 0) {
                std::cout << "Error: Density is a negative value - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }

            // Verify that positivity of Total Energy is preserved after application of theta2 limiter
            if (istate == (nstate - 1) && solution[current_dofs_indices[idof]] < 0) {
                std::cout << "Error: Total Energy is a negative value - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }

            // Verify that the solution values haven't been changed to NaN as a result of all quad pts in a cell having negative density 
            // (all quad pts having negative density would result in the local maximum convective eigenvalue being zero leading to division by zero)
            if (isnan(solution[current_dofs_indices[idof]])) {
                std::cout << "Error: Solution is NaN - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }
        }
    }
}

template <int dim, int nstate, typename real>
void BoltzmannLimiter<dim, nstate, real>::limit(
    dealii::LinearAlgebra::distributed::Vector<double>&                                         solution,
    const dealii::DoFHandler<dim>&                                                              dof_handler,
    const dealii::hp::FECollection<dim>&                                                        fe_collection,
    const dealii::hp::QCollection<dim>&                                                         volume_quadrature_collection,
    const unsigned int                                                                          grid_degree,
    const unsigned int                                                                          max_degree,
    const dealii::hp::FECollection<1>                                                           oneD_fe_collection_1state,
    const dealii::hp::QCollection<1>                                                            oneD_quadrature_collection,
    double                                                                                      dt,
    double                                                                                      current_time,
    bool                                                                                        is_it_a_stage,
    dealii::Vector<double>&                                                                     alpha_value,
    const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>>            mapping_field) 
{
    std::cout << "Running limit";

    // If use_tvb_limiter is true, apply TVB limiter before applying maximum-principle-satisfying limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection, dt, current_time, is_it_a_stage, alpha_value, mapping_field);

    //create 1D solution polynomial basis functions to interpolate the solution to the quadrature nodes
    const unsigned int init_grid_degree = grid_degree;

    // Construct 1D Quad Points
    dealii::QGauss<1> oneD_quad_GL(max_degree + 1);
    dealii::QGaussLobatto<1> oneD_quad_GLL(max_degree + 1);

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_GLL(1, max_degree, init_grid_degree);
    soln_basis_GLL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quad_GLL);
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_GL(1, max_degree, init_grid_degree);
    soln_basis_GL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quad_GL);

    if(first_run) {
        state_max.resize(nstate,1e-9);
        state_min.resize(nstate,1e9);
    }

    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();
        const dealii::types::global_dof_index cell_index = soln_cell->active_cell_index();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<real>, nstate> soln_coeff;

        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        bool nan_check = false;
        // Allocate solution dofs and set local min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]];

            if (isnan(soln_coeff[istate][ishape])) {
                nan_check = true;
            }
        }

        const unsigned int n_quad_pts = n_shape_fns;

        if (nan_check) {
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                std::cout << "Error: Density passed to limiter is NaN - Aborting... " << std::endl;

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    std::cout << soln_coeff[istate][iquad] << "    ";
                }

                std::cout << std::endl;
                std::abort();
            }  
        }

        std::array<std::array<std::vector<real>, nstate>, dim> soln_at_q;
        std::array<std::vector<real>, nstate> soln_at_q_dim;
        // Interpolate solution dofs to quadrature pts.
        for(unsigned int idim = 0; idim < dim; idim++) {
            for (int istate = 0; istate < nstate; istate++) {
                soln_at_q_dim[istate].resize(n_quad_pts);

                if(idim == 0) {
                    soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                        soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
                }

                if(idim == 1) {
                    soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                        soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
                }

                if(idim == 2) {
                    soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                        soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator);
                }
            }
            soln_at_q[idim] = soln_at_q_dim;
        }

        std::vector< real > GLL_weights = oneD_quad_GLL.get_weights();
        std::vector< real > GL_weights = oneD_quad_GL.get_weights();
        std::array<real, nstate> soln_cell_avg;
        // Obtain solution cell average
        soln_cell_avg = get_soln_cell_avg_PPL(soln_at_q, n_quad_pts, oneD_quad_GLL.get_weights(), oneD_quad_GL.get_weights(), dt);

        real lower_bound = this->all_parameters->limiter_param.min_density;
        real p_avg = 1e-13;

        if (nstate == dim + 2) {
            // Compute average value of pressure using soln_cell_avg
            p_avg = euler_physics->compute_pressure(soln_cell_avg);
        }

        real theta = 1.0;
        if(!first_run) {
            // using parameters shown, including soln_cell_min and _max, obtain alpha scaling factor for first scaling
            // std::cout << cell_index << "<<<< SOLUTION CELL INDEX" << std::endl;
            theta = get_alpha(soln_at_q_dim, n_quad_pts, soln_cell_avg, state_min, state_max);
            alpha_value[cell_index] = theta;

            // if(cell_index > 10 && cell_index < 15)
            //     sleep(5);

            // Apply limiter on density values at quadrature points
            for(int istate = 0; istate < nstate; ++istate) {
                for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                    soln_coeff[istate][ishape] = theta*(soln_coeff[istate][ishape] - soln_cell_avg[istate]) + soln_cell_avg[istate];
                }
            }

            // Interpolate new solution dofs to quadrature pts.
            for(unsigned int idim = 0; idim < dim; idim++) {
                for (int istate = 0; istate < nstate; istate++) {
                    soln_at_q_dim[istate].resize(n_quad_pts);

                    if(idim == 0) {
                        soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                            soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
                    }

                    if(idim == 1) {
                        soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                            soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
                    }

                    if(idim == 2) {
                        soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                            soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator);
                    }
                }
                soln_at_q[idim] = soln_at_q_dim;
            }
        }


        if (dim == 1) {
            // getting integrating domain limits for the cell for the distribution function based on k standard deviations around macroscopic velocity, U
            std::array<real, 2> integrating_limits;
            for (int i = 0; i < 2; ++i)
                integrating_limits[i] = get_integrating_domain(soln_at_q[0], n_quad_pts, 4.0)[i];
                                                                                        //   ^   this is the k-value; k=4 here

            dealii::QGaussLobatto<dim> quad_for_l2_norm(poly_degree + 1);
            // use the integrating domain limits to develop the min-max f-function against microscopic velocity (u) points
            std::vector< std::vector<real> > min_max_envelope = get_boltzmann_distribution(soln_at_q[0], n_quad_pts, this->resolution, integrating_limits[0], integrating_limits[1], mapping_field, quad_for_l2_norm, fe_collection, poly_degree);
                                                                                                                    //  ^  this is the resolution of the boltmann distribution plot
            std::vector<std::vector<real>> cell_max_and_mins = boltzmann_limits(min_max_envelope[0], min_max_envelope[1], min_max_envelope[2]);

            for(int istate = 0; istate < nstate; ++istate) {
                if(state_max[istate] < cell_max_and_mins[1][istate])
                    state_max[istate] = cell_max_and_mins[1][istate];
                if(state_min[istate] > cell_max_and_mins[0][istate])
                    state_min[istate] = cell_max_and_mins[0][istate];
            }
        }


///// 2D implementation /////
        if (dim == 2) {

            /// (1) get_integrating_domain ///

            std::vector<real> u_bounds(2, 0.0);
            std::vector<real> v_bounds(2, 0.0);
            std::array<real, nstate> soln_at_iquad;
            int k = 4;

            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    soln_at_iquad[istate] = soln_at_q[0][istate][iquad];
                }

                real U = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[1];
                real V = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[2];

                real density = soln_at_iquad[0];
                real pressure = euler_physics->compute_pressure(soln_at_iquad);

                if(pressure > 1e9)
                    std::cout << density << "   " << soln_at_iquad[1] << "    " << soln_at_iquad[2] << std::endl;

                real theta = pressure / density;              

                u_bounds[0] = std::min(U - k * sqrt(theta), u_bounds[0]);
                u_bounds[1] = std::max(U + k * sqrt(theta), u_bounds[1]);
                v_bounds[0] = std::min(V - k * sqrt(theta), v_bounds[0]);
                v_bounds[1] = std::max(V + k * sqrt(theta), v_bounds[1]);
            }    

            /// (1) /////////////////////////

            dealii::QGaussLobatto<2> quad_for_l2_norm(poly_degree + 1);

            /// (2+3) get_boltzmann_distribution + boltzmann_limits ///

            const int num_u = static_cast<int>((u_bounds[1] - u_bounds[0]) / resolution) + 1;
            const int num_v = static_cast<int>((v_bounds[1] - v_bounds[0]) / resolution) + 1;
            if(num_u < 0) {
                std::cout << "Error: Integrating limits are diverging from nonphysical values....Aborting" << std::endl;
                std::cout << "u lower bound:   " << u_bounds[0] << "    u upper bound:   " << u_bounds[1] << std::endl;
                std::abort();
            }
            if(num_v < 0) {
                std::cout << "Error: Integrating limits are diverging from nonphysical values....Aborting" << std::endl;
                std::cout << "v lower bound:   " << v_bounds[0] << "    v upper bound:   " << v_bounds[1] << std::endl;
                std::abort();
            }

            real pi = std::acos(-1.0);

            std::vector<std::vector<real>> f_min( num_u, std::vector<real>(num_v, std::numeric_limits<real>::max()) );
            std::vector<std::vector<real>> f_max( num_u, std::vector<real>(num_v, std::numeric_limits<real>::lowest()) );
            std::vector <std::vector <std::vector<real>>> g( num_u, std::vector <std::vector<real>>(num_v, std::vector<real>(n_quad_pts, 0.0)) );
            
            // initializing boltzmann limits
            real rho_min = 0.0;
            real rho_max = 0.0;
            real u_momentum_min = 0.0;
            real u_momentum_max = 0.0;
            real v_momentum_min = 0.0;
            real v_momentum_max = 0.0;
            real E_min = 0.0;
            real E_max = 0.0;

            dealii::FEValues<dim, dim> fe_values(*mapping_field, fe_collection[poly_degree], quad_for_l2_norm,
                dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            
            for (int i = 0; i < num_u; ++i) {
                
                real u = u_bounds[0] + i * resolution;

                real l2_squared = 0.0;
                    
                for (int j = 0; j < num_v; ++j) {
                    
                    real v = v_bounds[0] + j * resolution;

                    std::array<real, nstate> soln_at_iquad;

                    // computing l2_squared
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        for (unsigned int istate = 0; istate < nstate; ++istate) {          // iterates through each state variable (ρ, m, E)
                            soln_at_iquad[istate] = soln_at_q_dim[istate][iquad];               // sets state vector do be manipulated in the loop
                        }

                        real U = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[1];
                        real V = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[2];

                        // l2_squared += (pow(u - U, 2.0) + pow(v - V, 2.0));                                   // sums together L2 norm across element excluding quad weights
                        l2_squared += (pow(u - U, 2.0) + pow(v - V, 2.0)) * fe_values.JxW(iquad);               // sums together L2 norm across element including quad weights   
                        std::cout << "U = " << U << ", V = " << V << ", fe_values.JxW(iquad) = " << fe_values.JxW(iquad) << ", l2_squared = " << l2_squared << std::endl;
                    }

                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {

                        real pressure = 0.0;

                        real density = soln_at_iquad[0];

                        if (nstate == dim + 2)                                          // checks if it is a NS or Euler problem
                            pressure = euler_physics->convert_conservative_to_primitive(soln_at_iquad)[dim+1];

                        real theta = pressure/density;

                        g[i][j][iquad] = (density/(pow(2*pi*theta, dim/2.0)))*exp(-l2_squared/(2*theta));
                        std::cout << "density = " << density << ", theta = " << theta << ", l2_squared = " << l2_squared << ", g = " << g[i][j][iquad] << std::endl;

                        // std::cout << "u = " << u << ", v = " << v << ": g = " << g[i][j][iquad] << std::endl;

                        f_min[i][j] = std::min(f_min[i][j], g[i][j][iquad]);
                        f_max[i][j] = std::max(f_max[i][j], g[i][j][iquad]);
                    }
                    // output_points[0][i][j] = u;
                    // output_points[1][i][j] = v;
                    // output_points[2][i][j] = f_min[i][j];
                    // output_points[3][i][j] = f_max[i][j];

                }

            }

            for (int i = 0; i < num_u - 1; ++i) {

                real u = u_bounds[0] + i * resolution;
                real u_ave = u + resolution / 2;

                for (int j = 0; j < num_v - 1; ++j) {

                    real v = v_bounds[0] + j * resolution;
                    real v_ave = v + resolution / 2;

                    real u_v_dot_product = pow(u_ave, 2.0) + pow(v_ave, 2.0);

                    real f_min_ave = 0.25 * (f_min[i][j] + f_min[i + 1][j] + f_min[i][j+1] + f_min[i+1][j+1]);
                    real f_max_ave = 0.25 * (f_max[i][j] + f_max[i + 1][j] + f_max[i][j+1] + f_max[i+1][j+1]);

                    real dudv = resolution * resolution;

                    rho_min += f_min_ave * dudv;
                    rho_max += f_max_ave * dudv;
                    
                    if(u_ave > 0) {
                        u_momentum_min += f_max_ave * u_ave * dudv;
                        u_momentum_max += f_min_ave * u_ave * dudv;
                    } else {
                        u_momentum_min += f_min_ave * u_ave * dudv;
                        u_momentum_max += f_max_ave * u_ave * dudv;
                    }

                    if(v_ave > 0) {
                        v_momentum_min += f_max_ave * v_ave * dudv;
                        v_momentum_max += f_min_ave * v_ave * dudv;
                    } else {
                        v_momentum_min += f_min_ave * v_ave * dudv;
                        v_momentum_max += f_max_ave * v_ave * dudv;
                    }

                    E_min += 0.5 * u_v_dot_product * f_min_ave * dudv;
                    E_max += 0.5 * u_v_dot_product * f_max_ave * dudv;
                }
            }

            /////////////////////////// (2+3) ///////////////////////////

            ////////////////// (4) get_alpha //////////////////
            real alpha = 1.0;

            std::vector<real> min_state_values(nstate);
            std::vector<real> max_state_values(nstate);

            std::vector<real> min_denominators(nstate);
            std::vector<real> max_denominators(nstate);    

            for (int istate = 0; istate < nstate; ++istate) {
                
                // initialize values using the first quad point
                min_state_values[istate] = soln_at_q_dim[istate][0];
                max_state_values[istate] = soln_at_q_dim[istate][0];
                
                // iterate through the rest of the quad points and obtain minimum and maximum values
                for (unsigned int iquad = 1; iquad < n_quad_pts; ++iquad){

                    // replace minimum value if lower than previous minimum
                    if (soln_at_q_dim[istate][iquad] < min_state_values[istate])
                        min_state_values[istate] = soln_at_q_dim[istate][iquad];

                    if(min_state_values[istate] == 1e9)
                        std::cout << "the solution at the quadrature point for state  " << istate << "  is  " << soln_at_q_dim[istate][iquad] << std::endl;
                    
                    // replace maximum value if greater than previous minimum
                    if (soln_at_q_dim[istate][iquad] > max_state_values[istate])
                        max_state_values[istate] = soln_at_q_dim[istate][iquad];
                }

                min_denominators[istate] = min_state_values[istate] - soln_cell_avg[istate];
                max_denominators[istate] = max_state_values[istate] - soln_cell_avg[istate];
            }

            for (int istate = 0; istate < nstate; ++istate) {
                real max_term = std::abs((max_state_values[istate] - soln_cell_avg[istate]) / max_denominators[istate]);
                real min_term = std::abs((min_state_values[istate] - soln_cell_avg[istate]) / min_denominators[istate]);

                alpha = std::min(max_term, alpha);
                alpha = std::min(min_term, alpha);
            }
            ////////////////////   (4)   //////////////////////
        }
            

        // Get epsilon (lower bound for rho) for theta limiter
        if(state_min[0] < lower_bound)
            state_min[0] = lower_bound;

        real theta2 = 1.0;

        if (nstate == dim + 2) {
            std::array<real, dim> theta2_quad;
            for(unsigned int idim = 0; idim < dim; ++idim) {
                theta2_quad[idim] = get_theta2_Wang2012(soln_at_q[idim], n_quad_pts, p_avg);
            }

            for(unsigned int idim = 0; idim < dim; ++idim) {
                if(theta2_quad[idim] < theta2)
                    theta2 = theta2_quad[idim];
            }

            real theta2_soln = get_theta2_Wang2012(soln_coeff, n_quad_pts, p_avg);
            if(theta2_soln < theta2)
                    theta2 = theta2_soln;

            // Limit values at quadrature points
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    soln_coeff[istate][iquad] = theta2 * (soln_coeff[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                }
            }
        }

        // Write limited solution back and verify that positivity of density is satisfied
        write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);
    }
    if(first_run) {
        first_run = false;
    }
}

template class BoltzmannLimiter <PHILIP_DIM, PHILIP_DIM + 2, double>;

} // PHiLiP namespace
