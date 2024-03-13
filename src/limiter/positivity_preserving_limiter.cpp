#include "positivity_preserving_limiter.h"
#include "tvb_limiter.h"

namespace PHiLiP {
/**********************************
*
* Positivity Preserving Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
PositivityPreservingLimiter<dim, nstate, real>::PositivityPreservingLimiter(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiterState<dim,nstate,real>::BoundPreservingLimiterState(parameters_input)
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

    this->dx = (parameters_input->flow_solver_param.grid_right_bound-parameters_input->flow_solver_param.grid_left_bound)
                /parameters_input->flow_solver_param.number_of_grid_elements_x;
    this->dy = (parameters_input->flow_solver_param.grid_top_bound-parameters_input->flow_solver_param.grid_bottom_bound)
                /parameters_input->flow_solver_param.number_of_grid_elements_y;
}

template <int dim, int nstate, typename real>
std::vector<real> PositivityPreservingLimiter<dim, nstate, real>::get_theta2_Zhang2010(
    const std::vector< real >&                      p_lim,
    const std::array<real, nstate>&                 soln_cell_avg,
    const std::array<std::vector<real>, nstate>&    soln_at_q,
    const unsigned int                              n_quad_pts,
    const double                                    eps,
    const double                                    gamma)
{
    std::vector<real> theta2(n_quad_pts, 1); // Value used to linearly scale state variables
    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        if (p_lim[iquad] >= eps) {
            theta2[iquad] = 1;
        }
        else {
            real s_coeff1 = (soln_at_q[nstate - 1][iquad] - soln_cell_avg[nstate - 1]) * (soln_at_q[0][iquad] - soln_cell_avg[0]) - 0.5 * pow(soln_at_q[1][iquad] - soln_cell_avg[1], 2);

            real s_coeff2 = soln_cell_avg[0] * (soln_at_q[nstate - 1][iquad] - soln_cell_avg[nstate - 1]) + soln_cell_avg[nstate - 1] * (soln_at_q[0][iquad] - soln_cell_avg[0])
                - soln_cell_avg[1] * (soln_at_q[1][iquad] - soln_cell_avg[1]) - (eps / gamma) * (soln_at_q[0][iquad] - soln_cell_avg[0]);

            real s_coeff3 = (soln_cell_avg[nstate - 1] * soln_cell_avg[0]) - 0.5 * pow(soln_cell_avg[1], 2) - (eps / gamma) * soln_cell_avg[0];

            if (dim > 1) {
                s_coeff1 -= 0.5 * pow(soln_at_q[2][iquad] - soln_cell_avg[2], 2);
                s_coeff2 -= soln_cell_avg[2] * (soln_at_q[2][iquad] - soln_cell_avg[2]);
                s_coeff3 -= 0.5 * pow(soln_cell_avg[2], 2);
            }

            if (dim > 2) {
                s_coeff1 -= 0.5 * pow(soln_at_q[3][iquad] - soln_cell_avg[3], 2);
                s_coeff2 -= soln_cell_avg[3] * (soln_at_q[3][iquad] - soln_cell_avg[3]);
                s_coeff3 -= 0.5 * pow(soln_cell_avg[3], 2);
            }

            real discriminant = s_coeff2 * s_coeff2 - 4 * s_coeff1 * s_coeff3;

            if (discriminant >= 0) {
                real t1 = (-s_coeff2 + sqrt(discriminant)) / (2 * s_coeff1);
                real t2 = (-s_coeff2 - sqrt(discriminant)) / (2 * s_coeff1);
                theta2[iquad] = std::min(t1, t2);
            }
            else {
                theta2[iquad] = 0;
            }
        }
    }

    return theta2;
}

template <int dim, int nstate, typename real>
real PositivityPreservingLimiter<dim, nstate, real>::get_theta2_Wang2012(
    const std::array<std::vector<real>, nstate>&    soln_at_q,
    const unsigned int                              n_quad_pts,
    const double                                    p_avg)
{
    std::vector<real> t2(n_quad_pts, 1);
    real theta2 = 1.0; // Value used to linearly scale state variables 
    std::array<real, nstate> soln_at_iquad;

    // Obtain theta2 value
    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_at_iquad[istate] = soln_at_q[istate][iquad];
        }
        real p_lim = 0;

        if (nstate == dim + 2)
            p_lim = euler_physics->compute_pressure(soln_at_iquad);

        if (p_lim >= 0)
            t2[iquad] = 1;
        else
            t2[iquad] = p_avg / (p_avg - p_lim);

        if (t2[iquad] != 1) {
            if (t2[iquad] >= 0 && t2[iquad] <= 1 && t2[iquad] < theta2) {
                theta2 = t2[iquad];
            }
        }
    }
    return theta2;
}

template <int dim, int nstate, typename real>
real PositivityPreservingLimiter<dim, nstate, real>::get_theta2_Zhang2016(
    const std::array<std::vector<real>, nstate>& soln_at_q,
    const unsigned int                              n_quad_pts,
    const double                                    p_avg,
    const double                                    lower_bound)
{
    real theta2 = 1.0; // Value used to linearly scale state variables
    real gamm1 = euler_physics->gamm1;
    std::array<real, nstate> soln_at_iquad;
    double internalEnergy_avg = p_avg / gamm1;
    double internalEnergy_min = 1.0e6;

    // Obtain theta2 value
    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_at_iquad[istate] = soln_at_q[istate][iquad];
        }
        real p_lim = 0;

        if (nstate == dim + 2)
            p_lim = euler_physics->compute_pressure(soln_at_iquad);
        
        double internalEnergy_lim = p_lim / gamm1;

        if (internalEnergy_lim < internalEnergy_min)
            internalEnergy_min = internalEnergy_lim;
    }

    theta2 = (internalEnergy_avg - internalEnergy_min < 0 + lower_bound) ? 1.0 : std::min(1.0,(internalEnergy_avg - lower_bound) / (internalEnergy_avg - internalEnergy_min));
    //std::cout << internalEnergy_avg << "   " << lower_bound << "   " << internalEnergy_min << "   " << theta2 << "   " << std::endl;
    return theta2;
}

template <int dim, int nstate, typename real>
real PositivityPreservingLimiter<dim, nstate, real>::get_density_scaling_value(
    const double    density_avg,
    const double    density_min,
    const double    lower_bound,
    const double    p_avg)
{
    // Get epsilon (lower bound for rho) for theta limiter
    real eps = std::min({ lower_bound, density_avg, p_avg });
    if (eps < 0) eps = lower_bound;

    real theta = 1.0; // Value used to linearly scale density 
    if (density_avg - density_min > 1e-13)
        theta = (density_avg - density_min == 0) ? 1.0 : std::min((density_avg - eps) / (density_avg - density_min), 1.0);

    if (theta < 0 || theta > 1)
        theta = 0;

    return theta;
}

template <int dim, int nstate, typename real>
void PositivityPreservingLimiter<dim, nstate, real>::write_limited_solution(
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
        }
    }
}

template <int dim, int nstate, typename real>
void PositivityPreservingLimiter<dim, nstate, real>::limit(
    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
    const unsigned int                                      grid_degree,
    const unsigned int                                      max_degree,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
    const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
    double                                                  dt)
{
    // If use_tvb_limiter is true, apply TVB limiter before applying positivity-preserving limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection, dt);

    if (dim == 2) {
        limit_2D(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection, dt);
    }
    else {
        //create 1D solution polynomial basis functions and corresponding projection operator
        //to interpolate the solution to the quadrature nodes, and to project it back to the
        //modal coefficients.
        const unsigned int init_grid_degree = grid_degree;

        // Constructor for the operators
        OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis(1, max_degree, init_grid_degree);
        OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper(1, max_degree, init_grid_degree);

        // Build the oneD operator to perform interpolation/projection
        soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
        soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

        for (auto soln_cell : dof_handler.active_cell_iterators()) {
            if (!soln_cell->is_locally_owned()) continue;

            std::vector<dealii::types::global_dof_index> current_dofs_indices;
            // Current reference element related to this physical cell
            const int i_fele = soln_cell->active_fe_index();
            const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
            const int poly_degree = current_fe_ref.tensor_degree();

            const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            // Obtain the mapping from local dof indices to global dof indices
            current_dofs_indices.resize(n_dofs_curr_cell);
            soln_cell->get_dof_indices(current_dofs_indices);

            // Extract the local solution dofs in the cell from the global solution dofs
            std::array<std::vector<real>, nstate> soln_coeff;

            const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
            real local_min_density = 1e6;

            for (unsigned int istate = 0; istate < nstate; ++istate) {
                soln_coeff[istate].resize(n_shape_fns);
            }

            // Allocate solution dofs and set local min
            for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
                const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
                soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]]; //

                // if (istate == 0 && soln_coeff[istate][ishape] < local_min_density)
                //     local_min_density = soln_coeff[istate][ishape];
            }

            const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
            const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();
            std::array<std::vector<real>, nstate> soln_at_q;

            // Interpolate solution dofs to quadrature pts.
            for (int istate = 0; istate < nstate; istate++) {
                soln_at_q[istate].resize(n_quad_pts);
                soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                    soln_basis.oneD_vol_operator);
            }

            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                if (soln_at_q[0][iquad] < local_min_density)
                    local_min_density = soln_at_q[0][iquad];
            }
            // Obtain solution cell average
            std::array<real, nstate> soln_cell_avg = get_soln_cell_avg(soln_at_q, n_quad_pts, quad_weights);

            real lower_bound = this->all_parameters->limiter_param.min_density;
            real p_avg = 1e-13;

            if (nstate == dim + 2) {
                // Compute average value of pressure using soln_cell_avg
                p_avg = euler_physics->compute_pressure(soln_cell_avg);
            }
            
            // Obtain value used to linearly scale density
            real theta = get_density_scaling_value(soln_cell_avg[0], local_min_density, lower_bound, p_avg);

            // Apply limiter on density values at quadrature points
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                if (isnan(soln_at_q[0][iquad])) {
                    std::cout << "Error: Density at quadrature point is NaN - Aborting... " << std::endl << std::flush;
                    std::abort();
                }
                soln_at_q[0][iquad] = theta * (soln_at_q[0][iquad] - soln_cell_avg[0])
                    + soln_cell_avg[0];
            }

            using limiter_enum = Parameters::LimiterParam::LimiterType;
            limiter_enum limiter_type = this->all_parameters->limiter_param.bound_preserving_limiter;

            if (limiter_type == limiter_enum::positivity_preservingZhang2010 && nstate == dim + 2) {
                std::vector< real > p_lim(n_quad_pts);
                std::array<real, nstate> soln_at_iquad;

                // Compute pressure at quadrature points
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        soln_at_iquad[istate] = soln_at_q[istate][iquad];
                    }
                    p_lim[iquad] = euler_physics->compute_pressure(soln_at_iquad);
                }

                // Obtain value used to linearly scale state variables
                std::vector<real> theta2 = get_theta2_Zhang2010(p_lim, soln_cell_avg, soln_at_q, n_quad_pts, lower_bound, euler_physics->gam);

                // Limit values at quadrature points
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        soln_at_q[istate][iquad] = theta2[iquad] * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                    }
                }
            }

            else if (limiter_type == limiter_enum::positivity_preservingWang2012 && nstate == dim + 2) {
                real theta2 = get_theta2_Wang2012(soln_at_q, n_quad_pts, p_avg); // Value used to linearly scale state variables 

                // Limit values at quadrature points
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        soln_at_q[istate][iquad] = theta2 * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                    }
                }
            }

            else if (limiter_type == limiter_enum::positivity_preservingZhang2016 && nstate == dim + 2) {
                real theta2 = get_theta2_Zhang2016(soln_at_q, n_quad_pts, p_avg, lower_bound); // Value used to linearly scale state variables 

                // Limit values at quadrature points
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        soln_at_q[istate][iquad] = theta2 * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                    }
                }
            }

            // Project soln at quadrature points to dofs.
            for (int istate = 0; istate < nstate; istate++) {
                soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_coeff[istate], soln_basis_projection_oper.oneD_vol_operator);
            }

            // Write limited solution back and verify that positivity of density is satisfied
            write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);
        }
    }
}


template <int dim, int nstate, typename real>
std::array<real, nstate> PositivityPreservingLimiter<dim, nstate, real>::get_soln_cell_avg_2D(
    const std::array<std::vector<real>, nstate>&        soln_at_q_1,
    const std::array<std::vector<real>, nstate>&        soln_at_q_2,
    const unsigned int                                  n_quad_pts,
    const std::vector<real>&                            quad_weights_1,
    const std::vector<real>&                            quad_weights_2,
    double                                              dt)
{
    // Obtain solution cell average
    // std::array<real, nstate> soln_cell_avg = get_soln_cell_avg(soln_at_q, n_quad_pts, quad_weights);
    std::array<real, nstate> soln_cell_avg;
    std::array<real, nstate> soln_cell_avg_xGL = get_soln_cell_avg(soln_at_q_1, n_quad_pts, quad_weights_1);
    std::array<real, nstate> soln_cell_avg_yGL = get_soln_cell_avg(soln_at_q_2, n_quad_pts, quad_weights_2);

    for (unsigned int istate = 0; istate < nstate; ++istate) {
        soln_cell_avg[istate] = 0;
    }

    real lambda_1 = dt/this->dx; real lambda_2 = dt/this->dy;

    real max_local_wave_speed_1 = 0.0;
    real max_local_wave_speed_2 = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        std::array<real,nstate> local_soln_at_q_1;
        std::array<real,nstate> local_soln_at_q_2;
        for(unsigned int istate = 0; istate < nstate; ++istate){
            local_soln_at_q_1[istate] = soln_at_q_1[istate][iquad];
            local_soln_at_q_2[istate] = soln_at_q_2[istate][iquad];
        }
        // Update the maximum local wave speed (i.e. convective eigenvalue)
        const real local_wave_speed_1 = this->euler_physics->max_convective_eigenvalue(local_soln_at_q_1);
        const real local_wave_speed_2 = this->euler_physics->max_convective_eigenvalue(local_soln_at_q_2);
        if(local_wave_speed_1 > max_local_wave_speed_1) max_local_wave_speed_1 = local_wave_speed_1;
        if(local_wave_speed_2 > max_local_wave_speed_2) max_local_wave_speed_2 = local_wave_speed_2;
    }

    real mu = max_local_wave_speed_1*lambda_1 + max_local_wave_speed_2*lambda_2;
    real avg_weight_1 = (max_local_wave_speed_1*lambda_1)/mu;
    real avg_weight_2 = (max_local_wave_speed_2*lambda_2)/mu;

    for (unsigned int istate = 0; istate < nstate; istate++) {
        soln_cell_avg[istate] = avg_weight_1*soln_cell_avg_xGL[istate] + avg_weight_2*soln_cell_avg_yGL[istate];
    }

    return soln_cell_avg;
}

template <int dim, int nstate, typename real>
void PositivityPreservingLimiter<dim, nstate, real>::limit_2D(
    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    const dealii::hp::QCollection<dim>&                     /*volume_quadrature_collection*/,
    const unsigned int                                      grid_degree,
    const unsigned int                                      max_degree,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
    const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
    double                                                  dt)
{
    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = grid_degree;

    dealii::QGauss<1> quad_GL(max_degree+1);
    dealii::QGaussLobatto<1> quad_GLL(max_degree+1);

    std::vector< dealii::Point< 1 > > quad_GL_pts = quad_GL.get_points();
    std::vector< dealii::Point< 1 > > quad_GLL_pts = quad_GLL.get_points();
    std::vector< real > quad_GL_weights = quad_GL.get_weights();
    std::vector< real > quad_GLL_weights = quad_GLL.get_weights();

    dealii::Quadrature<2> quad_xGLyGLL(quad_GL,quad_GLL);
    dealii::Quadrature<2> quad_yGLxGLL(quad_GLL,quad_GL);

    std::vector< dealii::Point< 2 > > quad_xGLyGLL_pts = quad_xGLyGLL.get_points();
    std::vector< real > quad_xGLyGLL_weights = quad_xGLyGLL.get_weights();


    std::vector< dealii::Point< 2 > > quad_yGLxGLL_pts = quad_yGLxGLL.get_points();
    std::vector< real > quad_yGLxGLL_weights = quad_yGLxGLL.get_weights();

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_xGL(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper_xGL(1, max_degree, init_grid_degree);
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_yGL(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper_yGL(1, max_degree, init_grid_degree);

    // Build the oneD operator to perform interpolation/projection
    soln_basis_xGL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_projection_oper_xGL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_yGL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_projection_oper_yGL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<real>, nstate> soln_coeff;

        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        real local_min_density = 1e6;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            if (isnan(soln_coeff[istate][ishape])) {
                std::cout << "Error: Value passed to limiter is NaN - Aborting... " << std::endl << std::flush;
                std::abort();
            }
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]]; //
        }

        const unsigned int n_quad_pts = quad_xGLyGLL.size();
        const std::vector<real>& quad_weights_xGL = quad_xGLyGLL.get_weights();
        const std::vector<real>& quad_weights_yGL = quad_yGLxGLL.get_weights();
        std::array<std::vector<real>, nstate> soln_at_q_xGL;
        std::array<std::vector<real>, nstate> soln_at_q_yGL;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q_xGL[istate].resize(n_quad_pts);
            soln_basis_xGL.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_xGL[istate],
                soln_basis_xGL.oneD_vol_operator);

            soln_at_q_yGL[istate].resize(n_quad_pts);
            soln_basis_yGL.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_yGL[istate],
                soln_basis_yGL.oneD_vol_operator);
        }

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            if (soln_at_q_xGL[0][iquad] < local_min_density)
                local_min_density = soln_at_q_xGL[0][iquad];
            if (soln_at_q_yGL[0][iquad] < local_min_density)
                local_min_density = soln_at_q_yGL[0][iquad];
        }

        std::array<real, nstate> soln_cell_avg = get_soln_cell_avg_2D(soln_at_q_xGL, soln_at_q_yGL, n_quad_pts, quad_weights_xGL, quad_weights_yGL, dt);

        real lower_bound = this->all_parameters->limiter_param.min_density;
        real p_avg = 1e-13;

        if (nstate == dim + 2) {
            // Compute average value of pressure using soln_cell_avg
            p_avg = euler_physics->compute_pressure(soln_cell_avg);
        }
        
        // Obtain value used to linearly scale density
        real theta = get_density_scaling_value(soln_cell_avg[0], local_min_density, lower_bound, p_avg);

        // Apply limiter on density values at quadrature points
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            soln_coeff[0][ishape] = theta*(soln_coeff[0][ishape] - soln_cell_avg[0]) + soln_cell_avg[0];
        }

        // Interpolate new limited density to quadrature pts.
        soln_at_q_xGL[0].resize(n_quad_pts);
        soln_basis_xGL.matrix_vector_mult_1D(soln_coeff[0], soln_at_q_xGL[0],
                soln_basis_xGL.oneD_vol_operator);

        soln_at_q_yGL[0].resize(n_quad_pts);
        soln_basis_yGL.matrix_vector_mult_1D(soln_coeff[0], soln_at_q_yGL[0],
                soln_basis_yGL.oneD_vol_operator);
        
        std::array<real, nstate> soln_cell_avg_hat = get_soln_cell_avg_2D(soln_at_q_xGL, soln_at_q_yGL, n_quad_pts, quad_weights_xGL, quad_weights_yGL, dt);

        real p_avg_hat = 1e-13;

        if (nstate == dim + 2) {
            // Compute average value of pressure using soln_cell_avg
            p_avg_hat = euler_physics->compute_pressure(soln_cell_avg);
        }
        
        real theta2 = 0.0;
        using limiter_enum = Parameters::LimiterParam::LimiterType;
        limiter_enum limiter_type = this->all_parameters->limiter_param.bound_preserving_limiter;

        if (limiter_type == limiter_enum::positivity_preservingZhang2010 && nstate == dim + 2) {
            std::vector< real > p_lim_1(n_quad_pts);
            std::vector< real > p_lim_2(n_quad_pts);
            std::array<real, nstate> soln_at_iquad_1;
            std::array<real, nstate> soln_at_iquad_2;

            // Compute pressure at quadrature points
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    soln_at_iquad_1[istate] = soln_at_q_xGL[istate][iquad];
                    soln_at_iquad_2[istate] = soln_at_q_yGL[istate][iquad];
                }
                p_lim_1[iquad] = euler_physics->compute_pressure(soln_at_iquad_1);
                p_lim_2[iquad] = euler_physics->compute_pressure(soln_at_iquad_2);
            }

            // Obtain value used to linearly scale state variables
            std::vector<real> theta2_1 = get_theta2_Zhang2010(p_lim_1, soln_cell_avg_hat, soln_at_q_xGL, n_quad_pts, lower_bound, euler_physics->gam);
            std::vector<real> theta2_2 = get_theta2_Zhang2010(p_lim_1, soln_cell_avg_hat, soln_at_q_xGL, n_quad_pts, lower_bound, euler_physics->gam);

            // Limit values at quadrature points
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    if (theta2_1[iquad] <= theta2_2[iquad]) {
                        soln_coeff[istate][iquad] = theta2_1[iquad] * (soln_coeff[istate][iquad] - soln_cell_avg_hat[istate])
                            + soln_cell_avg_hat[istate];
                    } else {
                        soln_coeff[istate][iquad] = theta2_2[iquad] * (soln_coeff[istate][iquad] - soln_cell_avg_hat[istate])
                            + soln_cell_avg_hat[istate];
                    }
                }
            }
        }

        if (limiter_type == limiter_enum::positivity_preservingWang2012 && nstate == dim + 2) {
            real theta2_1 = get_theta2_Wang2012(soln_at_q_xGL, n_quad_pts, p_avg_hat); // Value used to linearly scale state variables 
            real theta2_2 = get_theta2_Wang2012(soln_at_q_yGL, n_quad_pts, p_avg_hat); // Value used to linearly scale state variables 

            theta2 = std::min(theta2_1, theta2_2);

            // Limit values at quadrature points
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    soln_coeff[istate][iquad] = theta2 * (soln_coeff[istate][iquad] - soln_cell_avg_hat[istate])
                        + soln_cell_avg_hat[istate];
                }
            }
        }

        if (limiter_type == limiter_enum::positivity_preservingZhang2016 && nstate == dim + 2) {
            real theta2_1 = get_theta2_Zhang2016(soln_at_q_xGL, n_quad_pts, p_avg_hat, lower_bound); // Value used to linearly scale state variables 
            real theta2_2 = get_theta2_Zhang2016(soln_at_q_yGL, n_quad_pts, p_avg_hat, lower_bound); // Value used to linearly scale state variables 

            theta2 = std::min(theta2_1, theta2_2);

            // Limit values at quadrature points
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    soln_coeff[istate][iquad] = theta2 * (soln_coeff[istate][iquad] - soln_cell_avg_hat[istate])
                        + soln_cell_avg_hat[istate];
                }
            }
        }

        // Write limited solution back and verify that positivity of density is satisfied
        write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);        
    }
}

template class PositivityPreservingLimiter <PHILIP_DIM, PHILIP_DIM+2, double>;
} // PHiLiP namespace