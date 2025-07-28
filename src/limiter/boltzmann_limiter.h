#ifndef __BOLTZMANN_LIMITER__
#define __BOLTZMANN_LIMITER__

#include "bound_preserving_limiter.h"
#include "positivity_preserving_limiter.h"

namespace PHiLiP {
/// Class for implementation of two forms of the Maximum-Principle limiter using bounds derived from the Boltzmann Equation
/// derived from PositivityPreservingLimiter class
/**********************************
* Dzanic and Martinelli. 
* "High-order limiting methods using maximum principle bounds 
*  derived from the Boltzmann equation I: Euler equations" 
* Journal of Computational Physics (jcp.2025.113895)
**********************************/
template<int dim, int nstate, typename real>
class BoltzmannLimiter : public PositivityPreservingLimiter <dim, nstate, real>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using DoFHandlerType = dealii::DoFHandler<dim>; ///< Alias for declaring DofHandler    
public:
    /// Constructor
    explicit BoltzmannLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~BoltzmannLimiter() = default;

    /// Flow solver parameters
    const Parameters::FlowSolverParam flow_solver_param; 

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiterState<dim, nstate, real>> tvbLimiter;

    /// Euler physics pointer. Used to compute pressure.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

    /// Function to obtain the solution cell average for one dimension
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Function to obtain the solution cell average for two or more dimensions
    using PositivityPreservingLimiter<dim, nstate, real>::get_soln_cell_avg_PPL;

    /// Function to obtain scaling value based on pressure
    using PositivityPreservingLimiter<dim, nstate, real>::get_theta2_Wang2012;

    /// FILL THIS OUT LATER !!!!!!!!!!!!!!!!!!!!!!!!!!
    void limit(
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
        const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>>            mapping_field) override;

protected:

    /// Obtain the microscopic velocity domain using the min-max strategy over the stencil of the cell
    /// Using 3.7 from Dzanic, Martinelli 2025
    std::vector<real> get_integrating_domain(
        const std::array<std::vector<real>, nstate>&    soln_at_q,
        const unsigned int                              n_quad_pts,
        const double                                    k);

    /// Obtain the Boltzmann distribution of microscopic velocities
    /// Using 2.2 from Dzanic, Martinelli 2025
    std::vector< std::vector<real>> get_boltzmann_distribution(
    const std::array<std::vector<real>, nstate>&                                                soln_at_q_dim,
    const unsigned int                                                                          n_quad_pts,
    const double                                                                                resolution,
    const double                                                                                lower_distribution_limit,
    const double                                                                                upper_distribution_limit,
    const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>>            mapping_field,
    dealii::QGaussLobatto<dim>                                                                  quad_for_l2_norm,
    const dealii::hp::FECollection<dim>&                                                        fe_collection,
    const int                                                                                   poly_degree);

    /// Use the Boltzmann distribution to obtain the macroscopic maxima and minima for density, momentum, and energy
    std::vector< std::vector<real>> boltzmann_limits(
    const std::vector<real>&            u_values,
    const std::vector<real>&            f_max_values,
    const std::vector<real>&            f_min_values);


    /// Using boltzman-distribution-derived limiting state vectors and cell-average values to obtain density scaling value which enforces limits
    /// Using 3.4 from Dzanicm, Martinelli 2025
    real get_alpha(
    const std::array<std::vector<real>, nstate>&    soln_at_q_dim,
    const unsigned int                              n_quad_pts,
    const std::array<real, nstate>&                 soln_cell_avg,
    const std::vector<real>&                        soln_cell_min,
    const std::vector<real>&                        soln_cell_max);

    /// Function to verify the limited solution preserves positivity of density and pressure
    /// and write back limited solution
    void write_limited_solution(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const std::array<std::vector<real>, nstate>&            soln_coeff,
        const unsigned int                                      n_shape_fns,
        const std::vector<dealii::types::global_dof_index>&     current_dofs_indices);

    // Values required to compute solution cell average in 2D/3D
    real dx; ///< Value required to compute solution cell average in 2D/3D, calculated using xmax and xmin parameters
    real dy; ///< Value required to compute solution cell average in 2D/3D, calculated using ymax and ymin parameters
    real dz; ///< Value required to compute solution cell average in 2D/3D, calculated using zmax and zmin parameters

    // Store the maximum and minimum bounds computed for the states to be applied at the next time step
    std::vector<real> state_max;
    std::vector<real> state_min;

    // Resolution for the trapezoidal rule pulled from flow_solver_param
    real resolution;


    bool first_run;
}; // End of PositivityPreservingLimiter Class
} // PHiLiP namespace

#endif

