#ifndef __MULTISPECIES_TESTS_H__
#define __MULTISPECIES_TESTS_H__

#include "flow_solver_case_base.h"
#include "cube_flow_uniform_grid.h"
#include "physics/navier_stokes_real_gas.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nspecies, int nstate>
class MultispeciesTests : public CubeFlow_UniformGrid<dim,nspecies,nstate>
{
    /** Number of different computed quantities
     *  Corresponds to the number of items in IntegratedQuantitiesEnum
     * */
    static const int NUMBER_OF_INTEGRATED_QUANTITIES = 6;

public:
    /// Constructor.
    explicit MultispeciesTests(const Parameters::AllParameters *const parameters_input);

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /** Computes the integrated quantities over the domain simultaneously and updates the array storing them
     *  Note: For efficiency, this also simultaneously updates the local maximum wave speed
     * */
    void compute_and_update_integrated_quantities(DGBase<dim, nspecies, double> &dg);

protected:
    /// List of possible integrated quantities over the domain
    enum IntegratedQuantitiesEnum {
        kinetic_energy,
        enstrophy,
        pressure_dilatation,
        incompressible_kinetic_energy,
        incompressible_enstrophy,
        incompressible_palinstrophy
    };
    /// Array for storing the integrated quantities; done for computational efficiency
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrated_quantities;

    /// Integrated kinetic energy over the domain at previous time step; used for ensuring a physically consistent simulation
    double integrated_kinetic_energy_at_previous_time_step;

    /// Pointer to Navier-Stokes Real Gas physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes_RealGas<dim,nspecies,dim+nspecies+1,double> > ns_real_gas_physics;

    /// Function to compute the adaptive time step
    using CubeFlow_UniformGrid<dim, nspecies, nstate>::get_adaptive_time_step;

    /// Function to compute the initial adaptive time step
    using CubeFlow_UniformGrid<dim, nspecies, nstate>::get_adaptive_time_step_initial;

    /// Updates the maximum local wave speed
    using CubeFlow_UniformGrid<dim, nspecies, nstate>::update_maximum_local_wave_speed;

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    using FlowSolverCaseBase<dim, nspecies, nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
        const std::shared_ptr<ODE::ODESolverBase<dim, nspecies, double>> ode_solver,
        const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

    /** Gets the nondimensional integrated kinetic energy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_kinetic_energy() const;

    /** Gets the nondimensional integrated enstrophy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_enstrophy() const;

    /// Gets the nondimensional integrated incompressible kinetic energy given a DG object from dg->solution
    double get_integrated_incompressible_kinetic_energy() const;

    /// Gets the nondimensional integrated incompressible enstrophy given a DG object from dg->solution
    double get_integrated_incompressible_enstrophy() const;

    /// Gets the nondimensional integrated incompressible palinstrophy given a DG object from dg->solution
    double get_integrated_incompressible_palinstrophy() const;

    /** Gets non-dimensional theoretical vorticity tensor based dissipation rate 
     *  Note: For incompressible flows or when dilatation effects are negligible 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_vorticity_based_dissipation_rate() const;

    /** Evaluate non-dimensional theoretical pressure-dilatation dissipation rate
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_pressure_dilatation_based_dissipation_rate () const;

    const unsigned int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_size; ///< Domain size (length in 1D, area in 2D, and volume in 3D)

    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.
    bool is_viscous_flow = true; ///< Identifies if viscous flow; initialized as true.


    /// Display additional more specific flow case parameters
    virtual void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const;
    
private:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
