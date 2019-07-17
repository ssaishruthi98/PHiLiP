#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "tests.h"
#include "euler_vortex.h"

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

//#include "template_instantiator.h"


namespace PHiLiP {
namespace Tests {

template <int dim, typename real>
EulerVortexFunction<dim,real>
::EulerVortexFunction (
        const Physics::Euler<dim, dim+2, real> euler_physics,
        const dealii::Point<2> initial_vortex_center,
        const real vortex_strength,
        const real vortex_decay)
    :
    dealii::Function<dim,real>(dim+2, 0.0)
    , euler_physics(euler_physics)
    , initial_vortex_center(initial_vortex_center)
    , vortex_strength(vortex_strength)
    , vortex_decay(vortex_decay)
{ 
    static_assert(dim==2);
}

template <int dim, typename real>
inline dealii::Point<2> EulerVortexFunction<dim,real>::advected_location(dealii::Point<2> old_location) const
{
    dealii::Point<2> new_location;
    for(int d=0; d<dim; d++) {
        new_location[d] = (old_location[d] - initial_vortex_center[d]) - euler_physics.velocities_inf[d] * this->time;
    }
    return new_location;
}

template <int dim, typename real>
inline real EulerVortexFunction<dim,real>
::value (const dealii::Point<dim> &point, const unsigned int istate) const
{
    dealii::Point<2> new_loc = advected_location(point);

    const double pi = dealii::numbers::PI;
    const double local_radius_sqr = new_loc.square();
    const double x = new_loc[0], y = new_loc[1];
    
    const double sound_inf = euler_physics.sound_inf;

    double vel_x = euler_physics.velocities_inf[0] / sound_inf - vortex_strength / (2.0*pi*sound_inf) * y * exp(0.5*vortex_decay*(1.0-local_radius_sqr));
    double vel_y = euler_physics.velocities_inf[1] / sound_inf - vortex_strength / (2.0*pi*sound_inf) * x * exp(0.5*vortex_decay*(1.0-local_radius_sqr));
    double temperature = 1.0 - vortex_strength*vortex_strength*euler_physics.gamm1 / (8.0*vortex_decay*pi*pi*sound_inf*sound_inf) * exp(vortex_decay*(1.0-local_radius_sqr));

    const double value = vel_x+vel_y+temperature;
    return value;
}

template <int dim, int nstate>
EulerVortex<dim,nstate>::EulerVortex(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
void EulerVortex<dim,nstate>
::initialize_perturbed_solution(DGBase<dim,double> &dg, const Physics::PhysicsBase<dim,nstate,double> &physics) const
{
    dealii::VectorTools::interpolate(dg.dof_handler, physics.manufactured_solution_function, dg.solution);
}

template<int dim, int nstate>
int EulerVortex<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int initial_grid_size   = manu_grid_conv_param.initial_grid_size;
    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;
    const double       grid_progression    = manu_grid_conv_param.grid_progression;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&param);

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input + 2;

        std::vector<int> n_1d_cells(n_grids);
        n_1d_cells[0] = initial_grid_size;
        if(poly_degree==0) n_1d_cells[0] = initial_grid_size + 1;

        std::vector<double> soln_error(n_grids);
        std::vector<double> grid_size(n_grids);

        for (unsigned int igrid=1;igrid<n_grids;++igrid) {
            n_1d_cells[igrid] = n_1d_cells[igrid-1]*grid_progression;
        }

        dealii::ConvergenceTable convergence_table;

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
            // Note that Triangulation must be declared before DG
            // DG will be destructed before Triangulation
            // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
            // Otherwise, a Subscriptor error will occur
            dealii::Triangulation<dim> grid;

            // Generate hypercube
            if ( manu_grid_conv_param.grid_type == GridEnum::hypercube || manu_grid_conv_param.grid_type == GridEnum::sinehypercube ) {

                dealii::GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid]);
                for (typename dealii::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
                    // Set a dummy boundary ID
                    cell->set_material_id(9002);
                    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
                    }
                }
            }

            // Distort grid by random amount if requested
            const double random_factor = manu_grid_conv_param.random_distortion;
            const bool keep_boundary = true;
            if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, grid, keep_boundary);

            // Read grid if requested
            if (manu_grid_conv_param.grid_type == GridEnum::read_grid) {
                //std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                std::string read_mshname = manu_grid_conv_param.input_grids+std::to_string(igrid)+".msh";
                std::cout<<"Reading grid: " << read_mshname << std::endl;
                std::ifstream inmesh(read_mshname);
                dealii::GridIn<dim,dim> grid_in;
                grid_in.attach_triangulation(grid);
                grid_in.read_msh(inmesh);
            }
            // Output grid if requested
            if (manu_grid_conv_param.output_meshes) {
                std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                std::ofstream outmesh(write_mshname);
                dealii::GridOutFlags::Msh msh_flags(true, true);
                dealii::GridOut grid_out;
                grid_out.set_flags(msh_flags);
                grid_out.write_msh(grid, outmesh);
            }

            // Show mesh if in 2D
            std::string gridname = "grid-"+std::to_string(igrid)+".eps";

            using ADtype = Sacado::Fad::DFad<double>;

            // Create DG object using the factory
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree);
            dg->set_triangulation(&grid);
            dg->allocate_system ();

            initialize_perturbed_solution(*(dg), *(physics_double));

            // Create ODE solver using the factory and providing the DG object
            std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            unsigned int n_active_cells = grid.n_active_cells();
            std::cout
                      << "Dimension: " << dim
                      << "\t Polynomial degree p: " << poly_degree
                      << std::endl
                      << "Grid number: " << igrid+1 << "/" << n_grids
                      << ". Number of active cells: " << n_active_cells
                      << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
                      << std::endl;

            // Solve the steady state problem
            ode_solver->steady_state();

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->fe_system.tensor_degree()+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(dg->mapping, dg->fe_system, quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0;

            // Integrate solution error
            typename dealii::DoFHandler<dim>::active_cell_iterator
               cell = dg->dof_handler.begin_active(),
               endc = dg->dof_handler.end();

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            for (; cell!=endc; ++cell) {

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    //std::array<double,nstate> uexact;
                    //std::cout << "cos(0.59*x+1 " << cos(0.59*qpoint[0]+1) << std::endl;
                    //std::cout << "uexact[1] " << uexact[1] << std::endl;

                    for (int istate=0; istate<nstate; ++istate) {
                        const double uexact = physics_double->manufactured_solution_function.value(qpoint, istate);
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                    }
                }

            }
            l2error = sqrt(l2error);

            // Convergence table
            double dx = 1.0/pow(n_active_cells,(1.0/dim));
            dx = dealii::GridTools::maximal_cell_diameter(grid);
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", grid.n_active_cells());
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error);


            std::cout   << " Grid size h: " << dx 
                        << " L2-soln_error: " << l2error
                        << " Residual: " << ode_solver->residual_norm
                        << std::endl;

            if (igrid > 0) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                std::cout << "From grid " << igrid-1
                          << "  to grid " << igrid
                          << "  dimension: " << dim
                          << "  polynomial degree p: " << dg->fe_system.tensor_degree()
                          << std::endl
                          << "  solution_error1 " << soln_error[igrid-1]
                          << "  solution_error2 " << soln_error[igrid]
                          << "  slope " << slope_soln_err
                          << std::endl;
            }
        }
        std::cout
            << " ********************************************"
            << std::endl
            << " Convergence rates for p = " << poly_degree
            << std::endl
            << " ********************************************"
            << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.write_text(std::cout);

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        const double last_slope = log(soln_error[n_grids-1]/soln_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        double before_last_slope = last_slope;
        if ( n_grids > 2 ) {
        before_last_slope = log(soln_error[n_grids-2]/soln_error[n_grids-3])
                            / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        }
        const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -0.1;
        if(poly_degree == 0) slope_deficit_tolerance = -0.2; // Otherwise, grid sizes need to be much bigger for p=0

        if (slope_diff < slope_deficit_tolerance) {
            std::cout << std::endl
                      << "Convergence order not achieved. Average last 2 slopes of "
                      << slope_avg << " instead of expected "
                      << expected_slope << " within a tolerance of "
                      << slope_deficit_tolerance
                      << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    std::cout << std::endl
              << std::endl
              << std::endl
              << std::endl;
    std::cout << " ********************************************"
              << std::endl;
    std::cout << " Convergence summary"
              << std::endl;
    std::cout << " ********************************************"
              << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        conv->write_text(std::cout);
        std::cout << " ********************************************"
                  << std::endl;
    }
    int n_fail_poly = fail_conv_poly.size();
    if (n_fail_poly > 0) {
        for (int ifail=0; ifail < n_fail_poly; ++ifail) {
            const double expected_slope = fail_conv_poly[ifail]+1;
            const double slope_deficit_tolerance = -0.1;
            std::cout << std::endl
                      << "Convergence order not achieved for polynomial p = "
                      << fail_conv_poly[ifail]
                      << ". Slope of "
                      << fail_conv_slop[ifail] << " instead of expected "
                      << expected_slope << " within a tolerance of "
                      << slope_deficit_tolerance
                      << std::endl;
        }
    }
    return n_fail_poly;
}

#if PHILIP_DIM==2
    template class EulerVortex <PHILIP_DIM,PHILIP_DIM+2>;
#endif


} // Tests namespace
} // PHiLiP namespace
