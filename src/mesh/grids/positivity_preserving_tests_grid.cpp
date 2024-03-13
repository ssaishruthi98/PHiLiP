#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "positivity_preserving_tests_grid.h"

namespace PHiLiP::Grids {
template<int dim, typename TriangulationType>
void shock_tube_1D_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input)
{
    double domain_left = parameters_input->flow_solver_param.grid_left_bound;
    double domain_right = parameters_input->flow_solver_param.grid_right_bound;

    dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, true);

    int left_boundary_id = 9999;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = parameters_input->flow_solver_param.flow_case_type;

    if (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube) {
        left_boundary_id = 1001;
    } else if (flow_case_type == flow_case_enum::shu_osher_problem) {
        left_boundary_id = 1004;
    } 

    if (left_boundary_id != 9999 && dim == 1) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
        }
    }

    const unsigned int number_of_refinements = parameters_input->flow_solver_param.number_of_mesh_refinements;
    grid.refine_global(number_of_refinements);
}

template<int dim, typename TriangulationType>
void double_mach_reflection_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double domain_left = parameters_input->flow_solver_param.grid_left_bound;
    double domain_right = parameters_input->flow_solver_param.grid_right_bound;
    double domain_bottom = parameters_input->flow_solver_param.grid_bottom_bound;
    double domain_top = parameters_input->flow_solver_param.grid_top_bound;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = domain_left; p1[1] = domain_bottom;
    p2[0] = domain_right; p2[1] = domain_top;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);


    double bottom_x = 0.0;

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    cell->face(face)->set_boundary_id(1007); // x_left, post-shock
                }
                else if (current_id == 1) {
                    cell->face(face)->set_boundary_id(1004); // x_right, riemann
                }
                else if (current_id == 2) {
                    if (bottom_x < (1.0 / 6.0)) {
                        //std::cout << "assigning post shock " << bottom_x << std::endl;
                        bottom_x += cell->extent_in_direction(0);
                        cell->face(face)->set_boundary_id(1007); // y_bottom, post-shock
                    }
                    else {
                        cell->face(face)->set_boundary_id(1001); // y_bottom, wall
                    }
                }
                else if (current_id == 3) {
                    cell->face(face)->set_boundary_id(1001);
                }
            }
        }
    } 
}

template<int dim, typename TriangulationType>
void sedov_blast_wave_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double domain_left = parameters_input->flow_solver_param.grid_left_bound;
    double domain_right = parameters_input->flow_solver_param.grid_right_bound;
    double domain_bottom = parameters_input->flow_solver_param.grid_bottom_bound;
    double domain_top = parameters_input->flow_solver_param.grid_top_bound;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = domain_left; p1[1] = domain_bottom;
    p2[0] = domain_right; p2[1] = domain_top;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                cell->face(face)->set_boundary_id(1001);
            }
        }
    }
}

template<int dim, typename TriangulationType>
void mach_3_wind_tunnel_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double domain_left = parameters_input->flow_solver_param.grid_left_bound;
    double domain_right = parameters_input->flow_solver_param.grid_right_bound;
    double domain_bottom = parameters_input->flow_solver_param.grid_bottom_bound;
    double domain_top = parameters_input->flow_solver_param.grid_top_bound;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = domain_left; p1[1] = domain_bottom;
    p2[0] = domain_right; p2[1] = domain_top;
    
    std::vector<unsigned int> n_subdivisions(2);
    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

    std::vector<int> n_cells_remove(2);
    n_cells_remove[0] = (-2.4/3.0)*n_subdivisions[0] - 1;
    n_cells_remove[1] = (0.2/1.0)*n_subdivisions[1];

    dealii::GridGenerator::subdivided_hyper_L(grid, n_subdivisions, p1, p2, n_cells_remove);

    // Set boundary type and design type
    double right_y = 0.0;
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                if (face == 0) {
                    cell->face(face)->set_boundary_id(1007); // x_left, Inflow
                }
                else if (face == 1) {
                    if (right_y < 0.2) {
                        right_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1001); // x_right, Symmetry/Wall 
                    }
                    else {
                        cell->face(face)->set_boundary_id(1002); // x_right, Outflow
                    }
                }
                else if (face == 2 || face == 3) {
                        cell->face(face)->set_boundary_id(1001); // y_top, y_bottom, Symmetry/Wall
                }
            }
        }
    }
}

#if PHILIP_DIM==1
template void shock_tube_1D_grid<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>&   grid,
    const Parameters::AllParameters *const parameters_input);
#else
template void double_mach_reflection_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
template void sedov_blast_wave_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
template void mach_3_wind_tunnel_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
#endif
} // namespace PHiLiP::Grids