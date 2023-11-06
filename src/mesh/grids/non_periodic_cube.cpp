#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "non_periodic_cube.h"

namespace PHiLiP::Grids {

template<int dim, typename TriangulationType>
void non_periodic_cube(
    TriangulationType&  grid,
    double              domain_left,
    double              domain_right,
    bool                colorize,
    const int           left_boundary_id) 
{
    dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);

    if (left_boundary_id != 9999 && dim == 1) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
        }
    }

    if (dim == 2){
        // Set boundary type and design type
        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 0 || current_id == 2) cell->face(face)->set_boundary_id (1001); // Bottom and left wall
                    if (current_id == 1 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Outflow with supersonic or back_pressure
                }
            }
        }
    }
}

#if PHILIP_DIM==1
template void non_periodic_cube<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>&   grid,
    double                      domain_left,
    double                      domain_right,
    bool                        colorize,
    const int                   left_boundary_id);
#else
template void non_periodic_cube<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    double                                              domain_left,
    double                                              domain_right,
    bool                                                colorize,
    const int                                           left_boundary_id);
#endif
} // namespace PHiLiP::Grids
