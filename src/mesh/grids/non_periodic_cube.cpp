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
    dealii::Point<2> p1(0.0, 0.0), p2(1.0, 0.25);
    //std::vector<unsigned int> n_subdivisions(2);
    //n_subdivisions[0] = 128;
    //n_subdivisions[1] = 32;

    if(dim==1)
        dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);
    else if (dim==2)
        dealii::GridGenerator::hyper_rectangle(grid, p1, p2, colorize);

    if (left_boundary_id != 9999 && dim == 1) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
        }
    }
    else {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            cell->set_material_id(9002);
            for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary()) {
                    cell->face(face)->set_boundary_id(1001);
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
