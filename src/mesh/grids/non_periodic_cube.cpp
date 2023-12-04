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
    const int           left_boundary_id,
    const int           n_subdivisions_0,
    const int           n_subdivisions_1) 
{
    // dealii::Point<2> p1(0.0, 0.0), p2(4.0, 3.0);
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    if (dim >= 1) {
        p1[0] = 0.0;
        p2[0] = 3.0;
    } 

    if(dim == 2) {
        p1[1] = 0.0;
        p2[1] = 1.0;
    }
    std::vector<unsigned int> n_subdivisions(2);
    n_subdivisions[0] = n_subdivisions_0;//log2(128);
    n_subdivisions[1] = n_subdivisions_1;//log2(64);
    
    std::vector<int> n_cells_remove(2);
    n_cells_remove[0] = (-2.4/3.0)*n_subdivisions[0];
    n_cells_remove[1] = (0.2/1.0)*n_subdivisions[1];

    if (dim == 1)
        dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);
    else if (PHILIP_DIM == 2)
        dealii::GridGenerator::subdivided_hyper_L(grid, n_subdivisions, p1, p2, n_cells_remove);

    if (left_boundary_id != 9999 && dim == 1) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
        }
    }
    else {
        // Set boundary type and design type
        double right_y = 0.0;
        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
            for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    if (face == 0) {
                        cell->face(face)->set_boundary_id(1007); // x_left, Farfield
                    }
                    else if (face == 1) {
                        if (right_y < 0.2) {
                            right_y += cell->extent_in_direction(1);
                            cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                        }
                        else {
                            cell->face(face)->set_boundary_id(1008); // x_right, Symmetry/Wall
                        }
                    }
                    else if (face == 2 || face == 3) {
                            cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                    }
                }
            }
        }
        sleep(5);
    }
}

#if PHILIP_DIM==1
template void non_periodic_cube<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>&   grid,
    double                      domain_left,
    double                      domain_right,
    bool                        colorize,
    const int                   left_boundary_id,
    const int                   n_subdivisions_0,
    const int                   n_subdivisions_1);
#else
template void non_periodic_cube<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    double                                              domain_left,
    double                                              domain_right,
    bool                                                colorize,
    const int                                           left_boundary_id,
    const int                                           n_subdivisions_0,
    const int                                           n_subdivisions_1);
#endif
} // namespace PHiLiP::Grids