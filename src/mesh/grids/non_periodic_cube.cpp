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
    // dealii::Point<2> p1(0.0, 0.0), p2(4.0, 3.0);
    dealii::Point<dim> p1, p2, p3, p4;
    if (dim >= 1) {
        p1[0] = 0.0;
        p2[0] = 4.0;
        p3[0] = 0.0;
        p4[0] = 0.6;
    }

    if (dim == 2) {
        p1[1] = 0.2;
        p2[1] = 1.0;
        p3[1] = 0.0;
        p4[1] = 0.2;

    }
    std::vector<unsigned int> n_subdivisions1(2);
    std::vector<unsigned int> n_subdivisions2(2);
    n_subdivisions1[0] = 480;
    n_subdivisions1[1] = 96;
    n_subdivisions1[0] = 288;
    n_subdivisions1[1] = 24;

    if (dim == 2) {
        std::shared_ptr<Triangulation> grid1 = std::make_shared<Triangulation>();
        std::shared_ptr<Triangulation> grid2 = std::make_shared<Triangulation>();

        dealii::GridGenerator::subdivided_hyper_rectangle(*grid1, n_subdivisions1, p1, p2, true);
        dealii::GridGenerator::subdivided_hyper_rectangle(*grid2, n_subdivisions2, p3, p4, true);

        double bottom_x = 0.0;

        // Set boundary type and design type
        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid1->begin_active(); cell != grid1->end(); ++cell) {
            for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 0) {
                        cell->face(face)->set_boundary_id(1007); // x_left, Farfield
                    }
                    else if (current_id == 1) {
                        cell->face(face)->set_boundary_id(1008); // x_right, Symmetry/Wall
                    }
                    else if (current_id == 2) {
                        if (bottom_x < 0.6) {
                            bottom_x += cell->extent_in_direction(0);
                            cell->face(face)->set_boundary_id(9999); // y_bottom, Symmetry/Wall
                        }
                        else {
                            cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                        }
                    }
                    else if (current_id == 3) {
                        cell->face(face)->set_boundary_id(1001);
                    }
                }
            }
        }

        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid2->begin_active(); cell != grid2->end(); ++cell) {
            for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 0) {
                        cell->face(face)->set_boundary_id(1007); // x_left, Farfield
                    }
                    else if (current_id == 1) {
                        cell->face(face)->set_boundary_id(1001); // x_right, Symmetry/Wall
                    }
                    else if (current_id == 2) {
                        cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                    }
                    else if (current_id == 3) {
                        cell->face(face)->set_boundary_id(9999);
                    }
                }
            }
        }
        const Triangulation& topGrid= dealii::Triangulation<2>::copy_triangulation(grid1);
        const Triangulation& bottomGrid = dealii::Triangulation<2>::copy_triangulation(grid2);
        dealii::GridGenerator::merge_triangulations(topGrid, bottomGrid, grid, 1e-12, true);
    }

    if (dim == 1)
        dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);

    if (left_boundary_id != 9999 && dim == 1) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
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
