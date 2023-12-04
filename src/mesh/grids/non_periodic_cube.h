#ifndef __NON_PERIODIC_CUBE_H__
#define __NON_PERIODIC_CUBE_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP::Grids {

/// Create a nonperiodic cube mesh
/// Boundary IDs are assigned for limiter_convergence_tests
/// Unassigned otherwise
template<int dim, typename TriangulationType>
void non_periodic_cube(
    TriangulationType&  grid,
    double              domain_left,
    double              domain_right,
    bool                colorize,
    const int           left_boundary_id,
    const int           n_subdivisions_0,
    const int           n_subdivisions_1);
} // namespace PHiLiP::Grids
#endif