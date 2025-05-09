configure_file(2d_euler_naca0012_subsonic_05_200.prm 2d_euler_naca0012_subsonic_05_200.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG    EULER_INTEGRATION
                                                                NACA0012
                                                                2D
                                                                PARALLEL
                                                                EULER
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                MANUFACTURED_SOLUTION
                                                                CONVERGENCE
                                                                LONG
                                                                INTEGRATION_TEST)

configure_file(2d_euler_naca0012_transonic_08_125.prm 2d_euler_naca0012_transonic_08_125.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_TRANSONIC_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_transonic_08_125.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_EULER_INTEGRATION_NACA0012_TRANSONIC_LONG   EULER_INTEGRATION
                                                                NACA0012
                                                                2D
                                                                PARALLEL
                                                                EULER
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                MANUFACTURED_SOLUTION
                                                                CONVERGENCE
                                                                LONG
                                                                INTEGRATION_TEST)

configure_file(inviscid_transonic_steady_state_naca0012.prm inviscid_transonic_steady_state_naca0012.prm COPYONLY)
add_test(
        NAME INVISCID_TRANSONIC_STEADY_STATE_NACA0012
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_transonic_steady_state_naca0012.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(INVISCID_TRANSONIC_STEADY_STATE_NACA0012   EULER_INTEGRATION
                                                            NACA0012
                                                            2D
                                                            PARALLEL
                                                            EULER
                                                            IMPLICIT
                                                            WEAK
                                                            UNCOLLOCATED
                                                            QUICK
                                                            INTEGRATION_TEST)

configure_file(2d_euler_transonic_naca0012_drag_based_grid_adaptation.prm 2d_euler_transonic_naca0012_drag_based_grid_adaptation.prm COPYONLY)
add_test(
        NAME INVISCID_TRANSONIC_STEADY_STATE_NACA0012_DRAG_BASED_GRID_ADAPTATION
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_transonic_naca0012_drag_based_grid_adaptation.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(INVISCID_TRANSONIC_STEADY_STATE_NACA0012_DRAG_BASED_GRID_ADAPTATION    EULER_INTEGRATION
                                                                                        NACA0012
                                                                                        2D
                                                                                        PARALLEL
                                                                                        EULER
                                                                                        IMPLICIT
                                                                                        WEAK
                                                                                        UNCOLLOCATED
                                                                                        MESH_ADAPTATION
                                                                                        ADJOINT
                                                                                        MODERATE
                                                                                        INTEGRATION_TEST)

configure_file(2d_euler_subsonic_naca0012_lift_based_hp_adaptation.prm  2d_euler_subsonic_naca0012_lift_based_hp_adaptation.prm  COPYONLY)
add_test(
        NAME INVISCID_SUBSONIC_NACA0012_LIFT_BASED_HP_ADAPTATION
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_subsonic_naca0012_lift_based_hp_adaptation.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(INVISCID_SUBSONIC_NACA0012_LIFT_BASED_HP_ADAPTATION    EULER_INTEGRATION
                                                                        NACA0012
                                                                        2D
                                                                        PARALLEL
                                                                        EULER
                                                                        IMPLICIT
                                                                        WEAK
                                                                        UNCOLLOCATED
                                                                        MESH_ADAPTATION
                                                                        ADJOINT
                                                                        MODERATE
                                                                        INTEGRATION_TEST)

configure_file(naca0012_unsteady.prm naca0012_unsteady.prm COPYONLY)
add_test(
    NAME 2D_EULER_INTEGRATION_UNSTEADY_NACA0012_SUBSONIC_STRONGvWEAK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/naca0012_unsteady.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_EULER_INTEGRATION_UNSTEADY_NACA0012_SUBSONIC_STRONGvWEAK    EULER_INTEGRATION
                                                                                NACA0012
                                                                                2D
                                                                                PARALLEL
                                                                                EULER
                                                                                RUNGE-KUTTA
                                                                                WEAK STRONG
                                                                                UNCOLLOCATED
                                                                                MODERATE
                                                                                INTEGRATION_TEST)

#configure_file(2d_navier_stokes_naca0012_subsonic_05_200.prm 2d_navier_stokes_naca0012_subsonic_05_200.prm COPYONLY)
#add_test(
#  NAME 2D_NAVIER_STOKES_INTEGRATION_NACA0012_SUBSONIC_LONG
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i #${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_naca0012_subsonic_05_200.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

#configure_file(naca0012.geo naca0012.geo COPYONLY)
#configure_file(naca0012_noTE.geo naca0012_noTE.geo COPYONLY)
#set(TAR_NAME naca0012_airfoil_Q3)
#add_custom_target(${TAR_NAME}
#    #WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski
#    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#    COMMAND gmsh naca0012_noTE.geo -
#    DEPENDS naca0012_noTE.geo
#    BYPRODUCTS naca0012.msh
#    #COMMAND mv
#    #     "${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski/new_msh41.msh"
#    #     "${CMAKE_BINARY_DIR}/tests/integration_tests_control_files/euler_integration/naca0012/joukowski_R${ref}_Q${order}.msh"
#)
#add_dependencies(grids ${TAR_NAME})
#add_dependencies(2D ${TAR_NAME})
