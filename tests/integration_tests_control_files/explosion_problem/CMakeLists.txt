set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 3D Sod Shock Tube test
# =======================================
configure_file(3D_explosion_problem.prm 3D_explosion_problem.prm COPYONLY)
add_test(
  NAME 3D_EXPLOSION_PROBLEM
  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_explosion_problem.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(3D_EXPLOSION_PROBLEM   EXPLOSION_PROBLEM
                                        3D
                                        PARALLEL
                                        EULER
                                        RUNGE-KUTTA
                                        STRONG-SPLIT
                                        COLLOCATED
                                        LIMITER
                                        INTEGRATION_TEST)

# =======================================
# 2D Sod Shock Tube test
# =======================================
configure_file(2D_explosion_problem.prm 2D_explosion_problem.prm COPYONLY)
add_test(
  NAME 2D_EXPLOSION_PROBLEM
  COMMAND mpirun -n 8 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_explosion_problem.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_EXPLOSION_PROBLEM   EXPLOSION_PROBLEM
                                        2D
                                        PARALLEL
                                        EULER
                                        RUNGE-KUTTA
                                        STRONG-SPLIT
                                        COLLOCATED
                                        LIMITER
                                        INTEGRATION_TEST)
