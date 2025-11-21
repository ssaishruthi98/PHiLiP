set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Riemann Problem
# =======================================
 configure_file(2D_riemann_problem.prm 2D_riemann_problem.prm COPYONLY)
 add_test(
   NAME 2D_RIEMANN_PROBLEM
   COMMAND mpirun -np 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_riemann_problem.prm
   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
 )
set_tests_labels(2D_RIEMANN_PROBLEM RIEMANN_PROBLEM
                                    2D
                                    PARALLEL
                                    EULER
                                    RUNGE-KUTTA
                                    STRONG
                                    COLLOCATED
                                    LIMITER
                                    EXTRA-LONG
                                    INTEGRATION_TEST)
