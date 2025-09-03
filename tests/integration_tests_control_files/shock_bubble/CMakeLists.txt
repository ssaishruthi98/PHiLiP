set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Strong Vortex Shock Wave Case
# =======================================
configure_file(2D_shock_bubble.prm 2D_shock_bubble.prm COPYONLY)
add_test(
  NAME 2D_SHOCK_BUBBLE_INTERACTION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_shock_bubble.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_SHOCK_BUBBLE_INTERACTION 2D_SHOCK_BUBBLE_INTERACTION
                                            2D
                                            PARALLEL
                                            EULER
                                            RUNGE-KUTTA
                                            STRONG
                                            COLLOCATED
                                            LIMITER
                                            MODERATE
                                            INTEGRATION_TEST)
