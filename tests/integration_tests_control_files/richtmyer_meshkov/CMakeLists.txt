set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Strong Vortex Shock Wave Case
# =======================================
configure_file(2D_richtmyer_meshkov.prm 2D_richtmyer_meshkov.prm COPYONLY)
add_test(
  NAME 2D_RICHTMYER_MESHKOV
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_richtmyer_meshkov.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_RICHTMYER_MESHKOV RICHTMYER_MESHKOV
                                            2D
                                            PARALLEL
                                            EULER
                                            RUNGE-KUTTA
                                            STRONG
                                            COLLOCATED
                                            LIMITER
                                            MODERATE
                                            INTEGRATION_TEST)
