set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Leblanc Shock Tube test
# =======================================

configure_file(1D_blast_wave.prm 1D_blast_wave.prm COPYONLY)
add_test(
  NAME 1D_BLAST_WAVE_TEST
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_blast_wave.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BLAST_WAVE_TEST BLAST_WAVE_1D
                                    1D
                                    SERIAL
                                    EULER
                                    RUNGE-KUTTA
                                    STRONG-SPLIT
                                    COLLOCATED
                                    LIMITER
                                    MODERATE
                                    INTEGRATION_TEST)