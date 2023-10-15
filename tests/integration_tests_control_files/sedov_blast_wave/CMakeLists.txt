set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Low Density Test - Euler
# =======================================

configure_file(2D_sedov_blast_wave.prm 2D_sedov_blast_wave.prm COPYONLY)
add_test(
  NAME 2D_SEDOV_BLAST_WAVE_TEST
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_sedov_blast_wave.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
