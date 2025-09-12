set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Multi Species Shock Bubble 2D
# =======================================
# ----------------------------------------
# Shock Bubble Interaction (Multi-Species)
# ----------------------------------------
configure_file(2d_multi_species_shock_bubble.prm 2d_multi_species_shock_bubble.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_SHOCK_BUBBLE
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_shock_bubble.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
