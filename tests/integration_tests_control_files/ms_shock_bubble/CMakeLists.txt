set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Multispecies Shock Bubble Test
# =======================================
if(${NUMBER_OF_SPECIES} EQUAL 2)
configure_file(2D_multi_species_shock_bubble.prm 2D_multi_species_shock_bubble.prm COPYONLY)
add_test(
  NAME 2D_MULTISPECIES_SHOCK_BUBBLE_TEST
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_multi_species_shock_bubble.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_MULTISPECIES_SHOCK_BUBBLE_TEST MS_SHOCK_BUBBLE_TEST
						     2D
						     PARALLEL
						     MULTISPECIES
						     RUNGE-KUTTA
						     COLLOCATED
						     LIMITER
						     QUICK
						     INTEGRATION_TEST)
endif()
