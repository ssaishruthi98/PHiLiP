set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 3D multispecies Taylor Green Vortex
# =======================================
if(${NUMBER_OF_SPECIES} EQUAL 2)
configure_file(3D_multi_species_taylor_green_vortex.prm 3D_multi_species_taylor_green_vortex.prm COPYONLY)
add_test(
  NAME 3D_MULTISPECIES_TAYLOR_GREEN_VORTEX_TEST
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_multi_species_taylor_green_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(3D_MULTISPECIES_TAYLOR_GREEN_VORTEX_TEST MS_TAYLOR_GREEN_VORTEX
														3D
														PARALLEL
														MULTISPECIES
														RUNGE-KUTTA
														COLLOCATED
														EXTRA-LONG
														INTEGRATION_TEST)
endif()
