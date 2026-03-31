set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Multispecies Kelvin Helmholtz Instability
# =======================================
if(${NUMBER_OF_SPECIES} EQUAL 2)
configure_file(2D_multi_species_kelvin_helmholtz_instability.prm 2D_multi_species_kelvin_helmholtz_instability.prm COPYONLY)
add_test(
  NAME 2D_MULTISPECIES_KHI
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_multi_species_kelvin_helmholtz_instability.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_MULTISPECIES_KHI 					2D_MULTISPECIES_KHI
														2D
														PARALLEL
														MULTISPECIES
														RUNGE-KUTTA
														COLLOCATED
														QUICK
														INTEGRATION_TEST)
endif()
