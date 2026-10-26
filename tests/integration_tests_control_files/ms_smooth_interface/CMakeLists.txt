set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Multispecies Sod Shock Tube test
# =======================================
if(${NUMBER_OF_SPECIES} EQUAL 2)
configure_file(1D_ms_smooth_interface.prm 1D_ms_smooth_interface.prm COPYONLY)
add_test(
  NAME 1D_MULTISPECIES_SMOOTH_INTERFACE_TEST
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_ms_smooth_interface.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_MULTISPECIES_SMOOTH_INTERFACE_TEST 	MULTISPECIES_SMOOTH_INTERFACE_TEST
														1D
														SERIAL
														MULTISPECIES
														RUNGE-KUTTA
														COLLOCATED
														QUICK
														INTEGRATION_TEST)
endif()
