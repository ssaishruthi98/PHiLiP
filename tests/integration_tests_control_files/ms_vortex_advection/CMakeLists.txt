set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D High Temperature Vortex Advection Test
# =======================================
configure_file(1d_multi_species_high_temperature_vortex_advection_3_species.prm 1d_multi_species_high_temperature_vortex_advection_3_species.prm COPYONLY)
add_test(
  NAME 1D_MS_HIGH_TEMP_VORTEX_ADVECTION_TEST
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_high_temperature_vortex_advection_3_species.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_MS_HIGH_TEMP_VORTEX_ADVECTION_TEST MS_VORTEX_ADVECTION
                                            	       1D)
