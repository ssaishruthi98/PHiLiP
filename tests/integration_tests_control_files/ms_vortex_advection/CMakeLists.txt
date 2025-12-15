set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_multi_species_vortex_advection.prm 1d_multi_species_vortex_advection.prm COPYONLY)
add_test(
  NAME 1D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES  MULTISPECIES_VORTEX_ADVECTION
                                                                            1D
                                                                            SERIAL
                                                                            MULTISPECIES
                                                                            WEAK
                                                                            COLLOCATED)

