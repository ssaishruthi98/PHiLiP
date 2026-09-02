set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_ms_inviscid_tgv_flux_check.prm 3D_ms_inviscid_tgv_flux_check.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME 3D_MULTISPECIES_INVISCID_ENTROPY_CONSERVING_SPLIT_FORMS_CHECK                                                                                                                                                                                  
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_ms_inviscid_tgv_flux_check.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)
set_tests_labels(3D_MULTISPECIES_INVISCID_ENTROPY_CONSERVING_SPLIT_FORMS_CHECK    3D_MULTISPECIES_INVISCID_ENTROPY_CONSERVING_SPLIT_FORMS_CHECK
                                                                                  3D
                                                                                  PARALLEL
                                                                                  REAL-GAS
                                                                                  RUNGE-KUTTA
                                                                                  STRONG-SPLIT
                                                                                  UNCOLLOCATED
                                                                                  EXTRA-LONG
                                                                                  INTEGRATION_TEST)

configure_file(3D_ms_inviscid_tgv.prm 3D_ms_inviscid_tgv.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME 3D_MULTISPECIES_INVISCID_TGV_TEST                                                                                                                                                                                  
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_ms_inviscid_tgv.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)
set_tests_labels(3D_MULTISPECIES_INVISCID_TGV_TEST  3D_MULTISPECIES_INVISCID_TGV_TEST
                                                    3D
                                                    PARALLEL
                                                    REAL-GAS
                                                    RUNGE-KUTTA
                                                    STRONG-SPLIT
                                                    UNCOLLOCATED
                                                    EXTRA-LONG
                                                    INTEGRATION_TEST)