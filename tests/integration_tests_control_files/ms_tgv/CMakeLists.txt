set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_ms_inviscid_tgv.prm 3D_ms_inviscid_tgv.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME 3D_MS_INVISCID_TGV_TEST                                                                                                                                                                                  
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_ms_inviscid_tgv.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)
set_tests_labels(3D_MS_INVISCID_TGV_TEST  3D_MS_INVISCID_TGV_TEST
                                          3D
                                          PARALLEL
                                          REAL-GAS
                                          RUNGE-KUTTA
                                          STRONG-SPLIT
                                          UNCOLLOCATED
                                          EXTRA-LONG
                                          INTEGRATION_TEST)