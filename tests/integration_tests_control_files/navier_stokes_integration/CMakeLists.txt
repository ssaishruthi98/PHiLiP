set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Default manufactured solution
# =======================================
# ----------------------------------------
# - dim = 1; conv_num_flux = LxF; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(1d_navier_stokes_lax_friedrichs_sipg_manufactured.prm 1d_navier_stokes_lax_friedrichs_sipg_manufactured.prm COPYONLY)
add_test(
  NAME 1D_NAVIER_STOKES_LAXFRIEDRICHS_SIPG_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_navier_stokes_lax_friedrichs_sipg_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_NAVIER_STOKES_LAXFRIEDRICHS_SIPG_MANUFACTURED_SOLUTION  NAVIER_STOKES_INTEGRATION
                                                                            1D
                                                                            SERIAL
                                                                            NAVIER_STOKES
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            CONVERGENCE
                                                                            QUICK
                                                                            INTEGRATION_TEST)
# ----------------------------------------
# - dim = 1; conv_num_flux = Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(1d_navier_stokes_roe_sipg_manufactured.prm 1d_navier_stokes_roe_sipg_manufactured.prm COPYONLY)
add_test(
  NAME 1D_NAVIER_STOKES_ROE_SIPG_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_navier_stokes_roe_sipg_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_NAVIER_STOKES_ROE_SIPG_MANUFACTURED_SOLUTION    NAVIER_STOKES_INTEGRATION
                                                                    1D
                                                                    SERIAL
                                                                    NAVIER_STOKES
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    CONVERGENCE
                                                                    QUICK
                                                                    INTEGRATION_TEST)
# ----------------------------------------
# - dim = 1; conv_num_flux = L2Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(1d_navier_stokes_l2roe_sipg_manufactured.prm 1d_navier_stokes_l2roe_sipg_manufactured.prm COPYONLY)
add_test(
  NAME 1D_NAVIER_STOKES_L2ROE_SIPG_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_navier_stokes_l2roe_sipg_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_NAVIER_STOKES_L2ROE_SIPG_MANUFACTURED_SOLUTION  NAVIER_STOKES_INTEGRATION
                                                                    1D
                                                                    SERIAL
                                                                    NAVIER_STOKES
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    CONVERGENCE
                                                                    MODERATE
                                                                    INTEGRATION_TEST)
# ----------------------------------------
# - dim = 1; conv_num_flux = Roe; diss_num_flux = BR2
# ----------------------------------------
configure_file(1d_navier_stokes_roe_br2_manufactured.prm 1d_navier_stokes_roe_br2_manufactured.prm COPYONLY)
add_test(
  NAME 1D_NAVIER_STOKES_ROE_BR2_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_navier_stokes_roe_br2_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_NAVIER_STOKES_ROE_BR2_MANUFACTURED_SOLUTION NAVIER_STOKES_INTEGRATION
                                                                1D
                                                                SERIAL
                                                                NAVIER_STOKES
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                MANUFACTURED_SOLUTION
                                                                CONVERGENCE
                                                                MODERATE
                                                                INTEGRATION_TEST)
# ----------------------------------------

# =======================================
# Navah's MS-1
# =======================================
# ----------------------------------------
# - conv_num_flux = LxF; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_laxfriedrichs_manufactured_navah_1.prm 2d_navier_stokes_laxfriedrichs_manufactured_navah_1.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_1
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_laxfriedrichs_manufactured_navah_1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_1   NAVIER_STOKES_INTEGRATION
                                                                                    2D
                                                                                    PARALLEL
                                                                                    NAVIER_STOKES
                                                                                    IMPLICIT
                                                                                    WEAK
                                                                                    UNCOLLOCATED
                                                                                    MANUFACTURED_SOLUTION
                                                                                    CONVERGENCE
                                                                                    MODERATE
                                                                                    INTEGRATION_TEST)

# ----------------------------------------
# - conv_num_flux = Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_roe_manufactured_navah_1.prm 2d_navier_stokes_roe_manufactured_navah_1.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_1
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_roe_manufactured_navah_1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_1 NAVIER_STOKES_INTEGRATION
                                                                        2D
                                                                        PARALLEL
                                                                        NAVIER_STOKES
                                                                        IMPLICIT
                                                                        WEAK
                                                                        UNCOLLOCATED
                                                                        MANUFACTURED_SOLUTION
                                                                        CONVERGENCE
                                                                        MODERATE
                                                                        INTEGRATION_TEST)

# ----------------------------------------
# - conv_num_flux = L2Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_l2roe_manufactured_navah_1.prm 2d_navier_stokes_l2roe_manufactured_navah_1.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_1
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_l2roe_manufactured_navah_1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_1   NAVIER_STOKES_INTEGRATION
                                                                            2D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            CONVERGENCE
                                                                            MODERATE
                                                                            INTEGRATION_TEST)

# ----------------------------------------

# =======================================
# Navah's MS-2
# =======================================
# ----------------------------------------
# - conv_num_flux = LxF; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_laxfriedrichs_manufactured_navah_2.prm 2d_navier_stokes_laxfriedrichs_manufactured_navah_2.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_2
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_laxfriedrichs_manufactured_navah_2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_2   NAVIER_STOKES_INTEGRATION
                                                                                    2D
                                                                                    PARALLEL
                                                                                    NAVIER_STOKES
                                                                                    IMPLICIT
                                                                                    WEAK
                                                                                    UNCOLLOCATED
                                                                                    MANUFACTURED_SOLUTION
                                                                                    CONVERGENCE
                                                                                    MODERATE
                                                                                    INTEGRATION_TEST)

# ----------------------------------------
# - conv_num_flux = Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_roe_manufactured_navah_2.prm 2d_navier_stokes_roe_manufactured_navah_2.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_2
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_roe_manufactured_navah_2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_2 NAVIER_STOKES_INTEGRATION
                                                                        2D
                                                                        PARALLEL
                                                                        NAVIER_STOKES
                                                                        IMPLICIT
                                                                        WEAK
                                                                        UNCOLLOCATED
                                                                        MANUFACTURED_SOLUTION
                                                                        CONVERGENCE
                                                                        MODERATE
                                                                        INTEGRATION_TEST)

# ----------------------------------------
# - conv_num_flux = L2Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_l2roe_manufactured_navah_2.prm 2d_navier_stokes_l2roe_manufactured_navah_2.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_2
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_l2roe_manufactured_navah_2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_2   NAVIER_STOKES_INTEGRATION
                                                                            2D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            CONVERGENCE
                                                                            MODERATE
                                                                            INTEGRATION_TEST)

# ----------------------------------------

# =======================================
# Navah's MS-3
# =======================================
# ----------------------------------------
# - conv_num_flux = LxF; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_laxfriedrichs_manufactured_navah_3.prm 2d_navier_stokes_laxfriedrichs_manufactured_navah_3.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_3
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_laxfriedrichs_manufactured_navah_3.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_3   NAVIER_STOKES_INTEGRATION
                                                                                    2D
                                                                                    PARALLEL
                                                                                    NAVIER_STOKES
                                                                                    IMPLICIT
                                                                                    WEAK
                                                                                    UNCOLLOCATED
                                                                                    MANUFACTURED_SOLUTION
                                                                                    CONVERGENCE
                                                                                    MODERATE
                                                                                    INTEGRATION_TEST)
# ----------------------------------------
# - conv_num_flux = Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_roe_manufactured_navah_3.prm 2d_navier_stokes_roe_manufactured_navah_3.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_3
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_roe_manufactured_navah_3.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_3 NAVIER_STOKES_INTEGRATION
                                                                        2D
                                                                        PARALLEL
                                                                        NAVIER_STOKES
                                                                        IMPLICIT
                                                                        WEAK
                                                                        UNCOLLOCATED
                                                                        MANUFACTURED_SOLUTION
                                                                        CONVERGENCE
                                                                        MODERATE
                                                                        INTEGRATION_TEST)

# ----------------------------------------
# - conv_num_flux = L2Roe; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_navier_stokes_l2roe_manufactured_navah_3.prm 2d_navier_stokes_l2roe_manufactured_navah_3.prm COPYONLY)
add_test(
  NAME MPI_2D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_3
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_navier_stokes_l2roe_manufactured_navah_3.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_3   NAVIER_STOKES_INTEGRATION
                                                                            2D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            CONVERGENCE
                                                                            MODERATE
                                                                            INTEGRATION_TEST)

# ----------------------------------------

# =======================================
# Navah's MS-4
# =======================================
# ----------------------------------------
# - conv_num_flux = LxF; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_rans_sa_laxfriedrichs_manufactured_navah_4.prm 2d_rans_sa_laxfriedrichs_manufactured_navah_4.prm COPYONLY)
add_test(
  NAME MPI_2D_RANS_SA_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_4
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_rans_sa_laxfriedrichs_manufactured_navah_4.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_RANS_SA_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_4 NAVIER_STOKES_INTEGRATION
                                                                            2D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            CONVERGENCE
                                                                            RANS
                                                                            MODERATE
                                                                            INTEGRATION_TEST)
# ----------------------------------------

# =======================================
# Navah's MS-5
# =======================================
# ----------------------------------------
# - conv_num_flux = LxF; diss_num_flux = SIP (default)
# ----------------------------------------
configure_file(2d_rans_sa_laxfriedrichs_manufactured_navah_5.prm 2d_rans_sa_laxfriedrichs_manufactured_navah_5.prm COPYONLY)
add_test(
  NAME MPI_2D_RANS_SA_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_5
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_rans_sa_laxfriedrichs_manufactured_navah_5.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_RANS_SA_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_5 NAVIER_STOKES_INTEGRATION
                                                                            2D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            CONVERGENCE
                                                                            RANS
                                                                            MODERATE
                                                                            INTEGRATION_TEST)