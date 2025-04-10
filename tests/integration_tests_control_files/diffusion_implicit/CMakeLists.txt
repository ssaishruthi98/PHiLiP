set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_diffusion_sipg_implicit.prm 1d_diffusion_sipg_implicit.prm COPYONLY)
add_test(
  NAME 1D_DIFFUSION_SIPG_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_diffusion_sipg_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_DIFFUSION_SIPG_IMPLICIT_MANUFACTURED_SOLUTION   DIFFUSION_IMPLICIT
                                                                    1D
                                                                    SERIAL
                                                                    DIFFUSION
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(2d_diffusion_sipg_implicit.prm 2d_diffusion_sipg_implicit.prm COPYONLY)
add_test(
  NAME 2D_DIFFUSION_SIPG_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_diffusion_sipg_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_DIFFUSION_SIPG_IMPLICIT_MANUFACTURED_SOLUTION   DIFFUSION_IMPLICIT
                                                                    2D
                                                                    SERIAL
                                                                    DIFFUSION
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(3d_diffusion_sipg_implicit.prm 3d_diffusion_sipg_implicit.prm COPYONLY)
add_test(
  NAME MPI_3D_DIFFUSION_SIPG_IMPLICIT_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_diffusion_sipg_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_3D_DIFFUSION_SIPG_IMPLICIT_MANUFACTURED_SOLUTION_MEDIUM    DIFFUSION_IMPLICIT
                                                                                3D
                                                                                PARALLEL
                                                                                DIFFUSION
                                                                                IMPLICIT
                                                                                WEAK
                                                                                UNCOLLOCATED
                                                                                MANUFACTURED_SOLUTION
                                                                                MODERATE
                                                                                INTEGRATION_TEST)

configure_file(1d_diffusion_br2_implicit.prm 1d_diffusion_br2_implicit.prm COPYONLY)
add_test(
  NAME 1D_DIFFUSION_BR2_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_diffusion_br2_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_DIFFUSION_BR2_IMPLICIT_MANUFACTURED_SOLUTION    DIFFUSION_IMPLICIT
                                                                    1D
                                                                    SERIAL
                                                                    DIFFUSION
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(1d_diffusion_sipg_explicit_strong.prm 1d_diffusion_sipg_explicit_strong.prm COPYONLY)
add_test(
  NAME 1D_DIFFUSION_SIPG_EXPLICIT_MANUFACTURED_SOLUTION_STRONG_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_diffusion_sipg_explicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_DIFFUSION_SIPG_EXPLICIT_MANUFACTURED_SOLUTION_STRONG_LONG   DIFFUSION_IMPLICIT
                                                                                1D
                                                                                SERIAL
                                                                                DIFFUSION
                                                                                RUNGE-KUTTA
                                                                                STRONG
                                                                                UNCOLLOCATED
                                                                                MANUFACTURED_SOLUTION
                                                                                LONG
                                                                                INTEGRATION_TEST)

configure_file(2d_diffusion_br2_implicit.prm 2d_diffusion_br2_implicit.prm COPYONLY)
add_test(
  NAME 2D_DIFFUSION_BR2_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_diffusion_br2_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_DIFFUSION_BR2_IMPLICIT_MANUFACTURED_SOLUTION    DIFFUSION_IMPLICIT
                                                                    2D
                                                                    PARALLEL
                                                                    DIFFUSION
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(3d_diffusion_br2_implicit.prm 3d_diffusion_br2_implicit.prm COPYONLY)
add_test(
  NAME MPI_3D_DIFFUSION_BR2_IMPLICIT_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_diffusion_br2_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_3D_DIFFUSION_BR2_IMPLICIT_MANUFACTURED_SOLUTION_MEDIUM DIFFUSION_IMPLICIT
                                                                            3D
                                                                            PARALLEL
                                                                            DIFFUSION
                                                                            IMPLICIT
                                                                            WEAK
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            MODERATE
                                                                            INTEGRATION_TEST)

# Adjoint test case
configure_file(1d_diffusion_exact_adjoint.prm 1d_diffusion_exact_adjoint.prm COPYONLY)
add_test(
  NAME 1D_DIFFUSION_EXACT_ADJOINT
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_diffusion_exact_adjoint.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_DIFFUSION_EXACT_ADJOINT DIFFUSION_IMPLICIT
                                            1D
                                            SERIAL
                                            DIFFUSION
                                            IMPLICIT
                                            WEAK
                                            UNCOLLOCATED
                                            ADJOINT
                                            MODERATE
                                            INTEGRATION_TEST)

configure_file(2d_diffusion_exact_adjoint.prm 2d_diffusion_exact_adjoint.prm COPYONLY)
add_test(
  NAME 2D_DIFFUSION_EXACT_ADJOINT
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_diffusion_exact_adjoint.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_DIFFUSION_EXACT_ADJOINT DIFFUSION_IMPLICIT
                                            2D
                                            PARALLEL
                                            DIFFUSION
                                            IMPLICIT
                                            WEAK
                                            UNCOLLOCATED
                                            ADJOINT
                                            MODERATE
                                            INTEGRATION_TEST)

configure_file(3d_diffusion_exact_adjoint.prm 3d_diffusion_exact_adjoint.prm COPYONLY)
add_test(
  NAME 3D_DIFFUSION_EXACT_ADJOINT_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_diffusion_exact_adjoint.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(3D_DIFFUSION_EXACT_ADJOINT_LONG    DIFFUSION_IMPLICIT
                                                    3D
                                                    PARALLEL
                                                    DIFFUSION
                                                    IMPLICIT
                                                    WEAK
                                                    UNCOLLOCATED
                                                    ADJOINT
                                                    LONG
                                                    INTEGRATION_TEST)