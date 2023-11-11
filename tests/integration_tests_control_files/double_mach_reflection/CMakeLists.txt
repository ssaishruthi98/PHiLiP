set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Double Mach Reflection Test - Euler
# =======================================

configure_file(2D_double_mach.prm 2D_double_mach.prm COPYONLY)
add_test(
  NAME 2D_DOUBLE_MACH_REFLECTION_TEST
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_double_mach.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}/doubleMachVTK
)