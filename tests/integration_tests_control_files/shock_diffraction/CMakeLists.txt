set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Shock Diffraction Test - Euler
# =======================================

configure_file(2D_shock_diffraction.prm 2D_shock_diffraction.prm COPYONLY)
add_test(
  NAME 2D_BACKWARDS_FACING_STEP_TEST
  COMMAND mpirun -n 3 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_shock_diffraction.prm
  WORKING_DIRECTORY /media/shruthi/doubleMachPartit/shockDiffractionCase
)
