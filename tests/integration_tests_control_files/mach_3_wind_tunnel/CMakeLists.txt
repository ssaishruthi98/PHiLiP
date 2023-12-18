set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Mach 3 Wind Tunnel with a Step Test - Euler
# =======================================

configure_file(2D_mach_3_wind_tunnel.prm 2D_mach_3_wind_tunnel.prm COPYONLY)
add_test(
  NAME 2D_MACH_3_WIND_TUNNEL_TEST
  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_mach_3_wind_tunnel.prm
  WORKING_DIRECTORY /media/shruthi/doubleMachPartit/mach3WindTunnelCase/anotherAttempt
)
