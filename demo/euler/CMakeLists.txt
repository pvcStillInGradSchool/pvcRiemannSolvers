set (libs
  sourceless
  rotorcraft
)
foreach(lib ${libs})
  set(target demo_euler_${lib})
  add_library(${target} ${lib}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${lib})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} ${CGNS_LIB} metis ${MPI_LIBRARIES})
endforeach(lib ${libs})

set (cases
  # shock tube problems:
  shock_tube_sod
  shock_tube_lax
  shock_tube_vacuum
  shu_osher
  # two-dimensional problems:
  double_mach
  forward_step
  riemann2d
  # three-dimensional problems:
  ship
  yf17
  mach_disk
  cylinder
)
foreach (case ${cases})
  set(target demo_euler_${case})
  add_executable(${target} ${case}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} demo_euler_sourceless)
endforeach (case ${cases})

set (cases
  wing
  rotor_in_tunnel
  rotor_over_ship
)
foreach (case ${cases})
  set(target demo_euler_${case})
  add_executable(${target} ${case}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} demo_euler_rotorcraft)
endforeach (case ${cases})
