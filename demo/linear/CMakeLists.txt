set (cases
  single
  double
  multiple
)
foreach (case ${cases})
  add_executable(demo_linear_${case} ${case}.cpp)
  set_target_properties(demo_linear_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_linear_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(demo_linear_${case} ${CGNS_LIB} metis ${MPI_LIBRARIES})
endforeach (case ${cases})

set (libs
  system
)
foreach(lib ${libs})
  set(target demo_linear_${lib})
  add_library(${target} ${lib}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${lib})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} ${CGNS_LIB} metis ${MPI_LIBRARIES})
endforeach(lib ${libs})

set (cases
  diffusion
)
foreach (case ${cases})
  set(target demo_linear_${case})
  add_executable(${target} ${case}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} demo_linear_system)
endforeach (case ${cases})
