set (libs
  shockless
)
foreach(lib ${libs})
  set(target demo_laminar_${lib})
  add_library(${target} ${lib}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${lib})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} ${CGNS_LIB} metis ${MPI_LIBRARIES})
endforeach(lib ${libs})

set (cases
  cavity
  plate
)
foreach (case ${cases})
  set(target demo_laminar_${case})
  add_executable(${target} ${case}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(${target} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(${target} ${CGNS_LIB} metis ${MPI_LIBRARIES})
endforeach (case ${cases})
