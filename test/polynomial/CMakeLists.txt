set (cases
  triangle
  quadrangle
  hexahedron
  tetrahedron
  extrapolation
)
foreach(case ${cases})
  add_executable(test_polynomial_${case} ${case}.cpp)
  target_include_directories(test_polynomial_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_polynomial_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_polynomial_${case} COMMAND ${case})
endforeach(case ${cases})

add_executable(test_polynomial_projection projection.cpp)
target_include_directories(test_polynomial_projection PRIVATE ${EIGEN_INC} ${CGNS_INC} ${METIS_INC} ${MPI_INCLUDE_PATH})
target_link_libraries(test_polynomial_projection ${CGNS_LIB} metis)
set_target_properties(test_polynomial_projection PROPERTIES OUTPUT_NAME projection)
add_test(NAME test_polynomial_projection COMMAND projection)
