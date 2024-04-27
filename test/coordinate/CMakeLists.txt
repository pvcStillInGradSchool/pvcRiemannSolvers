set (cases
  triangle
  quadrangle
  tetrahedron
  hexahedron
  wedge
  pyramid
)
foreach (case ${cases})
  add_executable(test_coordinate_${case} ${case}.cpp)
  target_include_directories(test_coordinate_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_coordinate_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_coordinate_${case} COMMAND ${case})
endforeach (case ${cases})
