add_executable(test_geometry_frame frame.cpp)
set_target_properties(test_geometry_frame PROPERTIES OUTPUT_NAME frame)
add_test(NAME test_geometry_frame COMMAND frame)

add_executable(test_geometry_intersect intersect.cpp)
set_target_properties(test_geometry_intersect PROPERTIES OUTPUT_NAME intersect)
add_test(NAME test_geometry_intersect COMMAND intersect)

set (cases
  triangle
  quadrangle
  tetrahedron
  hexahedron
  wedge
  pyramid
)
foreach (case ${cases})
  add_executable(test_geometry_${case} ${case}.cpp)
  target_include_directories(test_geometry_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_geometry_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_geometry_${case} COMMAND ${case})
endforeach (case ${cases})
