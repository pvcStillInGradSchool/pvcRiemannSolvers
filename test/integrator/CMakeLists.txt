set (cases
  legendre
  lobatto
  jacobi
)
foreach (case ${cases})
  add_executable(test_gauss_${case} ${case}.cpp)
  set_target_properties(test_gauss_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_gauss_${case} COMMAND ${case})
endforeach (case ${cases})

set (cases
  triangle
  quadrangle
  tetrahedron
  hexahedron
  wedge
  pyramid
)
foreach (case ${cases})
  add_executable(test_gauss_${case} ${case}.cpp)
  target_include_directories(test_gauss_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_gauss_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_gauss_${case} COMMAND ${case})
endforeach (case ${cases})
