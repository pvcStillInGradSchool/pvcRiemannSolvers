link_libraries(gtest_main)

add_subdirectory(constant)
add_subdirectory(coordinate)
add_subdirectory(geometry)
add_subdirectory(mesh)
add_subdirectory(integrator)
add_subdirectory(basis)
add_subdirectory(polynomial)
add_subdirectory(limiter)
add_subdirectory(riemann)
add_subdirectory(aircraft)
add_subdirectory(temporal)
add_subdirectory(spatial)

set (cases
  rand
)
foreach(case ${cases})
  add_executable(test_${case} ${case}.cpp)
  set_target_properties(test_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_${case} COMMAND ${case})
endforeach(case ${cases})
