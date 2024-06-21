set(cases
  simple
  euler
)

foreach (case ${cases})
  add_executable(test_riemann_rotated_${case} ${case}.cpp)
  set_target_properties(test_riemann_rotated_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_riemann_rotated_${case} COMMAND ${case})
endforeach (case ${cases})
