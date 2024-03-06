set (cases
  navier_stokes
)
foreach (case ${cases})
  add_executable(test_riemann_diffusive_${case} ${case}.cpp)
  set_target_properties(test_riemann_diffusive_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_riemann_diffusive_${case} COMMAND ${case})
endforeach (case ${cases})
