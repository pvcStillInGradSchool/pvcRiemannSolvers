set (tools
  add_iterative_data
  add_cell_quality
  fix_broken_cells
  delete_node
  sort_sections
)

set (use_omp
  add_cell_quality
  fix_broken_cells
)

foreach(tool ${tools})
  message(${tool})
  set(target demo_cgns_${tool})
  add_executable(${target} ${tool}.cpp)
  target_include_directories(${target} PRIVATE ${CGNS_INC})
  if(${tool} IN_LIST use_omp AND OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE ${CGNS_LIB} OpenMP::OpenMP_CXX)
  else()
    target_link_libraries(${target} PRIVATE ${CGNS_LIB})
  endif()
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${tool})
endforeach(tool ${tools})
