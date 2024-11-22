set (tools
  add_iterative_data
  add_cell_quality
  delete_node
)

foreach(tool ${tools})
  set(target demo_cgns_${tool})
  add_executable(${target} ${tool}.cpp)
  target_include_directories(${target} PRIVATE ${CGNS_INC})
  target_link_libraries(${target} ${CGNS_LIB})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${tool})
endforeach(tool ${tools})
