add_executable(test_mesh_cgns cgns.cpp)
target_include_directories(test_mesh_cgns PRIVATE ${CGNS_INC})
target_link_libraries(test_mesh_cgns ${CGNS_LIB})
set_target_properties(test_mesh_cgns PROPERTIES OUTPUT_NAME cgns)
add_test(NAME test_mesh_cgns COMMAND cgns)

add_executable(test_mesh_metis metis.cpp)
target_include_directories(test_mesh_metis PRIVATE ${METIS_INC})
target_link_libraries(test_mesh_metis metis)
set_target_properties(test_mesh_metis PROPERTIES OUTPUT_NAME metis)
add_test(NAME test_mesh_metis COMMAND metis)

add_executable(test_mesh_mapper mapper.cpp)
target_include_directories(test_mesh_mapper PRIVATE ${CGNS_INC} ${METIS_INC})
target_link_libraries(test_mesh_mapper ${CGNS_LIB} metis)
set_target_properties(test_mesh_mapper PROPERTIES OUTPUT_NAME mapper)
add_test(NAME test_mesh_mapper COMMAND mapper)

add_executable(test_mesh_shuffler shuffler.cpp)
target_include_directories(test_mesh_shuffler PRIVATE ${CGNS_INC} ${METIS_INC})
target_link_libraries(test_mesh_shuffler ${CGNS_LIB} metis)
set_target_properties(test_mesh_shuffler PROPERTIES OUTPUT_NAME shuffler)
add_test(NAME test_mesh_shuffler COMMAND shuffler)

add_executable(test_mesh_part part.cpp)
target_include_directories(test_mesh_part PRIVATE ${CGNS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH} ${GTestMPI_INC} ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR})
target_link_libraries(test_mesh_part ${CGNS_LIB} ${MPI_LIBRARIES})
set_target_properties(test_mesh_part PROPERTIES OUTPUT_NAME part)
add_test(NAME test_mesh_part COMMAND mpirun -n ${N_CORE} part)

add_executable(test_mesh_cgal cgal.cpp)
target_include_directories(test_mesh_cgal PRIVATE ${CGAL_INCLUDE_DIRS} ${CGNS_INC})
target_link_libraries(test_mesh_cgal ${CGNS_LIB})
set_target_properties(test_mesh_cgal PROPERTIES OUTPUT_NAME cgal)
add_test(NAME test_mesh_cgal COMMAND cgal)

cuda_add_executable(test_mesh_distance distance.cpp)
target_include_directories(test_mesh_distance PRIVATE ${CGAL_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(test_mesh_distance ${CUDA_CUDA_LIBRARY})
set_target_properties(test_mesh_distance PROPERTIES OUTPUT_NAME distance)

add_executable(test_mesh_overset overset.cpp)
target_include_directories(test_mesh_overset PRIVATE ${CGNS_INC} ${METIS_INC} ${CGAL_INCLUDE_DIRS})
target_link_libraries(test_mesh_overset ${CGNS_LIB} metis)
set_target_properties(test_mesh_overset PROPERTIES OUTPUT_NAME overset)
add_test(NAME test_mesh_overset COMMAND overset)

set (cases
  vtk
)
foreach (case ${cases})
  add_executable(test_mesh_${case} ${case}.cpp)
  target_include_directories(test_mesh_${case} PRIVATE ${CGNS_INC} ${EIGEN_INC} ${GTestMPI_INC} ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_mesh_${case} ${CGNS_LIB} ${MPI_LIBRARIES})
  set_target_properties(test_mesh_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_mesh_${case} COMMAND mpirun -n ${N_CORE} ${case})
  set_tests_properties(test_mesh_${case} PROPERTIES DEPENDS "test_mesh_part")
endforeach (case ${cases})

