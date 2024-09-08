cmake_minimum_required(VERSION 3.0)
project(miniCFD)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_FLAGS "-Wno-format")

enable_testing()

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries." FORCE)
if (${BUILD_SHARED_LIBS})
  set(SHARED TRUE CACHE BOOL "Build a shared library for METIS." FORCE)
  set(CGNS_BUILD_SHARED ON CACHE BOOL "Build a shared library for CGNS." FORCE)
  set(CGNS_USE_SHARED ON CACHE BOOL "Link programs to the CGNS shared library." FORCE)
  set(CGNS_LIB cgns_shared CACHE STRING "The target of the CGNS shared library." FORCE)
else (${BUILD_SHARED_LIBS})
  set(SHARED FALSE CACHE BOOL "Build a shared library for METIS." FORCE)
  set(CGNS_BUILD_SHARED OFF CACHE BOOL "Build a shared library for CGNS." FORCE)
  set(CGNS_USE_SHARED OFF CACHE BOOL "Link programs to the CGNS static library." FORCE)
  set(CGNS_LIB cgns_static CACHE STRING "The target of the CGNS static library." FORCE)
endif (${BUILD_SHARED_LIBS})

# GoogleTest related settings
option(${PROJECT_NAME}_SUBMODULE_GOOGLETEST "Add GoogleTest as a git submodule." "ON")
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
if (${PROJECT_NAME}_SUBMODULE_GOOGLETEST)
  add_subdirectory(ThirdParty/GoogleTest EXCLUDE_FROM_ALL)
else (${PROJECT_NAME}_SUBMODULE_GOOGLETEST)
  # Download and unpack `googletest` at configure time
  configure_file(CMakeLists.txt.in googletest/download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest/download )
  if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif ()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest/download )
  if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif ()
  # Add `googletest` directly to our build.
  # This defines the `gtest` and `gtest_main`s targets.
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest/src
                   ${CMAKE_CURRENT_BINARY_DIR}/googletest/build
                   EXCLUDE_FROM_ALL)
  # The `gtest`/`gtest_main` targets carry header search path
  # dependencies automatically when using CMake 2.8.11 or later.
  # Otherwise we have to add them here ourselves.
  if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
  endif ()
  # Now simply link against gtest or gtest_main as needed.
endif (${PROJECT_NAME}_SUBMODULE_GOOGLETEST)
# End of GoogleTest related settings

option(${PROJECT_NAME}_ENABLE_MPI "Enable MPI-based parallel computating." "ON")
if (${PROJECT_NAME}_ENABLE_MPI)
  set(N_CORE "4" CACHE STRING "Number of cores used in MPI tests.")
  find_package(MPI REQUIRED)
  message("MPIEXEC_EXECUTABLE = ${MPIEXEC_EXECUTABLE}")
  message("MPI_INCLUDE_PATH = ${MPI_INCLUDE_PATH}")
  message("MPI_C_INCLUDE_PATH = ${MPI_C_INCLUDE_PATH}")
  message("MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}")
  message("MPI_C_HEADER_DIR = ${MPI_C_HEADER_DIR}")
  message("MPI_CXX_HEADER_DIR = ${MPI_CXX_HEADER_DIR}")
  message("MPI_LIBRARY = ${MPI_LIBRARY}")
  message("MPI_LIBRARIES = ${MPI_LIBRARIES}")
  message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
  message("MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")
  set(GTestMPI_INC "${PROJECT_SOURCE_DIR}/ThirdParty/GTestMPI/include")
endif (${PROJECT_NAME}_ENABLE_MPI)

option(${PROJECT_NAME}_64BIT_INDEX "Enable 64-bit indexing." "ON")
set(CGNS_ENABLE_64BIT ${${PROJECT_NAME}_64BIT_INDEX} CACHE BOOL "Let `cgsize_t` in CGNS be 64-bit." FORCE)
set(METIS_IDX64 ${${PROJECT_NAME}_64BIT_INDEX} CACHE BOOL "Let `idx_t` in METIS be 64-bit." FORCE)

option(${PROJECT_NAME}_SUBMODULE_CGNS "Add CGNS as a git submodule." "ON")
if (${PROJECT_NAME}_SUBMODULE_CGNS)
  set(CGNS_ENABLE_HDF5 ON CACHE BOOL "Build CGNS on top of HDF5." FORCE)
  if (${PROJECT_NAME}_ENABLE_MPI)
    set(CGNS_ENABLE_PARALLEL ON CACHE BOOL "Enable parallel CGNS interface." FORCE)
    set(HDF5_NEED_MPI ON CACHE BOOL "Link HDF5 to MPI." FORCE)
  endif (${PROJECT_NAME}_ENABLE_MPI)
  add_subdirectory(ThirdParty/CGNS)
  set(CGNS_INC "${cgns_SOURCE_DIR}/src;${cgns_BINARY_DIR}/src")
else (${PROJECT_NAME}_SUBMODULE_CGNS) # Find installed CGNS.
  if (${PROJECT_NAME}_ENABLE_MPI)
    set(CGNS_HEADER "pcgnslib.h")
  else (${PROJECT_NAME}_ENABLE_MPI)
    set(CGNS_HEADER "cgnslib.h")
  endif (${PROJECT_NAME}_ENABLE_MPI)
  find_path(CGNS_INC ${CGNS_HEADER} HINTS ENV CGNS_ROOT PATH_SUFFIXES include)
  find_library(CGNS_LIB cgns HINTS ENV CGNS_ROOT PATH_SUFFIXES lib)
endif (${PROJECT_NAME}_SUBMODULE_CGNS)

option(${PROJECT_NAME}_ENABLE_VTK "Enable VTK-based IO." "OFF")
if (${PROJECT_NAME}_ENABLE_VTK)
  find_package(VTK COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkIOLegacy
    vtkIOXML
    vtkIOGeometry
    vtkIOImport
    vtkIOExport
    vtksys
  QUIET)
  if (VTK_VERSION VERSION_LESS "8.90.0")
    include(${VTK_USE_FILE})
  endif ()
endif (${PROJECT_NAME}_ENABLE_VTK)

add_subdirectory(ThirdParty/METIS)
set(METIS_INC "${METIS_BINARY_DIR}/include")

set(EIGEN_INC "${PROJECT_SOURCE_DIR}/ThirdParty/eigen")
include_directories(${EIGEN_INC})

set(CGAL_DISABLE_GMP ON)
find_package(CGAL REQUIRED PATHS "${PROJECT_SOURCE_DIR}/ThirdParty/CGAL/Installation/lib/cmake/CGAL")

find_package(Boost REQUIRED)
list(PREPEND CGAL_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
message("${CGAL_INCLUDE_DIRS}")

# Additional headers that depends on ${PROJECT_SOURCE_DIR}
include_directories("${PROJECT_SOURCE_DIR}/include")
# Additional headers that depends on ${PROJECT_BINARY_DIR}
configure_file(
  "${PROJECT_SOURCE_DIR}/input/path.hpp.in"
  "${PROJECT_BINARY_DIR}/include/mini/input/path.hpp"
)
include_directories("${PROJECT_BINARY_DIR}/include")

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "--std c++20" CACHE STRING "C++ standard version" FORCE)

option(${PROJECT_NAME}_BUILD_TESTS "Build tests for this project." "ON")
if (${PROJECT_NAME}_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif (${PROJECT_NAME}_BUILD_TESTS)

option(${PROJECT_NAME}_BUILD_DEMOS "Build demos for this project." "ON")
if (${PROJECT_NAME}_BUILD_DEMOS)
  # Typically you don't care so much for a third party library's tests to be
  # run from your own project's code.
  set(JSON_BuildTests OFF CACHE INTERNAL "")

  # If you only include this third party in PRIVATE source files, you do not
  # need to install it when your main project gets installed.
  # set(JSON_Install OFF CACHE INTERNAL "")

  # Don't use include(nlohmann_json/CMakeLists.txt) since that carries with it
  # unintended consequences that will break the build.  It's generally
  # discouraged (although not necessarily well documented as such) to use
  # include(...) for pulling in other CMake projects anyways.
  add_subdirectory(ThirdParty/nlohmann_json)
  set(nlohmann_json_INC "${nlohmann_json_SOURCE_DIR}/include")

  add_subdirectory(demo)
endif (${PROJECT_NAME}_BUILD_DEMOS)
