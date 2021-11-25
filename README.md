# miniCFD

![](https://github.com/pvcStillInGradSchool/miniCFD/workflows/Build/badge.svg)

## Intention
This repo is a minimum implementation of *Data Structures and Algorithms (DSA)* used in *Computational Fluid Dynamics (CFD)*.

## Build
```shell
git clone https://github.com/pvcStillInGradSchool/miniCFD.git
cd miniCFD
git submodule update --init --recursive
mkdir -p build/Release
cd build/Release
cmake -D CMAKE_BUILD_TYPE=Release -G Ninja -S ../.. -B .  # cmake 3.13.5+
cmake --build .
ctest
mkdir result
./demo/euler/tube sod tube.vtk 0.0 0.5 500 5
./demo/euler/box  sod  box.vtk 0.0 1.0 800 5
```

## Parallel Execution

```shell
cd build/Release/test/mesh
rm -rf forward_step && ./shuffler 20 forward_step
mpirun -n 20 ./galerkin forward_step 0.0 4.0 2000 20
pvpython vtk2vtu.py forward_step 20 0 2000 20 5
```

## Code Style

We follow [*Google's C++ style guide*](http://google.github.io/styleguide/cppguide.html) and use [`cpplint`](https://github.com/cpplint/cpplint) to check our code:

```shell
# Install `cpplint` (one time only):
pip3 install cpplint
# Go to the top source directory:
cd ${MINICFD_SOURCE_DIR}
# Check specific files:
cpplint test/mesh/*.cpp
# Check all source files in `include` and `test`:
cpplint --recursive include test --header=hpp
```
