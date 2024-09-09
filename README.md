# miniCFD

[![](https://github.com/pvc1989/miniCFD/workflows/Build/badge.svg)](https://github.com/pvc1989/miniCFD/actions/workflows/build.yml)
[![](https://github.com/pvc1989/miniCFD/workflows/Doxygen/badge.svg)](https://github.com/pvc1989/miniCFD/actions/workflows/doxygen.yml)

## Intention
This repo is a minimum implementation of *Data Structures and Algorithms (DSA)* used in *Computational Fluid Dynamics (CFD)*.

## Build and Test

### Build HDF5

```shell
mkdir HDF5 && cd HDF5
git clone https://github.com/HDFGroup/hdf5.git repo
mkdir build install && cd build
PATH="path-to-your-mpi-install/bin":$PATH
export PATH
cmake -S ../repo -B . -G Ninja -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=OFF -D HDF5_BUILD_TOOLS=OFF -D HDF5_ENABLE_PARALLEL=ON
cmake --build .
cpack -C Release CPackConfig.cmake
cd ../install
../build/HDF5-*-Linux.sh
```

After accepting the license, the script will prompt:

```shell
By default the HDF5 will be installed in:
  "<current directory>/HDF5-1.13.2.1-Linux"
Do you want to include the subdirectory HDF5-1.13.2.1-Linux?
Saying no will install in: "<current directory>" [Yn]:
```

Type `n` will get the following directory structure relative to `<current directory>`:

```
install
└── HDF_Group
    └── HDF5
        └── 1.13.2.1
            ├── bin
            ├── cmake
            ├── include
            ├── lib
            └── share
```

Set `MY_HDF5_DIR` to the `cmake` directory, which contains some `*.cmake` files.

### Build miniCFD

```shell
git clone https://github.com/pvc1989/miniCFD.git
cd miniCFD
git submodule update --init --recursive
mkdir -p build/Release
cd build/Release
cmake -D CMAKE_BUILD_TYPE=Release -D HDF5_DIR=$MY_HDF5_DIR -G Ninja -S ../.. -B .  # cmake 3.13.5+
cmake --build .
ctest
```

## Parallel Execution

```shell
#  mpirun -n <n_cores> <executable> <cgns_file> <hexa|tetra> <t_start> <t_stop> <n_steps_per_frame> <n_frames> [<i_frame_start> [n_parts_prev]]
cd build/Release/demo/euler
# start a new case (t = [0.0, 0.2], frame = [0, 20])
mpirun -n 2 ./sod <filename>.cgns hexa 0.0 0.2 10 20
# restart the old case with a new partition (t = [0.2, 0.5], frame = [20, 50])
mpirun -n 4 ./sod ./sod_hexa/shuffled.cgns hexa 0.2 0.5 10 30 20 2
# restart the old case with the old partition (t = [0.5, 0.8], frame = [50, 80])
mpirun -n 4 ./sod ./sod_hexa/shuffled.cgns hexa 0.5 0.8 10 30 50 4
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
