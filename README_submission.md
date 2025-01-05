# Content
- `Makefile` and `CMakeLists.txt`: make and cmake files for building the project. If possible, use the cmake version. If you are using make, make sure to follow the instructions written in the comments of the Makefile. Both build systems produce one or more executables. The executable to be considered is `mif`.
- `src` and `include`: source directories for the project.
- `test`: directory with some tests used during development. It is included in case some debugging is needed, but is not needed for the final executable.
- `input`: folder containing an example input file. Only change the values to the right of the colons (:), without changing whitespace. In case of parsing errors, they will be reported in cerr.

# Usage
After compiling using make or cmake, run the program by calling `mpirun -n <num_processors> mif <input-file>`, where` <input-file>` is the path to the input file, with the structure shown in the example input. Its parameters have the following meaning:
- `Nt`: number of time steps.
- `dt`: time discretization step. `Nt * dt` equals the final simulation time, assuming the simulation starts at t=0.
- `Nx`, `Ny`, `Nz`: number of points in the x, y, z direction respectively. Note that points on the border count toward this total, including in the case of periodic boundaries. For example, Nx=4 will result in a mesh with 4 equally spaced points along the x direction, one on each border and 2 inside the mesh.
- `Py`, `Pz`: number of processors to split the domain in the y and z direction respectively. Clearly, they should multiply to `<num_processors>`.
- `test_case_2`: flag signifying whether to simulate test case 1 or 2. If `true`, test case 2 is run, if `false` test case 1 is run.

# Outputs
The executable will produce one `solution.vtk` file, and 2 or 3 `profile*.dat` files. The rules requested values on specific planes and lines, but also requested the values to be located in pressure discretization points. However, in some cases, satisfying both requests is not possible. For example, for Nz even and test case 1, the plane z=0 is not located on pressure points. As such, we return the values on the closest plane of pressure points with a coordinate lower or equal to the requested one. In `solution.vtk`, the coordinates are stored exactly, so for example in the first test case the z coordinate for said plane will be slightly lower than 0 if Nz=480 is used. For the `profile*.dat`, the same issue holds. The values are computed in the same positions as the `solution.vtk` file, but the corresponding coordinate is rounded for readability.