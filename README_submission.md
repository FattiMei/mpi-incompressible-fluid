# Content
- `Makefile` and `CMakeLists.txt`: make and cmake files for building the project. If possible, use the make version. Make sure to follow the written instructionts at the top of the respective file. Both build systems produce one or more executables. The executable to be considered is `mif`.
- `src` and `include`: source directories for the project.
- `test`and `generators`: directories with some tests used during development. THey are included in case some debugging is needed, but not needed for the final executable.
- `input`: folder containing an example input file. Only change the values to the right of the colons (:), without changing whitespace. In case of parsing errors, they will be reported in cerr.
- `deps`: folder containing a modified version of the 2decomp library.

# Usage
After compiling using make or cmake, run the program by calling `mpirun -n <num_processors> mif <input-file>`, where` <input-file>` is the path to the input file, with the structure shown in the example input. The input file should be accessible to the processor with rank 0. Its parameters have the following meaning:
- `Nt`: number of time steps.
- `dt`: time discretization step. `Nt * dt` equals the final simulation time, assuming the simulation starts at t=0.
- `Nx`, `Ny`, `Nz`: number of points in the x, y, z direction respectively. Note that points on the border count toward this total, including in the case of periodic boundaries. For example, Nx=4 will result in a mesh with 4 equally spaced points along the x direction, one on each border and 2 inside the mesh.
- `Py`, `Pz`: number of processors to split the domain in the y and z direction respectively. Clearly, they should multiply to `<num_processors>`.
- `test_case_2`: flag signifying whether to simulate test case 1 or 2. If `true`, test case 2 is run, if `false` test case 1 is run.

# Outputs
The executable will produce one `solution.vtk` file, and 2 or 3 `profile*.dat` files. Note that it will also delete any previous output file in the execution directory. The rules for these output files were not clear on a few points, we made the following decisions:
- `vtk data`: we store data in `solution.vtk` with data on the pressure points of the three requested planes. If Paraview is used the "Point Gaussian" option must be selected to visualize the points.
- `duplicate points`: the rules for the `solution.vtk` file request 3 planes. However, the planes may, and in fact do, intersect in some points. We store intersection points once for each requested plane.
- `planes not on pressure points`: the rules on the content of these files were inconsistent, as they requested values on specific planes and lines, but also requested the values to be located in pressure discretization points. However, in some cases, satisfying both requests is not possible. For example, for Nz even and test case 1, the plane z=0 is not located on pressure points. As such, we return the values on the closest plane of pressure points with a coordinate lower or equal to the requested one. In `solution.vtk`, the coordinates are stored exactly, so for example in the first test case the z coordinate for said plane will be slightly lower than 0 if Nz=480 is used. 
- `profiles not on pressure points`: for the `profile*.dat` files, the same issue holds. The values are computed according to the same criterion as the `solution.vtk` file, but the corresponding coordinates are rounded to the requested ones for readability.
- `dat file precision`: the rules do not state the precision data should be stored in the `profile*.dat` files. We opted for 8 decimal places.