#include <cmath>
#include <cstdio>
#include <numeric>
#include <fstream>
#include <tuple>
#include "Endians.h"
#include "VTKDatExport.h"

// If someone wants to put the hands in this program, here is a brief description:
//
// REQUIREMENTS:
//   * export u, v, w, and p (interpolated on pressure points) at planes
//     - x = 0
//     - y = 0
//     - z = 0
//
// IMPLEMENTATION:
//   * each processor writes a little piece of a single big file using MPI file API
//   * each need to know where we are in the file (global offset)
//     - for the parts where only one first processor writes data, the other ones
//       will "render" the data and update their local index
//
// SECTIONS:
//   1. interpolation
//   2. point coordinates computing
//   3. vtk header
//   4. vtk point data
//   5. vtk point values
//
// MISSING:
//   * cell support for vtk: the file will be rendered as surfaces on paraview.
//     right now only "point gaussian" option is available and it's not good, not a priority though
//
// The desired structure of the vtk file is available at analysis/vtk/cell.vtk


namespace mif {

// Given a position on an axis, the global minimum of the domain and the discretization
// step, compute the index corresponding to the closest position to the left in pressure
// points. In general, this position may not be the exact position, which will be between
// this position and the position + delta. The result will be interpolated using the
// data at the index returned by this function and at the index + 1, using the value
// returned by this function as weight for the first value, and 1 - that weight for the
// second value.
std::tuple<size_t, float> pos_to_index(Real pos, Real min_pos_global, Real delta) {
    const Real offset = pos - min_pos_global;
    const Real float_index = offset / delta;
    const Real int_index_1 = std::floor(float_index);
    const Real index_1_importance = 1.0 - (float_index - int_index_1);
    return {int_index_1, index_1_importance};
}

// Compute the file cell offsets as each processor writes data about its local points,
// and the others need to know how much space it occupied.
std::vector<int> compute_displacements(int n_local_points, int mpi_size){
    std::vector<int> count(mpi_size);
    std::vector<int> displacements(mpi_size + 1);

    MPI_Allgather(&n_local_points, 1, MPI_INT, count.data(), 1, MPI_INT, MPI_COMM_WORLD);

    displacements[0] = 0;
    for (int i = 0; i < mpi_size; ++i){
        displacements[i + 1] = displacements[i] + count[i];
    }

    return displacements;
}

// Write the ASCII part of the VTK file.
inline MPI_Offset write_ascii_part(
    MPI_File fh,
    MPI_Offset global_offset,
    int strlen,
    char* buf,
    int rank){
    if (rank == 0){
        MPI_Status status;
        int outcome = MPI_File_write_at(fh, global_offset, buf, strlen, MPI_CHAR, &status);
        assert(outcome == MPI_SUCCESS);
        (void)outcome;
    }
    return strlen;
}

// Get sizes of local domain without ghosts points, but with both the start and end of the domain
// in case of periodic BC.
// Get indices in the unstaggered tensors for start and end of the local domain (start/end_*_write_local)
// and the global one (start/end_*_write_global), inclusive on the left and exclusive on the right.
#define COMPUTE_INDEXING()                                                                  \
    const size_t Nx = constants.Nx_global;                                                  \
    const size_t Ny = (constants.y_rank == constants.Py - 1 && constants.periodic_bc[1])    \
                            ? constants.Ny_owner + 1                                        \
                            : constants.Ny_owner;                                           \
    const size_t Nz = (constants.z_rank == constants.Pz - 1 && constants.periodic_bc[2])    \
                            ? constants.Nz_owner + 1                                        \
                            : constants.Nz_owner;                                           \
    const int start_i_write_local = constants.periodic_bc[0] ? 1 : 0;                       \
    const int start_j_write_local = (constants.prev_proc_y == -1) ? 0 : 1;                  \
    const int start_k_write_local = (constants.prev_proc_z == -1) ? 0 : 1;                  \
    const int end_i_write_local = start_i_write_local + Nx;                                 \
    const int end_j_write_local = start_j_write_local + Ny;                                 \
    const int end_k_write_local = start_k_write_local + Nz;                                 \
    const int start_i_write_global = constants.base_i + start_i_write_local;                \
    const int start_j_write_global = constants.base_j + start_j_write_local;                \
    const int start_k_write_global = constants.base_k + start_k_write_local;                \
    const int end_i_write_global = start_i_write_global + Nx;                               \
    const int end_j_write_global = start_j_write_global + Ny;                               \
    const int end_k_write_global = start_k_write_global + Nz;                               \
    (void) end_i_write_global;                                                              \
    (void) end_j_write_global;                                                              \
    (void) end_k_write_global;                                                              \

// Write the VTK file.
void writeVTK(const std::string& filename,
                const VelocityTensor& velocity,
                const StaggeredTensor& pressure){
    // Get needed constants.
    const Constants& constants = velocity.constants;
    const int rank = constants.rank;
    const int size = constants.Py * constants.Pz;

    const char* typestr = (sizeof(Real) == 8) ? "double" : "float";
    char buf[1024];

    // Open the file.
    MPI_Offset global_offset = 0;
    MPI_File fh;
    MPI_Status status;
    const int outcome = MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                                        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    assert(outcome == MPI_SUCCESS);
    (void) outcome;

    // Compute local and global indices.
    COMPUTE_INDEXING();

    // Allocate space for local results.
    // "local_cells" is the number of points the local processor will have to write results for.
    size_t local_cells = (Nx * Ny + Nx * Nz + Nx * Ny);
    std::vector<Real> points_coordinates;
    std::vector<Real> point_data_u, point_data_v, point_data_w, point_data_p, point_data_mag;
    points_coordinates.reserve(local_cells * 3);
    point_data_u.reserve(local_cells);
    point_data_v.reserve(local_cells);
    point_data_w.reserve(local_cells);
    point_data_p.reserve(local_cells);

    // Write data in a point given its indices.
    auto write_point = [&constants, &velocity, &pressure, &points_coordinates,
            &point_data_u, &point_data_v, &point_data_w, &point_data_p]
    (int i, int j, int k) {
        points_coordinates.push_back(constants.min_x_global + (constants.base_i + i) * constants.dx);
        points_coordinates.push_back(constants.min_y_global + (constants.base_j + j) * constants.dy);
        points_coordinates.push_back(constants.min_z_global + (constants.base_k + k) * constants.dz);
        point_data_u.push_back((velocity.u(i, j, k) + velocity.u(i + 1, j, k)) / 2);
        point_data_v.push_back((velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2);
        point_data_w.push_back((velocity.w(i, j, k) + velocity.w(i, j, k + 1)) / 2);
        point_data_p.push_back(pressure(i, j, k));
    };

    // z = 0 plane. This must be first to simplify the cell creation.
    {
        assert(constants.min_z_global <= 0.0 && (constants.min_z_global + constants.z_size_global) >= 0.0);
        const std::tuple<int, Real> index = pos_to_index(0.0, constants.min_z_global, constants.dz);
        const int k_global = std::get<0>(index);
        if (k_global >= start_k_write_global && k_global < end_k_write_global) {
            const int k = k_global - start_k_write_global + start_k_write_local;
            for (int i = start_i_write_local; i < end_i_write_local; i++){
                for (int j = start_j_write_local; j < end_j_write_local; j++){
                    write_point(i, j, k);
                }
            }
        }
    }

    // x = 0 plane.
    {
        assert(constants.min_x_global <= 0.0 && (constants.min_x_global + constants.x_size) >= 0.0);
        const std::tuple<int, Real> index = pos_to_index(0.0, constants.min_x_global, constants.dx);
        const int i_global = std::get<0>(index);
        assert(i_global >= start_i_write_global && i_global < end_i_write_global);
        const int i = i_global - start_i_write_global + start_i_write_local;
        for (int j = start_j_write_local; j < end_j_write_local; j++) {
            for (int k = start_k_write_local; k < end_k_write_local; k++){
                write_point(i, j, k);
            }
        }
    }

    // y = 0 plane.
    {
        assert(constants.min_y_global <= 0.0 && (constants.min_y_global + constants.y_size_global) >= 0.0);
        const std::tuple<int, Real> index = pos_to_index(0.0, constants.min_y_global, constants.dy);
        const int j_global = std::get<0>(index);
        if (j_global >= start_j_write_global && j_global < end_j_write_global) {
            const int j = j_global - start_j_write_global + start_j_write_local;
            for (int i = start_i_write_local; i < end_i_write_local; i++){
                for (int k = start_k_write_local; k < end_k_write_local; k++){
                    write_point(i, j, k);
                }
            }
        }
    }

    // Compute file displacements.
    const std::vector<int> displacements = compute_displacements(point_data_u.size(), size);
    const int num_elem = displacements.back();

    // Convert points_coordinate to big-endian format.
    vectorToBigEndian(points_coordinates);
    vectorToBigEndian(point_data_u);
    vectorToBigEndian(point_data_v);
    vectorToBigEndian(point_data_w);
    vectorToBigEndian(point_data_p);
    vectorToBigEndian(point_data_mag);

    // Write the coordinates.
    MPI_Offset my_offset;
    {
        global_offset += write_ascii_part(
            fh,
            global_offset,
            sprintf(
                buf,
                "# vtk DataFile Version 2.0\nvtk output\nBINARY\nDATASET UNSTRUCTURED_GRID \nPOINTS %d %s\n",
                displacements.back(),
                typestr
            ),
            buf,
            rank
        );

        my_offset = global_offset + 3 * displacements[rank] * sizeof(Real);
        MPI_File_write_at(fh, my_offset, points_coordinates.data(),
                            points_coordinates.size() * sizeof(Real),
                            MPI_BYTE, &status);
        global_offset += 3 * num_elem * sizeof(Real);
    }

    // Write the u component of the velocity.
    {
        global_offset += write_ascii_part(fh, global_offset,
                                            sprintf(
                                                buf,
                                                "\nPOINT_DATA %d\nSCALARS u %s 1\nLOOKUP_TABLE default\n",
                                                num_elem,
                                                typestr
                                            ),
                                            buf, rank
        );

        my_offset = global_offset + displacements[rank] * sizeof(Real);
        MPI_File_write_at(fh, my_offset, point_data_u.data(), point_data_u.size() * sizeof(Real), MPI_BYTE,
                            &status);
        global_offset += num_elem * sizeof(Real);
    }

    // Write the v component of the velocity.
    {
        global_offset += write_ascii_part(fh, global_offset,
                                            sprintf(
                                                buf,
                                                "\nSCALARS v %s 1\nLOOKUP_TABLE default\n",
                                                typestr
                                            ),
                                            buf, rank
        );

        my_offset = global_offset + displacements[rank] * sizeof(Real);
        MPI_File_write_at(fh, my_offset, point_data_v.data(), point_data_v.size() * sizeof(Real), MPI_BYTE,
                            &status);
        global_offset += num_elem * sizeof(Real);
    }

    // Write the w component of the velocity.
    {
        global_offset += write_ascii_part(fh, global_offset,
                                            sprintf(
                                                buf,
                                                "\nSCALARS w %s 1\nLOOKUP_TABLE default\n",
                                                typestr
                                            ),
                                            buf, rank
        );

        my_offset = global_offset + displacements[rank] * sizeof(Real);
        MPI_File_write_at(fh, my_offset, point_data_w.data(), point_data_w.size() * sizeof(Real), MPI_BYTE,
                            &status);
        global_offset += num_elem * sizeof(Real);
    }

    // Write the pressure.
    {
        global_offset += write_ascii_part(fh, global_offset,
                                            sprintf(
                                                buf,
                                                "\nSCALARS p %s 1\nLOOKUP_TABLE default\n",
                                                typestr
                                            ),
                                            buf, rank
        );

        my_offset = global_offset + displacements[rank] * sizeof(Real);
        MPI_File_write_at(fh, my_offset, point_data_p.data(), point_data_p.size() * sizeof(Real), MPI_BYTE,
                            &status);
        global_offset += num_elem * sizeof(Real);
    }

    // Close the file.
    MPI_File_close(&fh);
}


void insertionSort(std::vector<Real>& coordinates, std::vector<Real>& u, 
                   std::vector<Real>& v, std::vector<Real>& w,
                   std::vector<Real>& p) {
    const int n = coordinates.size();
    for (int i = 1; i < n; i++){
        const Real key = coordinates[i];
        const Real key_u = u[i];
        const Real key_v = v[i];
        const Real key_w = w[i];
        const Real key_p = p[i];
        int j = i - 1;
        while (j >= 0 && coordinates[j] > key) {
            coordinates[j + 1] = coordinates[j];
            u[j + 1] = u[j];
            v[j + 1] = v[j];
            w[j + 1] = w[j];
            p[j + 1] = p[j];
            j--;
        }
        coordinates[j + 1] = key;
        u[j + 1] = key_u;
        v[j + 1] = key_v;
        w[j + 1] = key_w;
        p[j + 1] = key_p;
    }
}

void writeDat(
    const std::string& filename,
    const VelocityTensor& velocity,
    const StaggeredTensor& pressure,
    const int direction,
    const Real x, const Real y, const Real z
) {
    const Constants &constants = velocity.constants;
    const int rank = constants.rank;
    const int mpi_size = constants.Py * constants.Pz;
    assert(direction >= 0 && direction <= 2);

    // Get the global indices of the point the line passes through.
    const std::tuple<int, Real> index_1 = pos_to_index(x, constants.min_x_global, constants.dx);
    const std::tuple<int, Real> index_2 = pos_to_index(y, constants.min_y_global, constants.dy);
    const std::tuple<int, Real> index_3 = pos_to_index(z, constants.min_z_global, constants.dz);
    const int i_global = std::get<0>(index_1);
    const int j_global = std::get<0>(index_2);
    const int k_global = std::get<0>(index_3);

    // Compute local and global indices.
    COMPUTE_INDEXING();

    // Create the needed structures.
    std::vector<Real> point_data_u, point_data_v, point_data_w, point_data_p;
    const int size = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
    point_data_u.reserve(size);
    point_data_v.reserve(size);
    point_data_w.reserve(size);
    point_data_p.reserve(size);
    std::vector<Real> points_coordinates;

    // Write data in a point given its indices.
    auto write_point = [&constants, &velocity, &pressure, &points_coordinates,
            &point_data_u, &point_data_v, &point_data_w, &point_data_p]
    (int i, int j, int k, Real pos) {
        points_coordinates.push_back(pos);
        point_data_u.push_back((velocity.u(i, j, k) + velocity.u(i + 1, j, k)) / 2);
        point_data_v.push_back((velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2);
        point_data_w.push_back((velocity.w(i, j, k) + velocity.w(i, j, k + 1)) / 2);
        point_data_p.push_back(pressure(i, j, k));
    };

    // Write the requested line.
    // Parallel to x axis.
    if (direction == 0) {
        if (j_global >= start_j_write_global && j_global < end_j_write_global && 
            k_global >= start_k_write_global && k_global < end_k_write_global) {
            const int j = j_global - start_j_write_global + start_j_write_local;
            const int k = k_global - start_k_write_global + start_k_write_local;
            for (int i = start_i_write_local; i < end_i_write_local; i++) {
                write_point(i, j, k, constants.min_x_global + (constants.base_i + i) * constants.dx);
            }
        }
    }
    // Parallel to y axis.
    else if (direction == 1) {
        if (i_global >= start_i_write_global && i_global < end_i_write_global && 
            k_global >= start_k_write_global && k_global < end_k_write_global) {
            const int i = i_global - start_i_write_global + start_i_write_local;
            const int k = k_global - start_k_write_global + start_k_write_local;
            for (int j = start_j_write_local; j < end_j_write_local; j++) {
                write_point(i, j, k, constants.min_y_global + (constants.base_j + j) * constants.dy);
            }
        }
    }
    // Parallel to z axis.
    else if (direction == 2) {
        if (i_global >= start_i_write_global && i_global < end_i_write_global && 
            j_global >= start_j_write_global && j_global < end_j_write_global) {
            const int i = i_global - start_i_write_global + start_i_write_local;
            const int j = j_global - start_j_write_global + start_j_write_local;
            for (int k = start_k_write_local; k < end_k_write_local; k++) {
                write_point(i, j, k, constants.min_z_global + (constants.base_k + k) * constants.dz);
            }
        }
    }

    // Send size data to the first processor.
    const size_t local_size = point_data_u.size();
    std::vector<int> counts(mpi_size), displacements(mpi_size);
    MPI_Gather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < mpi_size; i++){
            counts[i] *= sizeof(Real);
        }
    }
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 0; i < mpi_size; i++){
            displacements[i] = displacements[i - 1] + counts[i - 1];
        }
    }

    // Allocate space for the data.
    std::vector<Real> point_data_u_global(std::accumulate(counts.begin(), counts.end(), 0) / sizeof(Real));
    std::vector<Real> point_data_v_global(std::accumulate(counts.begin(), counts.end(), 0) / sizeof(Real));
    std::vector<Real> point_data_w_global(std::accumulate(counts.begin(), counts.end(), 0) / sizeof(Real));
    std::vector<Real> point_data_p_global(std::accumulate(counts.begin(), counts.end(), 0) / sizeof(Real));
    std::vector<Real> points_coordinate_global(std::accumulate(counts.begin(), counts.end(), 0) / sizeof(Real));

    // Send the data to the first processor.
    if (rank == 0 || local_size > 0) {
        MPI_Gatherv(point_data_u.data(), local_size * sizeof(Real), MPI_BYTE, point_data_u_global.data(),
                    counts.data(),
                    displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(point_data_v.data(), local_size * sizeof(Real), MPI_BYTE, point_data_v_global.data(),
                    counts.data(),
                    displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(point_data_w.data(), local_size * sizeof(Real), MPI_BYTE, point_data_w_global.data(),
                    counts.data(),
                    displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(point_data_p.data(), local_size * sizeof(Real), MPI_BYTE, point_data_p_global.data(),
                    counts.data(),
                    displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(points_coordinates.data(), local_size * sizeof(Real), MPI_BYTE, points_coordinate_global.data(),
                    counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // Sort the data based on the coordinates and write it to the file.
    if (rank == 0) {
        insertionSort(points_coordinate_global, point_data_u_global, point_data_v_global, point_data_w_global,
                      point_data_p_global);

        FILE* file = fopen(filename.c_str(), "w");
        if (direction == 0) {
            for (size_t i = 0; i < points_coordinate_global.size(); i++) {
                fprintf(file, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", points_coordinate_global[i], y, z, point_data_u_global[i],
                        point_data_v_global[i], point_data_w_global[i], point_data_p_global[i]);
            }
        }
        else if (direction == 1) {
            for (size_t i = 0; i < points_coordinate_global.size(); i++) {
                fprintf(file, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", x, points_coordinate_global[i], z, point_data_u_global[i],
                        point_data_v_global[i], point_data_w_global[i], point_data_p_global[i]);
            }
        }
        else if (direction == 2) {
            for (size_t i = 0; i < points_coordinate_global.size(); i++) {
                fprintf(file, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", x, y, points_coordinate_global[i], point_data_u_global[i],
                        point_data_v_global[i], point_data_w_global[i], point_data_p_global[i]);
            }
        }
        fclose(file);
    }
}


void writeVTKFullMesh(const std::string& filename,
                        const mif::VelocityTensor& velocity,
                        const mif::StaggeredTensor& pressure) {
    std::ofstream out(filename);
    const mif::Constants& constants = velocity.constants;
    assert(constants.Py * constants.Pz == 1);

    // Compute local and global indices.
    COMPUTE_INDEXING();

    out
        << "# vtk DataFile Version 3.0\n"
        << "pressure mesh solution\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << Nx << ' ' << Ny << ' ' << Nz << '\n'
        << "ORIGIN " << constants.min_x_global << " " << constants.min_y_global << " " << constants.min_z_global << "\n"
        << "SPACING " << constants.dx << ' ' << constants.dy << ' ' << constants.dz << '\n'
        << "POINT_DATA " << Nx*Ny*Nz << '\n';

    out
        << "SCALARS u double 1\n"
        << "LOOKUP_TABLE default\n";

    for (int k = start_k_write_local; k < end_k_write_local; ++k) {
        for (int j = start_j_write_local; j < end_j_write_local; ++j) {
            for (int i = start_i_write_local; i < end_i_write_local; ++i) {
                out << ((velocity.u(i, j, k) + velocity.u(i + 1, j, k)) / 2) << ' ';
            }
        }
    }

    out
        << "SCALARS v double 1\n"
        << "LOOKUP_TABLE default\n";

    for (int k = start_k_write_local; k < end_k_write_local; ++k) {
        for (int j = start_j_write_local; j < end_j_write_local; ++j) {
            for (int i = start_i_write_local; i < end_i_write_local; ++i) {
                out << ((velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2) << ' ';
            }
        }
    }

    out
        << "SCALARS w double 1\n"
        << "LOOKUP_TABLE default\n";

    for (int k = start_k_write_local; k < end_k_write_local; ++k) {
        for (int j = start_j_write_local; j < end_j_write_local; ++j) {
            for (int i = start_i_write_local; i < end_i_write_local; ++i) {
                out << ((velocity.w(i, j, k) + velocity.w(i, j, k + 1)) / 2) << ' ';
            }
        }
    }

    out
        << "SCALARS |u| double 1\n"
        << "LOOKUP_TABLE default\n";

    for (int k = start_k_write_local; k < end_k_write_local; ++k) {
        for (int j = start_j_write_local; j < end_j_write_local; ++j) {
            for (int i = start_i_write_local; i < end_i_write_local; ++i) {
                const Real ux = ((velocity.u(i, j, k) + velocity.u(i + 1, j, k)) / 2);
                const Real uy = ((velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2);
                const Real uz = ((velocity.w(i, j, k) + velocity.w(i, j, k + 1)) / 2);

                out << std::sqrt(ux * ux + uy * uy + uz * uz) << ' ';
            }
        }
    }

    out
        << "SCALARS p double 1\n"
        << "LOOKUP_TABLE default\n";

    for (int k = start_k_write_local; k < end_k_write_local; ++k){
        for (int j = start_j_write_local; j < end_j_write_local; ++j){
            for (int i = start_i_write_local; i < end_i_write_local; ++i){
                out << pressure(i, j, k) << ' ';
            }
        }
    }
}

} // mif
