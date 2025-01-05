#include <cmath>
#include <cstdio>
#include <numeric>
#include <fstream>
#include <tuple>
#include "Endians.h"
#include "VTKExport.h"

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
    std::tuple<size_t, float> pos_to_index(Real pos, Real min_pos_global, Real delta, bool periodic) {
        const Real offset = pos - min_pos_global + periodic * delta;
        const Real float_index = offset / delta;
        const Real int_index_1 = std::floor(float_index);
        const Real index_1_importance = 1.0 - (float_index - int_index_1);
        return {int_index_1, index_1_importance};
    }

    // TODO: add coordinates offsets to the functions.
    // Compute the file cell offsets as each processor writes data about its local points,
    // and the others need to know how much space it occupied.
    std::vector<int> compute_displacements(int n_local_points, int mpi_size) {
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
        int rank) {
        if (rank == 0) {
            MPI_Status status;
            int outcome = MPI_File_write_at(fh, global_offset, buf, strlen, MPI_CHAR, &status);
            assert(outcome == MPI_SUCCESS);
            (void) outcome;
        }
        return strlen;
    }

    // Write the VTK file.
    void writeVTK(const std::string& filename,
                  const VelocityTensor& velocity,
                  const StaggeredTensor& pressure) {
        // Get needed constants.
        const Constants &constants = velocity.constants;
        const int rank = constants.rank;
        const int size = constants.Py * constants.Pz;

        const char* typestr = (sizeof(Real) == 8) ? "double" : "float";
        char buf[1024];

        // Open the file.
        MPI_Offset global_offset = 0;
        MPI_File fh;
        MPI_Status status;
        const int outcome = MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                                          MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        assert(outcome == MPI_SUCCESS);

        // Get sizes of local domain without ghosts points, but with both the start and end of the domain
        // in case of periodic BC.
        // Get indices in the unstaggered tensors for start and end of the local domain (start/end_*_write_local)
        // and the global one (start/end_*_write_global), inclusive on the left and exclusive on the right.
        const size_t Nx = constants.Nx_global;
        const size_t Ny = (constants.y_rank == constants.Py-1 && constants.periodic_bc[1]) ? constants.Ny_owner+1 : constants.Ny_owner;
        const size_t Nz = (constants.z_rank == constants.Pz-1 && constants.periodic_bc[2]) ? constants.Nz_owner+1 : constants.Nz_owner;
        const int start_i_write_local = constants.periodic_bc[0] ? 1 : 0;
        const int start_j_write_local = (constants.prev_proc_y == -1) ? 0 : 1;
        const int start_k_write_local = (constants.prev_proc_z == -1) ? 0 : 1;
        const int end_i_write_local = start_i_write_local + Nx;
        const int end_j_write_local = start_j_write_local + Ny;
        const int end_k_write_local = start_k_write_local + Nz;
        const int start_i_write_global = constants.base_i + start_i_write_local;
        const int start_j_write_global = constants.base_j + start_j_write_local;
        const int start_k_write_global = constants.base_k + start_k_write_local;
        const int end_i_write_global = start_i_write_global + Nx;
        const int end_j_write_global = start_j_write_global + Ny;
        const int end_k_write_global = start_k_write_global + Nz;
        (void) end_i_write_global;

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
        point_data_mag.reserve(local_cells);


        // TODO: for now, data is returned on the pressure point closest to the left to the requested plane.
        // Find out if we should interpolate the data.
        // TODO: this may contain duplicate points. Find out if it is a problem.

        // Write data in a point given its indices.
        auto write_point = [&constants, &velocity, &pressure, &points_coordinates,
                            &point_data_u, &point_data_v, &point_data_w, &point_data_p, &point_data_mag]
                            (int i, int j, int k) {
            points_coordinates.push_back(constants.min_x_global + (constants.base_i + i) * constants.dx);
            points_coordinates.push_back(constants.min_y_global + (constants.base_j + j) * constants.dy);
            points_coordinates.push_back(constants.min_z_global + (constants.base_k + k) * constants.dz);
            point_data_u.push_back((velocity.u(i, j, k) + velocity.u(i + 1, j, k)) / 2);
            point_data_v.push_back((velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2);
            point_data_w.push_back((velocity.w(i, j, k) + velocity.w(i, j, k + 1)) / 2);
            point_data_p.push_back(pressure(i, j, k));

	    point_data_mag.push_back(
	        std::sqrt(
		      point_data_u.back()*point_data_u.back()
		    + point_data_v.back()*point_data_v.back()
		    + point_data_w.back()*point_data_w.back()
		)
	    );
        };

        // x = 0 plane.
        {
            assert(constants.min_x_global <= 0.0 && (constants.min_x_global + constants.x_size) >= 0.0);
            const std::tuple<int, Real> index = pos_to_index(0.0, constants.min_x_global, constants.dx, constants.periodic_bc[0]);
            const int i = std::get<0>(index);
            assert(i >= start_i_write_global && i < end_i_write_global);
            for (int j = start_j_write_local; j < end_j_write_local; j++) {
                for (int k = start_k_write_local; k < end_k_write_local; k++) {
                    write_point(i, j, k);
                }
            }
        }

        // y = 0 plane.
        {
            assert(constants.min_y_global <= 0.0 && (constants.min_y_global + constants.y_size_global) >= 0.0);
            const std::tuple<int, Real> index = pos_to_index(0.0, constants.min_y_global, constants.dy, constants.periodic_bc[1]);
            const int j_global = std::get<0>(index);
            if (j_global >= start_j_write_global && j_global < end_j_write_global) {
                const int j = j_global - start_j_write_global;
                for (int i = start_i_write_local; i < end_i_write_local; i++) {
                    for (int k = start_k_write_local; k < end_k_write_local; k++) {
                        write_point(i, j, k);
                    }
                }
            }
        }

        // z = 0 plane.
        {
            assert(constants.min_z_global <= 0.0 && (constants.min_z_global + constants.z_size_global) >= 0.0);
            const std::tuple<int, Real> index = pos_to_index(0.0, constants.min_z_global, constants.dz, constants.periodic_bc[2]);
            const int k_global = std::get<0>(index);
            if (k_global >= start_k_write_global && k_global < end_k_write_global) {
                const int k = k_global - start_k_write_global;
                for (int i = start_i_write_local; i < end_i_write_local; i++) {
                    for (int j = start_j_write_local; j < end_j_write_local; j++) {
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

        // Write the velocity magnitude, pretty useful
        {
            global_offset += write_ascii_part(fh, global_offset,
                                              sprintf(
                                                  buf,
                                                  "\nSCALARS |u| %s 1\nLOOKUP_TABLE default\n",
                                                  typestr
                                              ),
                                              buf, rank
            );

            my_offset = global_offset + displacements[rank] * sizeof(Real);
            MPI_File_write_at(fh, my_offset, point_data_mag.data(), point_data_mag.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += num_elem * sizeof(Real);
        }

        // Close the file.
        MPI_File_close(&fh);
    }


    void insertionSort(std::vector<Real>& coordinates, std::vector<Real>& u, std::vector<Real>& v, std::vector<Real>& w,
                       std::vector<Real>& p){
        int n = coordinates.size();
        for (int i = 1; i < n; i++){
            Real key = coordinates[i];
            Real key_u = u[i];
            Real key_v = v[i];
            Real key_w = w[i];
            Real key_p = p[i];
            int j = i - 1;
            while (j >= 0 && coordinates[j] > key){
                coordinates[j + 1] = coordinates[j];
                u[j + 1] = u[j];
                v[j + 1] = v[j];
                w[j + 1] = w[j];
                p[j + 1] = p[j];
                j = j - 1;
            }
            coordinates[j + 1] = key;
            u[j + 1] = key_u;
            v[j + 1] = key_v;
            w[j + 1] = key_w;
            p[j + 1] = key_p;
        }
    }

    //direction is 0 for x, 1 for y, 2 for z. this is the axis witch the line is parallel to
    // x,y,z are the coordinates of the point contained in the line
    void writeDat(
        const std::string& filename,
        const VelocityTensor& velocity,
        const Constants& constants,
        const StaggeredTensor& pressure,
        const int rank,
        const int mpisize,
        const int direction,
        const Real x, const Real y, const Real z
    ){
        //get the index of the point
        int i = (int)((x - constants.min_x_global) / constants.dx);
        int j = (int)((y - constants.min_y_global) / constants.dy);
        int k = (int)((z - constants.min_z_global) / constants.dz);

        MPI_File* fh;

        std::vector<Real> point_data_u, point_data_v, point_data_w, point_data_p;
        int size = constants.Nx * (direction == 0) + constants.Ny * (direction == 1) + constants.Nz * (direction == 2);
        point_data_u.reserve(size);
        point_data_v.reserve(size);
        point_data_w.reserve(size);
        point_data_p.reserve(size);
        std::vector<Real> points_coordinate;
        int base_j = constants.base_j + 1;
        int base_k = constants.base_k + 1;
        if (base_k == 1) base_k = 0; //TODO: check if this is correct
        if (base_j == 1) base_j = 0;
        if (direction == 0){
            //x axis
            //check if the point is in the domain using base_j and base_k

            for (int i = 0; i < constants.Nx; i++){
                if (base_j + constants.Ny_owner > j && base_j <= j && base_k + constants.Nz_owner > k && base_k <= k){
                    points_coordinate.push_back(i * constants.dx + constants.min_x_global);
                    point_data_u.push_back(
                        (velocity.u(i, j - base_j, k - base_k) + velocity.u(i + 1, j - base_j, k - base_k)) / 2);
                    point_data_v.push_back(
                        (velocity.v(i, j - base_j, k - base_k) + velocity.v(i, j - base_j + 1, k - base_k)) / 2);
                    point_data_w.push_back(
                        (velocity.w(i, j - base_j, k - base_k) + velocity.w(i, j - base_j, k - base_k + 1)) / 2);
                    point_data_p.push_back(pressure(i, j - base_j, k - base_k));
                }
            }
        }
        else if (direction == 1){
            for (int j = 0; j < constants.Ny_owner; j++){
                if (base_k + constants.Nz_owner > k && base_k <= k){
                    points_coordinate.push_back((j + base_j) * constants.dy + constants.min_y_global);
                    point_data_u.push_back(
                        (velocity.u(i, j, k - base_k) + velocity.u(i + 1, j, k - base_k)) / 2);
                    point_data_v.push_back(
                        (velocity.v(i, j, k - base_k) + velocity.v(i, j + 1, k - base_k)) / 2);
                    point_data_w.push_back(
                        (velocity.w(i, j, k - base_k) + velocity.w(i, j, k - base_k + 1)) / 2);
                    point_data_p.push_back(pressure(i, j, k - base_k));
                }
            }
        }
        else if (direction == 2){
            for (int z = 0; z < constants.Nz_owner; z++){
                if (base_j + constants.Ny_owner > j && base_j <= j){
                    points_coordinate.push_back((z+base_k) * constants.dz + constants.min_z_global);
                    point_data_u.push_back(
                        (velocity.u(i , j - base_j, z) + velocity.u(i  + 1, j - base_j, z)) / 2);
                    point_data_v.push_back(
                        (velocity.v(i , j - base_j, z) + velocity.v(i , j - base_j + 1, z)) / 2);
                    point_data_w.push_back(
                        (velocity.w(i , j - base_j, z) + velocity.w(i , j - base_j, z + 1)) / 2);
                    point_data_p.push_back(pressure(i , j - base_j, z));
                }
            }
        }
        size_t local_size = point_data_u.size();
        //send all the data to the first processor
        std::vector<int> counts(mpisize), displacements(mpisize);
        MPI_Gather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0){
            for (int i = 0; i < mpisize; i++){
                counts[i] *= sizeof(Real);
            }
        }
        if (rank == 0){
            displacements[0] = 0;
            for (int i = 0; i < mpisize; i++){
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }
        }

      for (int k = start_k_write_local; k < end_k_write_local; ++k) {
	for (int j = start_j_write_local; j < end_j_write_local; ++j) {
	  for (int i = start_i_write_local; i < end_i_write_local; ++i) {
	    out << ((velocity.v(i, j, k) + velocity.v(i, j + 1, k)) / 2) << ' ';
          }
	}
      }

        //alocate the space for the data
        std::vector<Real> point_data_u_global(std::accumulate(counts.begin(), counts.end(), 0) / 8);
        std::vector<Real> point_data_v_global(std::accumulate(counts.begin(), counts.end(), 0) / 8);
        std::vector<Real> point_data_w_global(std::accumulate(counts.begin(), counts.end(), 0) / 8);
        std::vector<Real> point_data_p_global(std::accumulate(counts.begin(), counts.end(), 0) / 8);
        std::vector<Real> points_coordinate_global(std::accumulate(counts.begin(), counts.end(), 0) / 8);


        if (rank == 0 || local_size > 0){
            std::cout << "Rank: " << rank << " Local size: " << local_size << std::endl;
            if (rank == 0){
                std::cout << "displacements: ";
                for (int i = 0; i < displacements.size(); i++){
                    std::cout << displacements[i] << " ";
                }
                std::cout << std::endl;
            }
            //send the data to the first processor
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
            MPI_Gatherv(points_coordinate.data(), local_size * sizeof(Real), MPI_BYTE, points_coordinate_global.data(),
                        counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        //sort the data based on the coordinates and write it to the file
        if (rank == 0){
            std::cout << "print pressure data" << std::endl;

            for (int i = 0; i < points_coordinate_global.size(); i++){
                std::cout << points_coordinate_global[i] << " " << point_data_u_global[i] << " " << point_data_v_global[i]
                          << " " << point_data_w_global[i] << " " << point_data_p_global[i] << std::endl;
            }
            insertionSort(points_coordinate_global, point_data_u_global, point_data_v_global, point_data_w_global,
                          point_data_p_global);
            std::cout << "Writing to file" << std::endl;
            //log some data to console
            std::cout << "Writing to file" << std::endl;
            std::cout << "Size: " << points_coordinate_global.size() << std::endl;
            std::cout << "Counts: " << counts.size() << std::endl;

            FILE* file = fopen(filename.c_str(), "w");
            if (direction == 0){
                for (int i = 0; i < points_coordinate_global.size(); i++){
                    fprintf(file, "%f %f %f %f %f %f %f\n", points_coordinate_global[i], y, z, point_data_u_global[i],
                            point_data_v_global[i], point_data_w_global[i], point_data_p_global[i]);
                }
            }
            else if (direction == 1){
                for (int i = 0; i < points_coordinate_global.size(); i++){
                    fprintf(file, "%f %f %f %f %f %f %f\n", x, points_coordinate_global[i], z, point_data_u_global[i],
                            point_data_v_global[i], point_data_w_global[i], point_data_p_global[i]);
                }
            }
            else if (direction == 2){
                for (int i = 0; i < points_coordinate_global.size(); i++){
                    fprintf(file, "%f %f %f %f %f %f %f\n", x, y, points_coordinate_global[i], point_data_u_global[i],
                            point_data_v_global[i], point_data_w_global[i], point_data_p_global[i]);
                }
            }
            fclose(file);
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

	    out << std::sqrt(ux*ux + uy*uy + uz*uz) << ' ';
          }
	}
      }

      out
        << "SCALARS p double 1\n"
	<< "LOOKUP_TABLE default\n";

      for (int k = start_k_write_local; k < end_k_write_local; ++k) {
	for (int j = start_j_write_local; j < end_j_write_local; ++j) {
	  for (int i = start_i_write_local; i < end_i_write_local; ++i) {
	    out << pressure(i,j,k) << ' ';
          }
	}
      }
    }
 void writeVTKFullMesh(const std::string&     filename,
			  const VelocityTensor&  velocity,
			  const StaggeredTensor& pressure) {
      std::ofstream out(filename);
      const Constants &constants = velocity.constants;
      const int size = constants.Py * constants.Pz;

      assert(size == 1);

      const size_t Nx = constants.Nx_global;
      const size_t Ny = (constants.y_rank == 0 && constants.periodic_bc[1]) ? constants.Ny_owner+1 : constants.Ny_owner;
      const size_t Nz = (constants.z_rank == 0 && constants.periodic_bc[2]) ? constants.Nz_owner+1 : constants.Nz_owner;
      const int start_i_write_local = constants.periodic_bc[0] ? 1 : 0;
      const int start_j_write_local = (constants.prev_proc_y == -1) ? 0 : 1;
      const int start_k_write_local = (constants.prev_proc_z == -1) ? 0 : 1;
      const int end_i_write_local = start_i_write_local + Nx;
      const int end_j_write_local = start_j_write_local + Ny;
      const int end_k_write_local = start_k_write_local + Nz;

      out
        << "# vtk DataFile Version 3.0\n"
	<< "pressure mesh solution\n"
	<< "ASCII\n"
	<< "DATASET STRUCTURED_POINTS\n"
	<< "DIMENSIONS " << Nx << ' ' << Ny << ' ' << Nz << '\n'
	<< "ORIGIN 0 0 0\n"
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

	    out << std::sqrt(ux*ux + uy*uy + uz*uz) << ' ';
          }
	}
      }

      out
        << "SCALARS p double 1\n"
	<< "LOOKUP_TABLE default\n";

      for (int k = start_k_write_local; k < end_k_write_local; ++k) {
	for (int j = start_j_write_local; j < end_j_write_local; ++j) {
	  for (int i = start_i_write_local; i < end_i_write_local; ++i) {
	    out << pressure(i,j,k) << ' ';
          }
	}
      }
    }

} // mif
