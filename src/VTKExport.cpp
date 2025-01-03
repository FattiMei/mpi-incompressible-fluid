#include <cstdio>
#include "Endian.h"
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
//     - for the parts where only the first processor writes data, the other ones
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
//     right now only "point gaussian" option is available and it's not good
//
// The desired structure of the vtk file is available at analysis/vtk/cell.vtk


namespace mif {

void writeVTK(
	const std::string&     filename,
	const VelocityTensor&  velocity,
	const Constants&       constants,
	const StaggeredTensor& pressure,
	const int              rank,
	const int              size
) {
	const char* typestr = (sizeof(Real) == 8) ? "double" : "float";
	char buf[1024];
	int bufsize = 0;
	
        MPI_Offset my_offset, my_current_offset;
        MPI_File fh;
        MPI_Status status;
        MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                      MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        int number_of_cells = constants.Nx * constants.Ny * constants.Nz * 3;
        int local_cells = 3 * (1 * constants.Ny_owner * constants.Nz_owner +
            constants.Nx * 1 * constants.Nz_owner +
            constants.Nx * constants.Ny_owner * 1);
        std::vector<Real> global_points;
        std::vector<Real> global_data;

        if (rank == 0){
            global_points.resize(constants.Nx * constants.Ny_global * 3 * 3 + constants.Nx * constants.Ny_global * 3);
            global_data.resize(constants.Nx * constants.Ny_global * 3 * 3 + constants.Nx * constants.Ny_global * 3);
        }

        int Nx = constants.Nx;
        int Ny = constants.Ny_global;
        int Nz = constants.Nz_global;

        int Ny_owner = constants.Ny_owner;
        int Nz_owner = constants.Nz_owner;

        // Start indices (base_j, base_k) based on rank
        int base_j = constants.base_j + 1;
        int base_k = constants.base_k + 1;
        if (base_k == 1) base_k = 0; //TODO: check if this is correct
        if (base_j == 1) base_j = 0;
        std::vector<Real> points_coordinate;
        std::vector<Real> point_data_u, point_data_v, point_data_w, point_data_p;
        //reserve space
        points_coordinate.reserve(local_cells * 3);
        point_data_u.reserve(local_cells);
        point_data_v.reserve(local_cells);
        point_data_w.reserve(local_cells);
        point_data_p.reserve(local_cells);
        //my_offset = header_size + (base_j * Ny + base_k)
        {
            int x = 0; //x=0 plane
            for (int y = 0; y < Ny_owner; y++){
                for (int z = 0; z < Nz_owner; z++){
                    points_coordinate.push_back(x * constants.dx);
                    points_coordinate.push_back((y + base_j) * constants.dy);
                    points_coordinate.push_back((z + base_k) * constants.dz);

                    point_data_u.push_back(velocity.u(x, y, z));
                    point_data_v.push_back(velocity.v(x, y, z));
                    point_data_w.push_back(velocity.w(x, y, z));
                    point_data_p.push_back(pressure(x, y, z));
                }
            }
        }
        if (base_j == 0){
            // ALLERT THIS ONLY IF AT THE BORDER
            int y = 0; //y=0 plane
            for (int x = 0; x < Nx; x++)
                for (int z = 0; z < Nz_owner; z++){
                    points_coordinate.push_back(x * constants.dx);
                    points_coordinate.push_back((y + base_j) * constants.dy);
                    points_coordinate.push_back((z + base_k) * constants.dz);

                    point_data_u.push_back(velocity.u(x, y, z));
                    point_data_v.push_back(velocity.v(x, y, z));
                    point_data_w.push_back(velocity.w(x, y, z));
                    point_data_p.push_back(pressure(x, y, z));
                }
        }
        if (base_k == 0){
            int z = 0;
            for (int x = 0; x < Nx; x++)
                for (int y = 0; y < Ny_owner; y++){
                    if (base_j == 0 && y == 0) continue;
                    points_coordinate.push_back(x * constants.dx);
                    points_coordinate.push_back((y + base_j) * constants.dy);
                    points_coordinate.push_back((z + base_k) * constants.dz);

                    point_data_u.push_back(velocity.u(x, y, z));
                    point_data_v.push_back(velocity.v(x, y, z));
                    point_data_w.push_back(velocity.w(x, y, z));
                    point_data_p.push_back(pressure(x, y, z));
                }
        }


        std::vector<int> counts(size), displacements(size);
        int local_size = points_coordinate.size();
        MPI_Gather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);


        if (rank == 0){
            displacements[0] = 0;
            for (int i = 1; i < size; ++i){
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }
            std::cout << "Displacements: ";
            for (int i = 0; i < size; i++){
                std::cout << displacements[i] << " ";
            }
            std::cout << std::endl;
        }
        MPI_Request request[2];

        // Initiate non-blocking broadcast for displacements and counts
        MPI_Ibcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Ibcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD, &request[1]);

        // Convert points_coordinate to big-endian format
        vectorToBigEndian(points_coordinate);
        vectorToBigEndian(point_data_u);
        vectorToBigEndian(point_data_v);
        vectorToBigEndian(point_data_w);
        vectorToBigEndian(point_data_p);

        // Wait for the non-blocking broadcasts to complete
        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        //BAD, I'dont know if I need this or not, shouldn't be too bad though, but I should check TODO check this

	bufsize = sprintf(
		buf,
		"# vtk DataFile Version 2.0\nvtk output\nBINARY\nDATASET UNSTRUCTURED_GRID \nPOINTS %d %s\n",
		(displacements[size - 1] + counts[size - 1]) / 3,
		typestr
	);

        if (rank == 0) {
            MPI_File_write(fh, buf, strlen(buf), MPI_CHAR, &status);
        }

        my_offset = bufsize + displacements[rank] * sizeof(Real);
        //write all arguments to console for debugging
        // std::cout << "Rank: " << rank << " Displacement: " << displacements[rank] << " My offset: " << my_offset
        //   << " Size of points_coordinate: " << points_coordinate.size() << " Size of point_data_u: "
        //   << point_data_u.size() << " Size of point_data_v: " << point_data_v.size() << " Size of point_data_w: "
        //   << point_data_w.size() << " Size of local_cells: " << local_cells << std::endl;

        MPI_File_write_at(fh, my_offset, points_coordinate.data(),
                          points_coordinate.size() * sizeof(Real),
                          MPI_BYTE, &status);
        //maybe I could use a non-blocking write here and then wait for all to finish at the end of the function TODO check this

        //now we write the u component of the velocity
        int num_elem = displacements[size - 1] + counts[size - 1];
        int global_offset = num_elem * sizeof(Real) + bufsize;
        {
	    int local_u_header_size = sprintf(
		buf,
		"\nPOINT_DATA %d\nSCALARS u %s 1\nLOOKUP_TABLE default\n",
		(displacements[size - 1] + counts[size - 1]) / 3,
		typestr
	    );

            if (rank == 0) {
                MPI_File_write_at(fh, global_offset, buf, local_u_header_size, MPI_CHAR, &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_u_header_size;
            std::cout << "rank " << rank << " my_offset: " << my_offset << " size: " << size << std::endl;
            MPI_File_write_at(fh, my_offset, point_data_u.data(), point_data_u.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += (displacements[size - 1] + counts[size - 1]) / 3 * sizeof(Real) + local_u_header_size;
        }


        {
	    int local_v_header_size = sprintf(
		buf,
		"\nSCALARS v %s 1\nLOOKUP_TABLE default\n",
		typestr
	    );

            if (rank == 0) {
                MPI_File_write_at(fh, global_offset, buf, local_v_header_size, MPI_CHAR, &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_v_header_size;
            MPI_File_write_at(fh, my_offset, point_data_v.data(), point_data_v.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += (displacements[size - 1] + counts[size - 1]) / 3 * sizeof(Real) + local_v_header_size;
        }

        {
	    int local_w_header_size = sprintf(
		buf,
		"\nSCALARS w %s 1\nLOOKUP_TABLE default\n",
		typestr
	    );

            if (rank == 0) {
                MPI_File_write_at(fh, global_offset, buf, local_w_header_size, MPI_CHAR, &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_w_header_size;
            MPI_File_write_at(fh, my_offset, point_data_w.data(), point_data_w.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += (displacements[size - 1] + counts[size - 1]) / 3 * sizeof(Real) + local_w_header_size;
        }
        {
	    int local_p_header_size = sprintf(
		buf,
		"\nSCALARS p %s 1\nLOOKUP_TABLE default\n",
		typestr
	    );

            if (rank == 0) {
                MPI_File_write_at(fh, global_offset, buf, local_p_header_size, MPI_CHAR, &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_p_header_size;
            MPI_File_write_at(fh, my_offset, point_data_p.data(), point_data_p.size() * sizeof(Real), MPI_BYTE,
                              &status);
        }


        // points offset and write data
        MPI_Barrier(MPI_COMM_WORLD); //Do I need this? Maybe not
        MPI_File_close(&fh);
}

};
