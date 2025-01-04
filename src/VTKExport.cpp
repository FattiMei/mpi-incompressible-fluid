#include <tuple>
#include <cstdio>
#include <numeric>
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

// computes the file cell offset as each processor writes data about its local points, the other need to know how much space he has occupied
// gives a number of cells, the caller knows how many data per cell
std::vector<int> compute_displacement(int n_local_points, int size) {
	std::vector<int> count(size);
	std::vector<int> displacements(size+1);

	MPI_Allgather(&n_local_points, 1, MPI_INT, count.data(), 1, MPI_INT, MPI_COMM_WORLD);

	displacements[0] = 0;
	for (int i = 0; i < size; ++i) {
		displacements[i+1] = displacements[i] + count[i];
	}

	return displacements;
}


MPI_Offset write_ascii_part(
	MPI_File fh,
	MPI_Offset global_offset,
	int strlen,
	char *buf,
	int rank
) {
	MPI_Status status;

	if (rank == 0) {
		MPI_File_write_at(fh, global_offset, buf, strlen, MPI_CHAR, &status);
	}

	return strlen;
}


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
	
        MPI_Offset my_offset;
	MPI_Offset global_offset = 0;
        MPI_File fh;
        MPI_Status status;
        MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                      MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        // int number_of_cells = constants.Nx * constants.Ny * constants.Nz * 3;
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

        {
            int x = 0; //x=0 plane
	    int index = 0;
            for (int y = 0; y < Ny_owner; y++){
                for (int z = 0; z < Nz_owner; z++){
                    points_coordinate.push_back(x * constants.dx);
                    points_coordinate.push_back((y + base_j) * constants.dy);
                    points_coordinate.push_back((z + base_k) * constants.dz);

                    // point_data_u.push_back(velocity.u(x, y, z));
                    point_data_u.push_back(index++);
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


	std::vector<int> displacements = compute_displacement(point_data_u.size(), size);
	std::vector<int> cell_displacements = compute_displacement((Ny_owner-1) * (Nz_owner-1), size);

	if (rank == 0) {
		for (int d : displacements) std::cout << d << ' ';
		std::cout << std::endl;
	}

        // Convert points_coordinate to big-endian format
        vectorToBigEndian(points_coordinate);
        vectorToBigEndian(point_data_u);
        vectorToBigEndian(point_data_v);
        vectorToBigEndian(point_data_w);
        vectorToBigEndian(point_data_p);

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

        MPI_File_write_at(fh, my_offset, points_coordinate.data(),
                          points_coordinate.size() * sizeof(Real),
                          MPI_BYTE, &status);

        int num_elem = displacements.back();
        global_offset += 3 * num_elem * sizeof(Real);

	// write information about the cells, I cheat and make the processor 0 write all things
	int n_cells_x_plane = (Ny_owner-1) * (Nz_owner-1);
	{
		global_offset += write_ascii_part(fh, global_offset,
			sprintf(
				buf,
				"\nCELLS %d %d\n",
				cell_displacements.back(),
				5 * cell_displacements.back()
			),
			buf, rank
		);

		std::vector<int> cell_data(5 * n_cells_x_plane, 4);
		int cell_count = 0;
		for (int y = 0; y < Ny_owner-1; ++y) {
			for (int z = 0; z < Nz_owner-1; ++z) {
				const int cell_offset = displacements[rank] + Nz_owner*y + z;

				cell_data[5*cell_count + 0] = 4;
				cell_data[5*cell_count + 1] = cell_offset;
				cell_data[5*cell_count + 2] = cell_offset + 1;
				cell_data[5*cell_count + 3] = cell_offset + 1 + Nz_owner;
				cell_data[5*cell_count + 4] = cell_offset + Nz_owner;

				++cell_count;
			}
		}
		vectorToBigEndian(cell_data);

		my_offset = global_offset + 5 * sizeof(int) * cell_displacements[rank];
		MPI_File_write_at(fh, my_offset, cell_data.data(), cell_data.size(), MPI_INT, &status);

		global_offset += 5 * cell_displacements.back() * sizeof(int);
	}
	{
		global_offset += write_ascii_part(
			fh,
			global_offset,
			sprintf(
				buf,
				"\nCELL_TYPES %d\n",
				cell_displacements.back()
			),
			buf, rank
		);

		std::vector<int> cell_type(n_cells_x_plane, 9);
		vectorToBigEndian(cell_type);

		my_offset = global_offset + sizeof(int) * cell_displacements[rank];
		MPI_File_write_at(fh, my_offset, cell_type.data(), cell_type.size(), MPI_INT, &status);

		global_offset += cell_displacements.back() * sizeof(int);
	}


        //now we write the u component of the velocity
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
		MPI_File_write_at(fh, my_offset, point_data_u.data(), point_data_u.size() * sizeof(Real), MPI_BYTE, &status);
		global_offset += num_elem * sizeof(Real);
	}

	// v velocity component
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
		MPI_File_write_at(fh, my_offset, point_data_v.data(), point_data_v.size() * sizeof(Real), MPI_BYTE, &status);
		global_offset += num_elem * sizeof(Real);
	}

	// w velocity component
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
		MPI_File_write_at(fh, my_offset, point_data_w.data(), point_data_w.size() * sizeof(Real), MPI_BYTE, &status);
		global_offset += (num_elem) * sizeof(Real);
	}


	// p component
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
		MPI_File_write_at(fh, my_offset, point_data_p.data(), point_data_p.size() * sizeof(Real), MPI_BYTE, &status);
	}


        // points offset and write data
        MPI_Barrier(MPI_COMM_WORLD); //Do I need this? Maybe not
        MPI_File_close(&fh);
}

};
