//
// Created by giorgio on 28/12/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H
#define MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H

#include <string>
#include <iostream>
#include <vector>
#include <byteswap.h>
#ifndef ENDIANESS
#define ENDIANESS 0 //0 for little endian, 1 for big endian
#endif


namespace mif{
    Real bitswap(Real value){
        if constexpr (ENDIANESS == 1){
            return value;
        }
        else{
            if constexpr (constexpr size_t size = sizeof(Real); size == 4){
                return bswap_32(value);
            }
            else{
                return bswap_64(value);
            }
        }
    }


    void writeVTK(const std::string& filename, VelocityTensor& velocity, const Constants & constants,
                  StaggeredTensor& pressure, int rank){
        MPI_Offset my_offset, my_current_offset;
        MPI_File fh;
        MPI_Status status;
        MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                      MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        int number_of_cells = constants.Nx * constants.Ny * constants.Nz * 3;
        int local_cells = 3 * (1 * constants.Ny_owner * constants.Nz_owner +
                           constants.Nx * 1 * constants.Nz_owner +
                           constants.Nx * constants.Ny_owner * 1);
        std::vector<Real> local_points(local_cells);
        std::vector<Real> local_data(local_cells);
        std::vector<Real> global_points;
        std::vector<Real> global_data;

        if (rank == 0) {
            global_points.resize( constants.Nx * constants.Ny * constants.Nz * 3+ constants.Nx * constants.Ny * 3);
            global_data.resize(constants.Nx * constants.Ny * constants.Nz * 3+ constants.Nx * constants.Ny * 3);
        }
        stringstream type;
        if constexpr (constexpr size_t _size = sizeof(Real); _size == 8){
            type << "double";
        }
        else
            type << "float";
        stringstream header;

        header << "# vtk DataFile Version 5.1\n";
        header << "vtk output\n";
        header << "BINARY\n";
        header << "DATASET UNSTRUCTURED_GRID\n";
        header << "POINTS " << number_of_cells << type.str() << "\n";
        int header_size = header.str().size();

        if (rank == 0){
            MPI_File_write(fh, header.str().c_str(), header_size, MPI_CHAR, &status);
        }


        int Nx = constants.Nx;
        int Ny = constants.Ny;
        int Nz = constants.Nz;

        int Ny_owner = constants.Ny_owner;
        int Nz_owner = constants.Nz_owner;

        // Start indices (base_j, base_k) based on rank
        int base_j = constants.base_j + 1;
        int base_k = constants.base_k + 1;
        std::vector<Real> points_coordinate(3 * (1 * Ny_owner * Nz_owner + Nx * 1 * Nz_owner + Nx * Ny_owner * 1));
        std::vector<Real> point_data(3 * (1 * Ny_owner * Nz_owner + Nx * 1 * Nz_owner + Nx * Ny_owner * 1));
        //my_offset = header_size + (base_j * Ny + base_k)
        {
            int x = 0; //x=0 plane
            for (int y = 0; y < Ny_owner; y++){
                for (int z = 0; z < Nz_owner; z++){
                    points_coordinate[3 * (y + z * Ny_owner)] = bitswap(x * constants.dx);
                    points_coordinate[3 * (y + z * Ny_owner) + 1] = bitswap((y + base_j) * constants.dy);
                    points_coordinate[3 * (y + z * Ny_owner) + 2] = bitswap((z + base_k) * constants.dz);


                    point_data[3 * (y + z * Ny_owner)] =
                        bitswap(
                            velocity.u(x, y, z) - 0.5 * (velocity.u(x + 1, y, z) - velocity.u(x + 2, y, z)) * (x == 0));

                    point_data[3 * (y + z * Ny_owner) + 1] =
                        bitswap(
                            velocity.v(x, y, z) - 0.5 * (velocity.v(x, y + 1, z) - velocity.v(x, y + 2, z)) * (y == 0));

                    point_data[3 * (y + z * Ny_owner) + 2] =
                        bitswap(
                            velocity.w(x, y, z) - 0.5 * (velocity.w(x, y, z + 1) - velocity.w(x, y, z + 2)) * (z == 0));

                    point_data[3 * (y + z * Ny_owner)] +=
                        bitswap(((velocity.u(x, y, z) + velocity.u(x - 1, y, z)) / 2.0) * (x != 0));

                    point_data[3 * (y + z * Ny_owner) + 1] +=
                        bitswap(((velocity.v(x, y, z) + velocity.v(x, y - 1, z)) / 2.0) * (y != 0));

                    point_data[3 * (y + z * Ny_owner) + 2] +=
                        bitswap(((velocity.w(x, y, z) + velocity.w(x, y, z - 1)) / 2.0) * (z != 0));
                }
            }
        }
        {
            int plane_offset = Nz_owner * Ny_owner;
            int y = 0; //y=0 plane
            for (int x = 0; x < Nx; x++)
                for (int z = 0; z < Nz_owner; z++){
                    int index = plane_offset + x * Nz_owner + z;
                    points_coordinate[3 * index] = bitswap(x * constants.dx);
                    points_coordinate[3 * index + 1] = bitswap((y + base_j) * constants.dy);
                    points_coordinate[3 * index + 2] = bitswap((z + base_k) * constants.dz);


                    point_data[3 * index] =
                        bitswap(
                            velocity.u(x, y, z) - 0.5 * (velocity.u(x + 1, y, z) - velocity.u(x + 2, y, z)) * (x == 0));

                    point_data[3 * index + 1] =
                        bitswap(
                            velocity.v(x, y, z) - 0.5 * (velocity.v(x, y + 1, z) - velocity.v(x, y + 2, z)) * (y == 0));

                    point_data[3 * index + 2] =
                        bitswap(
                            velocity.w(x, y, z) - 0.5 * (velocity.w(x, y, z + 1) - velocity.w(x, y, z + 2)) * (z == 0));

                    point_data[3 * index] +=
                        bitswap(((velocity.u(x, y, z) + velocity.u(x - 1, y, z)) / 2.0) * (x != 0));

                    point_data[3 * index + 1] +=
                        bitswap(((velocity.v(x, y, z) + velocity.v(x, y - 1, z)) / 2.0) * (y != 0));

                    point_data[3 * index + 2] +=
                        bitswap(((velocity.w(x, y, z) + velocity.w(x, y, z - 1)) / 2.0) * (z != 0));
                }
        }
        {
            int plane_offset = Nz_owner * Ny_owner + Nx * Nz_owner;
            for (int x = 0; x < Nx; x++)
                for (int y = 0; y < Ny_owner; y++){
                    int z = 0;
                    int index = plane_offset + x * Ny_owner + y;
                    points_coordinate[3 * index] = bitswap(x * constants.dx);
                    points_coordinate[3 * index + 1] = bitswap((y + base_j) * constants.dy);
                    points_coordinate[3 * index + 2] = bitswap((z + base_k) * constants.dz);


                    point_data[3 * index] =
                        bitswap(
                            velocity.u(x, y, z) - 0.5 * (velocity.u(x + 1, y, z) - velocity.u(x + 2, y, z)) * (x == 0));

                    point_data[3 * index + 1] =
                        bitswap(
                            velocity.v(x, y, z) - 0.5 * (velocity.v(x, y + 1, z) - velocity.v(x, y + 2, z)) * (y == 0));

                    point_data[3 * index + 2] =
                        bitswap(
                            velocity.w(x, y, z) - 0.5 * (velocity.w(x, y, z + 1) - velocity.w(x, y, z + 2)) * (z == 0));

                    point_data[3 * index] +=
                        bitswap(((velocity.u(x, y, z) + velocity.u(x - 1, y, z)) / 2.0) * (x != 0));

                    point_data[3 * index + 1] +=
                        bitswap(((velocity.v(x, y, z) + velocity.v(x, y - 1, z)) / 2.0) * (y != 0));

                    point_data[3 * index + 2] +=
                        bitswap(((velocity.w(x, y, z) + velocity.w(x, y, z - 1)) / 2.0) * (z != 0));
                }
        }


        local_data = point_data;
        local_points = points_coordinate;
        MPI_Gather(local_points.data(), local_cells, MPI_REAL,
               global_points.data(), local_cells, MPI_REAL, 0, MPI_COMM_WORLD);

        MPI_Gather(local_data.data(), local_cells, MPI_REAL,
                   global_data.data(), local_cells, MPI_REAL, 0, MPI_COMM_WORLD);


        if (rank == 0){
            MPI_File_write_at(fh, header_size, global_points.data(),
                              3 * (1 * Ny * Nz + Nx * 1 * Nz + Nx * Ny * 1) * sizeof(Real),
                              MPI_REAL, &status);
            //wirte look up table
            // MPI_File_write_at(fh, header_size + 3 * (1 * Ny * Nz + Nx * 1 * Nz + Nx * Ny * 1) * sizeof(Real),
            //                   global_data.data(),
            //                   3 * (1 * Ny * Nz + Nx * 1 * Nz + Nx * Ny * 1) * sizeof(Real),
            //                   MPI_REAL, &status);
        }





//         MPI_File_write_at(fh, my_offset, points_coordinate.data(),
//                           3 * (1 * Ny_owner * Nz_owner + Nx * 1 * Nz_owner + Nx * Ny_owner * 1) * sizeof(Real),
//                           MPI_REAL, &status);


        // stringstream point_data_header;
        // /*
        // POINT_DATA 27
        // SCALARS scalars float 1
        // LOOKUP_TABLE default
        // */
        //
        // point_data_header << "POINT_DATA " << number_of_cells << "\n";
        // point_data_header << "VECTORS velocity " << type.str() << " 3" << "\n";
        // point_data_header << "LOOKUP_TABLE velocity" << "\n";
        // int point_data_header_size = point_data_header.str().size();
        //
        //
        // my_offset =header_size + 3 * (1 * Ny_owner * Nz_owner + Nx * 1 * Nz_owner + Nx * Ny_owner * 1) * sizeof(Real) + rank * Nx * Ny_owner * Nz_owner * 3 * sizeof(Real);
        // if (rank == 0){
        //
        //
        //
        //
        //
        //
        //
        //
        // }


        // points offset and write data
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_close(&fh);
    }
} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H
