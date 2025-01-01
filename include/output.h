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


    /*void writeVTK(const std::string& filename, VelocityTensor& velocity, const Constants& constants,
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

        if (rank == 0){
            global_points.resize(constants.Nx * constants.Ny_global  * 3 *3+ constants.Nx * constants.Ny_global * 3);
            global_data.resize(constants.Nx * constants.Ny_global  * 3*3 + constants.Nx * constants.Ny_global * 3);
        }
        stringstream type;
        if constexpr (constexpr size_t _size = sizeof(Real); _size == 8){
            type << "double";
        }
        else
            type << "float";



        int Nx = constants.Nx;
        int Ny = constants.Ny_global;
        int Nz = constants.Nz_global;

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
                    points_coordinate[3 * (y + z * Ny_owner)] = bitswap(x);
                    points_coordinate[3 * (y + z * Ny_owner) + 1] = bitswap((y + base_j) );
                    points_coordinate[3 * (y + z * Ny_owner) + 2] = bitswap((z + base_k) );


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
                    points_coordinate[3 * index] = bitswap(x);
                    points_coordinate[3 * index + 1] = bitswap((y + base_j));
                    points_coordinate[3 * index + 2] = bitswap((z + base_k) );


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
                    points_coordinate[3 * index] = bitswap(x );
                    points_coordinate[3 * index + 1] = bitswap((y + base_j) );
                    points_coordinate[3 * index + 2] = bitswap((z + base_k) );


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
            //remove duplicates from global_points, duplicatesa are tripletes  of poits where x1==x2 and y1==y2 ecc  and the respective data TODO: this is BAD very BAD
             std::vector<int> indices;
             for (int i = 0; i < global_points.size(); i+=3){
                 bool found = false;
                    for (int j = 0; j < global_points.size(); j+=3){
                        if (i != j){
                            if (global_points[i] == global_points[j] && global_points[i + 1] == global_points[j + 1] && global_points[i + 2] == global_points[j + 2]){
                                found = true;
                                std::cout << "Found duplicate" << std::endl;
                                std::cout << "x: " << global_points[i] << " y: " << global_points[i + 1] << " z: " << global_points[i + 2] << std::endl;
                                std::cout << "x: " << global_points[j] << " y: " << global_points[j + 1] << " z: " << global_points[j + 2] << std::endl;

                                break;
                            }
                        }
                    }
                 if (!found){
                     indices.push_back(i);
                 }
             }
            std::cout << "Number of index: " << indices.size() << std::endl;
             std::vector<Real> new_points;
             std::vector<Real> new_data;
             for (int i = 0; i < global_points.size(); i+=3){
                 if (std::find(indices.begin(), indices.end(), i) != indices.end()){
                     new_points.push_back(global_points[i]);
                     new_points.push_back(global_points[i + 1]);
                     new_points.push_back(global_points[i + 2]);
                     new_data.push_back(global_data[i]);
                     new_data.push_back(global_data[i + 1]);
                     new_data.push_back(global_data[i + 2]);
                 }
             }
             global_points = new_points;
             global_data = new_data;
            stringstream header;

            header << "# vtk DataFile Version 5.1\n";
            header << "vtk output\n";
            header << "BINARY\n";
            header << "DATASET UNSTRUCTURED_GRID\n";
            header << "POINTS " << global_points.size()/3  << " " <<type.str() << "\n";
            std::cout << "Number of points: " << global_points.size() << std::endl;
            int header_size = header.str().size();

            MPI_File_write(fh, header.str().c_str(), header_size, MPI_CHAR, &status);

            MPI_File_write_at(fh, header_size, global_points.data(),
                              global_points.size()  * sizeof(Real),
                              MPI_CHAR, &status);
            std::cout << "Nx: " << constants.Nx  << std::endl;
            std::cout << "Ny: " << constants.Ny_global  << std::endl;
            std::cout << "Nx * Ny *3: " << constants.Nx * constants.Ny_global * 3 << std::endl;
            stringstream point_data_header;
            point_data_header << "\n" << "POINT_DATA " << global_points.size()/3 << "\n";
            point_data_header << "VECTORS velocity " << type.str() << " 3" << "\n";
            point_data_header << "LOOKUP_TABLE default" << "\n";
            int point_data_header_size = point_data_header.str().size();
            MPI_File_write_at(fh, header_size + global_points.size() *3 * sizeof(Real),
                              point_data_header.str().c_str(),
                              point_data_header_size, MPI_CHAR, &status);

            //wirte look up table
            MPI_File_write_at(fh, header_size +point_data_header_size + global_points.size() *3 * sizeof(Real),
                              global_data.data(),
                              global_points.size() * 3  * sizeof(Real),
                              MPI_CHAR, &status);
        }


        //         MPI_File_write_at(fh, my_offset, points_coordinate.data(),
        //                           3 * (1 * Ny_owner * Nz_owner + Nx * 1 * Nz_owner + Nx * Ny_owner * 1) * sizeof(Real),
        //                           MPI_REAL, &status);


        // stringstream point_data_header;
        // /*
        // POINT_DATA 27
        // SCALARS scalars float 1
        // LOOKUP_TABLE default
        // #1#
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
    }*/


    void writeVTK(const std::string& filename, VelocityTensor& velocity, const Constants& constants,
                  StaggeredTensor& pressure, int rank, int size){
        {
            MPI_File fh;
            MPI_Status status;
            MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                          MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

            int total_points = constants.Nx * constants.Ny_global * constants.Nz_global;

            // Prepare local arrays for velocities with interpolation
            std::vector<Real> local_u(constants.Nx * constants.Ny_owner * constants.Nz_owner);
            std::vector<Real> local_v(constants.Nx * constants.Ny_owner * constants.Nz_owner);
            std::vector<Real> local_w(constants.Nx * constants.Ny_owner * constants.Nz_owner);

            // Fill local arrays
            int index = 0;
            for (int x = 0; x < constants.Nx; x++){
                for (int y = 0; y < constants.Ny_owner; y++){
                    for (int z = 0; z < constants.Nz_owner; z++){
                        local_u[index] =
                            bitswap(
                                velocity.u(x, y, z) - 0.5 * (velocity.u(x + 1, y, z) - velocity.u(x + 2, y, z)) * (x ==
                                    0)) +
                            bitswap(((velocity.u(x, y, z) + velocity.u(x - 1, y, z)) / 2.0) * (x != 0));

                        local_v[index] =
                            bitswap(
                                velocity.v(x, y, z) - 0.5 * (velocity.v(x, y + 1, z) - velocity.v(x, y + 2, z)) * (y ==
                                    0)) +
                            bitswap(((velocity.v(x, y, z) + velocity.v(x, y - 1, z)) / 2.0) * (y != 0));

                        local_w[index] =
                            bitswap(
                                velocity.w(x, y, z) - 0.5 * (velocity.w(x, y, z + 1) - velocity.w(x, y, z + 2)) * (z ==
                                    0)) +
                            bitswap(((velocity.w(x, y, z) + velocity.w(x, y, z - 1)) / 2.0) * (z != 0));

                        index++;
                    }
                }
            }

            // Calculate local array sizes
            int local_size = local_u.size();

            // Gather sizes from all ranks
            std::vector<int> counts(size), displacements(size);
            MPI_Gather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);


            if (rank == 0){
                displacements[0] = 0;
                for (int i = 1; i < size; ++i){
                    displacements[i] = displacements[i - 1] + counts[i - 1];
                }
            }

            std::vector<Real> global_u, global_v, global_w;
            if (rank == 0){
                global_u.resize(displacements[size - 1] + counts[size - 1], 0.0);
                global_v.resize(displacements[size - 1] + counts[size - 1], 0.0);
                global_w.resize(displacements[size - 1] + counts[size - 1], 0.0);
            }

            MPI_Gatherv(local_u.data(), local_size, MPI_DOUBLE, global_u.data(),
                        counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            MPI_Gatherv(local_v.data(), local_size, MPI_DOUBLE, global_v.data(),
                        counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            MPI_Gatherv(local_w.data(), local_size, MPI_DOUBLE, global_w.data(),
                        counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0){
                std::stringstream header;
                header << "# vtk DataFile Version 3.0\n";
                header << "Velocity field\n";
                header << "BINARY\n";
                header << "DATASET STRUCTURED_GRID\n";
                header << "DIMENSIONS " << constants.Nx << " " << constants.Ny_global << " " << constants.Nz_global <<
                    "\n";
                header << "POINTS " << total_points << " float\n";
                MPI_File_write(fh, header.str().c_str(), header.str().size(), MPI_CHAR, &status);
                std::stringstream scalars_u;
                scalars_u << "\nPOINT_DATA 32768\nSCALARS scalars double 1\nLOOKUP_TABLE default\n";
                MPI_File_write(fh, scalars_u.str().c_str(), scalars_u.str().size(), MPI_CHAR, &status);
                std::cout << "Size of global_u: " << global_u.size() << std::endl;
                MPI_File_write(fh, global_u.data(), global_u.size() * sizeof(Real), MPI_CHAR, &status);

                std::stringstream scalars_v;
                scalars_v << "\nPOINT_DATA 32768\nSCALARS v double 1\nLOOKUP_TABLE default\n";
                MPI_File_write(fh, scalars_v.str().c_str(), scalars_v.str().size(), MPI_CHAR, &status);
                MPI_File_write(fh, global_v.data(), global_v.size() * sizeof(Real), MPI_CHAR, &status);

                std::stringstream scalars_w;
                scalars_w << "\nPOINT_DATA 32768\nSCALARS w double 1\nLOOKUP_TABLE default\n";
                MPI_File_write(fh, scalars_w.str().c_str(), scalars_w.str().size(), MPI_CHAR, &status);
                MPI_File_write(fh, global_w.data(), global_w.size() * sizeof(Real), MPI_CHAR, &status);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_File_close(&fh);
        }
    }
} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H
