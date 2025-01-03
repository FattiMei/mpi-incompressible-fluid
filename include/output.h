//
// Created by giorgio on 28/12/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H
#define MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H

#include <string>
#include <iostream>
#include <vector>
#include "Endian.h"
#ifndef ENDIANESS
#define ENDIANESS 0 //0 for little endian, 1 for big endian
#endif


namespace mif{
    void writeVTK(const std::string& filename, VelocityTensor& velocity, const Constants& constants,
                  StaggeredTensor& pressure, int rank, int size){
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
        std::stringstream type;
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
                    //TODO: push back is slow should use index arithmetic instead with resize
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
            for (int x = 1; x < Nx; x++)
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
            for (int x = 1; x < Nx; x++)
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

        std::stringstream header;
        header << "# vtk DataFile Version 2.0\n";
        header << "vtk output\n";
        header << "BINARY\n";
        header << "DATASET UNSTRUCTURED_GRID \n";
        header << "POINTS " << (displacements[size - 1] + counts[size - 1]) / 3 << " " << type.str() << "\n";
        int header_size = header.str().size();
        if (rank == 0){
            MPI_File_write(fh, header.str().c_str(), header_size, MPI_CHAR, &status);
            for (int point = 0; point < points_coordinate.size(); point = point + 3){
                // Real x =  correct_endianness<Real>(points_coordinate[point]);
                // Real y = correct_endianness<Real>(points_coordinate[point + 1]);
                // Real z = correct_endianness<Real>(points_coordinate[point + 2]);
                // std::cout << "Point: " << x << " " << y << " " << z << std::endl;
            }
        }
        my_offset = header_size + displacements[rank] * sizeof(Real);
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
        int global_offset = num_elem * sizeof(Real) + header_size;
        {
            std::stringstream local_u_header;
            local_u_header << "\nPOINT_DATA " << (displacements[size - 1] + counts[size - 1]) / 3 << "\n";
            local_u_header << "SCALARS u " << type.str() << " 1\n";
            local_u_header << "LOOKUP_TABLE default\n";
            int local_u_header_size = local_u_header.str().size();
            if (rank == 0){
                MPI_File_write_at(fh, global_offset, local_u_header.str().c_str(), local_u_header_size, MPI_CHAR,
                                  &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_u_header_size;
            std::cout << "rank " << rank << " my_offset: " << my_offset << " size: " << size << std::endl;
            MPI_File_write_at(fh, my_offset, point_data_u.data(), point_data_u.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += (displacements[size - 1] + counts[size - 1]) / 3 * sizeof(Real) + local_u_header_size;
        }


        {
            std::stringstream local_v_header;
            local_v_header << "\nSCALARS v " << type.str() << " 1\n";
            local_v_header << "LOOKUP_TABLE default\n";
            int local_v_header_size = local_v_header.str().size();
            if (rank == 0){
                MPI_File_write_at(fh, global_offset, local_v_header.str().c_str(), local_v_header_size, MPI_CHAR,
                                  &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_v_header_size;
            MPI_File_write_at(fh, my_offset, point_data_v.data(), point_data_v.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += (displacements[size - 1] + counts[size - 1]) / 3 * sizeof(Real) + local_v_header_size;
        }

        {
            std::stringstream local_w_header;
            local_w_header << "\nSCALARS w " << type.str() << " 1\n";
            local_w_header << "LOOKUP_TABLE default\n";
            int local_w_header_size = local_w_header.str().size();
            if (rank == 0){
                MPI_File_write_at(fh, global_offset, local_w_header.str().c_str(), local_w_header_size, MPI_CHAR,
                                  &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_w_header_size;
            MPI_File_write_at(fh, my_offset, point_data_w.data(), point_data_w.size() * sizeof(Real), MPI_BYTE,
                              &status);
            global_offset += (displacements[size - 1] + counts[size - 1]) / 3 * sizeof(Real) + local_w_header_size;
        }
        {
            std::stringstream local_p_header;
            local_p_header << "\nSCALARS p " << type.str() << " 1\n";
            local_p_header << "LOOKUP_TABLE default\n";
            int local_p_header_size = local_p_header.str().size();
            if (rank == 0){
                MPI_File_write_at(fh, global_offset, local_p_header.str().c_str(), local_p_header_size, MPI_CHAR,
                                  &status);
            }

            my_offset = global_offset + displacements[rank] / 3 * sizeof(Real) + local_p_header_size;
            MPI_File_write_at(fh, my_offset, point_data_p.data(), point_data_p.size() * sizeof(Real), MPI_BYTE,
                              &status);
        }


        // points offset and write data
        MPI_Barrier(MPI_COMM_WORLD); //Do I need this? Maybe not
        MPI_File_close(&fh);
    }



    void writeSerialVTK(const std::string& filename, VelocityTensor& velocity, const Constants& constants,
                        StaggeredTensor& pressure, int rank){
        {
            FILE* fh = fopen(filename.c_str(), "wb");
            if (fh == nullptr){
                std::cerr << "Error opening file" << std::endl;
                return;
            }
            std::vector<Real> local_u, local_v, local_w, local_p;


            local_u.reserve(constants.Nx * constants.Ny_global * constants.Nz_global);
            local_v.reserve(constants.Nx * constants.Ny_global * constants.Nz_global);
            local_w.reserve(constants.Nx * constants.Ny_global * constants.Nz_global);
            local_p.reserve(constants.Nx * constants.Ny_global * constants.Nz_global);

            for (int z = 0; z < constants.Nz_global; z++){
                for (int y = 0; y < constants.Ny_global; y++){
                    for (int x = 0; x < constants.Nx; x++){
                        //TODO preallocate
                        local_u.push_back(velocity.u(x, y, z));
                        local_v.push_back(velocity.v(x, y, z));
                        local_w.push_back(velocity.w(x, y, z));
                        local_p.push_back(pressure(x, y, z));
                    }
                }
            }


            vectorToBigEndian(local_u);
            vectorToBigEndian(local_v);
            vectorToBigEndian(local_w);
            vectorToBigEndian(local_p);


            if (rank == 0){
                std::stringstream header;
                header << "# vtk DataFile Version 2.0\n";
                header << "Velocity field\n";
                header << "BINARY\n";
                header << "DATASET STRUCTURED_POINTS\n";
                header << "DIMENSIONS " << constants.Nx << " " << constants.Ny_global << " " << constants.Nz_global <<
                    "\n";
                header << "ORIGIN 0 0 0\n";
                header << "SPACING " << constants.dx << " " << constants.dy << " " << constants.dz << "\n";
                // MPI_File_write(fh, header.str().c_str(), header.str().size(), MPI_CHAR, &status);
                fwrite(header.str().c_str(), sizeof(char), header.str().size(), fh);
                std::stringstream scalars_u;
                scalars_u << "\nPOINT_DATA " << constants.Nx * constants.Ny_global * constants.Nz_global <<
                    " \nSCALARS u double 1\nLOOKUP_TABLE default\n";
                // MPI_File_write(fh, scalars_u.str().c_str(), scalars_u.str().size(), MPI_CHAR, &status);
                fwrite(scalars_u.str().c_str(), sizeof(char), scalars_u.str().size(), fh);
                // MPI_File_write(fh, global_u.data(), global_u.size() * sizeof(Real), MPI_CHAR, &status);
                fwrite(local_u.data(), sizeof(Real), local_u.size(), fh);
                std::stringstream scalars_v;
                scalars_v << "\nSCALARS v double 1\nLOOKUP_TABLE default\n";
                // MPI_File_write(fh, scalars_v.str().c_str(), scalars_v.str().size(), MPI_CHAR, &status);
                // MPI_File_write(fh, global_v.data(), global_v.size() * sizeof(Real), MPI_CHAR, &status);
                fwrite(scalars_v.str().c_str(), sizeof(char), scalars_v.str().size(), fh);
                fwrite(local_v.data(), sizeof(Real), local_v.size(), fh);

                std::stringstream scalars_w;
                scalars_w << "\nSCALARS w double 1\nLOOKUP_TABLE default\n";
                // MPI_File_write(fh, scalars_w.str().c_str(), scalars_w.str().size(), MPI_CHAR, &status);
                // MPI_File_write(fh, global_w.data(), global_w.size() * sizeof(Real), MPI_CHAR, &status);
                // std::cout << global_w.size() << std::endl;
                fwrite(scalars_w.str().c_str(), sizeof(char), scalars_w.str().size(), fh);
                fwrite(local_w.data(), sizeof(Real), local_w.size(), fh);
                std::stringstream scalars_p;
                scalars_p << "\nSCALARS p double 1\nLOOKUP_TABLE default\n";
                // MPI_File_write(fh, scalars_w.str().c_str(), scalars_w.str().size(), MPI_CHAR, &status);
                // MPI_File_write(fh, global_w.data(), global_w.size() * sizeof(Real), MPI_CHAR, &status);
                // std::cout << global_w.size() << std::endl;
                fwrite(scalars_p.str().c_str(), sizeof(char), scalars_p.str().size(), fh);
                fwrite(local_p.data(), sizeof(Real), local_p.size(), fh);
            }

            fclose(fh);
        }
    }
} // mif

#endif //MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H
