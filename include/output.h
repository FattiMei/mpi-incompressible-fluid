//
// Created by giorgio on 28/12/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H
#define MPI_INCOMPRESSIBLE_FLUID_OUTPUT_H

#include <bit>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#ifndef ENDIANESS
#define ENDIANESS 0 //0 for little endian, 1 for big endian
#endif


namespace mif{
    template <std::floating_point Type>
    constexpr Type correct_endianness(const Type x) noexcept{
        if constexpr (std::endian::native == std::endian::little){
            if constexpr (std::is_same_v<Type, float>){
                const auto transmute = std::bit_cast<std::uint32_t>(x);
                auto swapped = std::byteswap(transmute);
                return std::bit_cast<Type>(swapped);
            }
            else if constexpr (std::is_same_v<Type, double>){
                const auto transmute = std::bit_cast<std::uint64_t>(x);
                auto swapped = std::byteswap(transmute);
                return std::bit_cast<Type>(swapped);
            }
        }
        return x;
    }

    double toBigEndian(double x){
        uint64_t transmute = *reinterpret_cast<uint64_t*>(&x);
        uint64_t swapped = std::byteswap(transmute);

        return *reinterpret_cast<double*>(&swapped);
    }

    void vectorToBigEndian(std::vector<Real>& xs){
        for (Real& x : xs) x = correct_endianness<Real>(x);
    }


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
        std::vector<Real> local_points(local_cells);
        std::vector<Real> local_data(local_cells);
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
        std::vector<Real> point_data_u, point_data_v, point_data_w;
        //reserve space
        points_coordinate.reserve(local_cells * 3);
        point_data_u.reserve(local_cells);
        point_data_v.reserve(local_cells);
        point_data_w.reserve(local_cells);
        //my_offset = header_size + (base_j * Ny + base_k)
        {
            int x = 0; //x=0 plane
            for (int y = 0; y < Ny_owner; y++){
                for (int z = 0; z < Nz_owner; z++){
                    points_coordinate.push_back(x);
                    points_coordinate.push_back(y + base_j);
                    points_coordinate.push_back(z + base_k);
                    point_data_u.push_back(velocity.u(x, y, z));
                    point_data_v.push_back(velocity.v(x, y, z));
                    point_data_w.push_back(velocity.w(x, y, z));
                }
            }
        }
        {
            int y = 0; //y=0 plane
            for (int x = 1; x < Nx; x++)
                for (int z = 0; z < Nz_owner; z++){
                    points_coordinate.push_back(x);
                    points_coordinate.push_back(y + base_j);
                    points_coordinate.push_back(z + base_k);

                    point_data_u.push_back(velocity.u(x, y, z));
                    point_data_v.push_back(velocity.v(x, y, z));
                    point_data_w.push_back(velocity.w(x, y, z));
                }
        }
        {
            for (int x = 1; x < Nx; x++)
                for (int y = 1; y < Ny_owner; y++){
                    int z = 0;
                    points_coordinate.push_back(x);
                    points_coordinate.push_back(y + base_j);
                    points_coordinate.push_back(z + base_k);

                    point_data_u.push_back(velocity.u(x, y, z));
                    point_data_v.push_back(velocity.v(x, y, z));
                    point_data_w.push_back(velocity.w(x, y, z));
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
        }

        // Broadcast the displacements to all processes
        MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        //wait for recive
        MPI_Barrier(MPI_COMM_WORLD); //BAD I GUESS

        //now we can write the data to the file
        //first we write the header of the file the the points coordinates as unstructured points

        std::stringstream header;
        header << "# vtk DataFile Version 3.0\n";
        header << "vtk output\n";
        header << "BINARY\n";
        header << "DATASET UNSTRUCTURED_GRID \n";
        header << "POINTS " << points_coordinate.size() / 3 << " " << type.str() << "\n";
        int header_size = header.str().size();
        if (rank == 0){
            MPI_File_write(fh, header.str().c_str(), header_size, MPI_CHAR, &status);
        }
        vectorToBigEndian(points_coordinate);
        my_offset = header_size + displacements[rank] * sizeof(Real);
        //write all arguments to console for debugging
        std::cout << "Rank: " << rank << std::endl;
        std::cout << "Displacement: " << displacements[rank] << std::endl;
        std::cout << "My offset: " << my_offset << std::endl;
        std::cout << "Size of points_coordinate: " << points_coordinate.size() << std::endl;
        std::cout << "Size of point_data_u: " << point_data_u.size() << std::endl;
        std::cout << "Size of point_data_v: " << point_data_v.size() << std::endl;
        std::cout << "Size of point_data_w: " << point_data_w.size() << std::endl;
        std::cout << "Size of local_cells: " << local_cells << std::endl;

        MPI_File_write_at(fh, my_offset, points_coordinate.data(),
                          points_coordinate.size() * sizeof(Real),
                          MPI_CHAR, &status);


        // points offset and write data
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_close(&fh);
    }


    /*void writeVTK(const std::string& filename, VelocityTensor& velocity, const Constants& constants,
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


            int index = 0;
            for (int x = 0; x < constants.Nx; x++){
                // For first processor (rank == 0), include the full range (with ghost points).
                // For other processors, start from index 1, as the first row/column is a ghost point.
                int y_start = (rank == 0) ? 0 : 1;
                int y_end = constants.Ny_owner; // This is the local size without the ghost points

                int z_start = (rank == 0) ? 0 : 1;
                int z_end = constants.Nz_owner;

                for (int y = y_start; y < y_end; y++){
                    for (int z = z_start; z < z_end; z++){
                        // Calculate global indices with base_j and base_k
                        int global_y = y + constants.base_j; // Adjust for processor offset in y-direction
                        int global_z = z + constants.base_k;
                        // Adjust for processor offset in z-direction DO I NEED IT? maybve for reconstruction

                        // Interpolations
                        {
                            local_u[index] = (x == 0)
                                                 ? (velocity.u(x, y, z) - 0.5 * (velocity.u(x + 1, y, z) - velocity.
                                                     u(x + 2, y, z)))
                                                 : (velocity.u(x, y, z) + velocity.u(x - 1, y, z)) / 2;

                            local_v[index] = (y == 0)
                                                 ? (velocity.v(x, y, z) - 0.5 * (velocity.v(x, y + 1, z) - velocity.
                                                     v(x, y + 2, z)))
                                                 : ((velocity.v(x, y, z) + velocity.v(x, y - 1, z)) / 2.0);

                            local_w[index] = (z == 0)
                                                 ? (velocity.w(x, y, z) - 0.5 * (velocity.w(x, y, z + 1) - velocity.
                                                     w(x, y, z + 2)))
                                                 : ((velocity.w(x, y, z) + velocity.w(x, y, z - 1)) / 2.0);
                        }
                        // Increment index for next local point
                        index++;
                    }
                }
            }


            if (rank == 0){
                std::cout << "Size of local_u: " << local_u.size() << std::endl;
                std::cout << "Size of local_v: " << local_v.size() << std::endl;
                std::cout << "Size of local_w: " << local_w.size() << std::endl;
            }


            assert(std::ranges::any_of(local_u.begin(), local_u.end(), [](auto x) { return x != 0.0; }));
            assert(std::ranges::any_of(local_v.begin(), local_v.end(), [](auto x) { return x != 0.0; }));
            assert(std::ranges::any_of(local_w.begin(), local_w.end(), [](auto x) { return x != 0.0; }));

            // vtk format requires big endian data (x86 arch is LITTLE endian)

            // I think the bug lies in the conversion to big endian
            vectorToBigEndian(local_u);
            vectorToBigEndian(local_v);
            vectorToBigEndian(local_w);


            // Calculate local array sizes
            int local_size = local_u.size();

            // Gather sizes from all ranks
            std::vector<int> counts(size), displacements(size);
            MPI_Gather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);


            // maybe every processor needs to know the displacements?
            if (rank == 0){
                displacements[0] = 0;
                for (int i = 1; i < size; ++i){
                    displacements[i] = displacements[i - 1] + counts[i - 1];
                }
            }

            // Broadcast the displacements to all processes
            MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

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
                header << "# vtk DataFile Version 2.0\n";
                header << "Velocity field\n";
                header << "BINARY\n";
                header << "DATASET STRUCTURED_POINTS\n";
                header << "DIMENSIONS " << constants.Nx << " " << constants.Ny_global << " " << constants.Nz_global <<
                    "\n";
                header << "ORIGIN 0 0 0\n";
                header << "SPACING " << constants.dx << " " << constants.dy << " " << constants.dz << "\n";
                MPI_File_write(fh, header.str().c_str(), header.str().size(), MPI_CHAR, &status);
                std::stringstream scalars_u;
                scalars_u << "\nPOINT_DATA 32768\nSCALARS u double 1\nLOOKUP_TABLE default\n";
                MPI_File_write(fh, scalars_u.str().c_str(), scalars_u.str().size(), MPI_CHAR, &status);
                std::cout << "Size of global_u: " << global_u.size() << std::endl;
                MPI_File_write(fh, global_u.data(), global_u.size() * sizeof(Real), MPI_CHAR, &status);

                std::stringstream scalars_v;
                scalars_v << "\nSCALARS v double 1\nLOOKUP_TABLE default\n";
                MPI_File_write(fh, scalars_v.str().c_str(), scalars_v.str().size(), MPI_CHAR, &status);
                MPI_File_write(fh, global_v.data(), global_v.size() * sizeof(Real), MPI_CHAR, &status);

                std::stringstream scalars_w;
                scalars_w << "\nSCALARS w double 1\nLOOKUP_TABLE default\n";
                MPI_File_write(fh, scalars_w.str().c_str(), scalars_w.str().size(), MPI_CHAR, &status);
                MPI_File_write(fh, global_w.data(), global_w.size() * sizeof(Real), MPI_CHAR, &status);
                std::cout << global_w.size() << std::endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_File_close(&fh);
        }
    }*/

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
