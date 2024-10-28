//
// Created by giorgio on 25/10/2024.
//

#include "vtk_parser.h"
#include <byteswap.h>

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 1024
#endif
#ifndef ENDIANESS
#define ENDIANESS 0 //0 for little endian, 1 for big endian
#endif




    //buffer as char array
    std::size_t buffer_cursor = 1024;
    size_t global_pos = 0;
namespace vtk {
    void read_line(std::string &line, char *data) {
        line = "";//TODO: return directly the pointer to the data at position, instead of copying it to a string 
        char c;
        while (data[global_pos] != '\n' && data[global_pos] != '\0') {//TODO: reverse this loop into a do while
            c = data[global_pos];
            line += c;
            global_pos++;
        }
        line += '\n';
        buffer_cursor++;
        global_pos++;
    }

    void parse(const std::string &filename, Tensor<> &velocity_u, Tensor<> &velocity_v, Tensor<> &velocity_w,
               Tensor<> &pressure) {
        //TODO:
        //  change this to create the tensors inside the function,
        //  since we shouldn't know the size of the tensors before parsing the file*/
        int fd = -1;
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1)
            throw std::runtime_error("Error opening file");
        struct stat st;
        fstat(fd, &st);
        std::size_t size = st.st_size;
        char *data = (char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED)
            throw std::runtime_error("Error mapping file");
        printf("Mapped file\n");

        std::string line;
        for (int i = 0; i < 4; i++) {
            read_line(line, data);
        }

        read_line(line, data);
        char *token = strtok((char *) line.c_str(), " ");
        token = strtok(NULL, " ");
        std::size_t Nx = std::atol(token);
        token = strtok(NULL, " ");
        std::size_t Ny = std::atol(token);
        token = strtok(NULL, " ");
        std::size_t Nz = std::atol(token);
        printf("Nx: %lu, Ny: %lu, Nz: %lu\n", Nx, Ny, Nz);
        read_line(line, data);
        token = strtok((char *) line.c_str(), " ");
        //discard first token "SPACING"
        token = strtok(NULL, " ");
        Real dx = std::atof(token);
        token = strtok(NULL, " ");
        Real dy = std::atof(token);
        token = strtok(NULL, " ");
        Real dz = std::atof(token);
        printf("dx: %f, dy: %f, dz: %f\n", dx, dy, dz);
        //discard origin
        read_line(line, data);
        read_line(line, data);
        token = strtok((char *) line.c_str(), " ");
        //discard first token "POINT_DATA"
        token = strtok(NULL, " ");
        std::size_t point_data = std::atol(token);
        if (point_data != Nx * Ny * Nz)
            throw std::runtime_error("Point data is not equal to Nx*Ny*Nz");
        //check if is float or double
        read_line(line, data);
        token = strtok((char *) line.c_str(), " ");
        token = strtok(NULL, " ");
        token = strtok(NULL, " ");
        int type_size = 4;
        if (strcmp(token, "float") == 0)
            type_size = 4;
        else if (strcmp(token, "double") == 0)
            type_size = 8;
        //discard lookup table
        read_line(line, data);        //close file
        close(fd);
        //parse data consider vtk endianess (big endian) unfurtunatly
        #if ENDIANESS == 0
        if (type_size == 4) {
            // Float parsing loop
            uint32_t *velocity_data = (uint32_t *) (data + global_pos);
            //velocity data is stored in a 3D matrix continuous in memory



            for (std::size_t i = 0; i < Nx; i++) {
                for (std::size_t j = 0; j < Ny; j++) {
                    for (std::size_t k = 0; k < Nz; k++) {
                        size_t pos = (i * Ny * Nz + j * Nz + k) * 3;
                        // X component
                        uint32_t x = velocity_data[pos];
                        Real x_real = static_cast<Real>(std::bit_cast<float>(bswap_32(
                                                                                     x))); //TODO: we can avoid the cast if we directly cast the pointer to float *, so somethjing like float_data = (float *) data + global_pos and then iterate with indixes instead of this mess
                        velocity_u(i, j, k) = (x_real);

                        // Y component
                        uint32_t y = velocity_data[pos + 1];
                        Real y_real = static_cast<Real>(std::bit_cast<float>(bswap_32(y)));
                        velocity_v(i, j, k) = y_real;

                        // Z component
                        uint32_t z = velocity_data[pos + 2];
                        Real z_real = static_cast<Real>(std::bit_cast<float>(bswap_32(z)));
                        velocity_w(i, j, k) = z_real;
                    }
                }
            }

        } else {
            uint64_t *velocity_data = (uint64_t *) (data + global_pos);

            // Double parsing loop
            for (std::size_t i = 0; i < Nx; i++) {
                for (std::size_t j = 0; j < Ny; j++) {
                    for (std::size_t k = 0; k < Nz; k++) {
                        size_t pos = (i * Ny * Nz + j * Nz + k) * 3;
                        // X component
                        uint64_t x = velocity_data[pos];
                        Real x_real = static_cast<Real>(std::bit_cast<double>(bswap_64(x)));
                        velocity_u(i, j, k) = x_real;

                        // Y component
                        uint64_t y = velocity_data[pos + 1];
                        Real y_real = static_cast<Real>(std::bit_cast<double>(bswap_64(y)));
                        velocity_v(i, j, k) = y_real;

                        // Z component
                        uint64_t z = velocity_data[pos + 2];
                        Real z_real = static_cast<Real>(std::bit_cast<double>(bswap_64(z)));
                        velocity_w(i, j, k) = z_real;
                    }
                }
            }
        }
        //TODO:
        // decide if we want to directly stagger the data in the tensor
        // or if we want to store it in the tensor as it is and stagger it later


        #else
        //TODO: implement big endian, probably not needed as the target machine will be le
        throw std::runtime_error("Big endian not implemented");
        #endif

        size_t pos = global_pos + 3 * type_size * (Nx) * (Ny) * (Nz) + 1;
        global_pos = pos;
        printf("Next line: %s\n", &data[pos]);

        //SCALARS pressure float 1
        read_line(line, data);
        //we need to discard the first token "SCALARS" and the second token "pressure" to get the type
        token = strtok((char *) line.c_str(), " ");
        type_size = 4;
        token = strtok(NULL, " ");
        token = strtok(NULL, " ");
        if (strcmp(token, "float") == 0)
            type_size = 4;
        else if (strcmp(token, "double") == 0)
            type_size = 8;

        //discard the next line "LOOKUP_TABLE default"
        read_line(line, data);
        //parse pressure

        if (type_size == 4) {
            uint32_t *pressure_data = (uint32_t *) (data + global_pos);
            for (std::size_t i = 0; i < Nx; i++) {
                for (std::size_t j = 0; j < Ny; j++) {
                    for (std::size_t k = 0; k < Nz; k++) {
                        uint32_t p = pressure_data[i * Ny * Nz + j * Nz +
                                                   k];//I really should check if this is correct in the vtk documentation
                        Real p_real = static_cast<Real>(std::bit_cast<float>(bswap_32(p)));
                        pressure(i, j, k) = p_real;
                    }
                }
            }
        } else {
            uint64_t *pressure_data = (uint64_t *) (data + global_pos);
            for (std::size_t i = 0; i < Nx; i++) {
                for (std::size_t j = 0; j < Ny; j++) {
                    for (std::size_t k = 0; k < Nz; k++) {
                        uint64_t p = pressure_data[i * Ny * Nz + j * Nz + k];
                        Real p_real = static_cast<Real>(std::bit_cast<double>(bswap_64(p)));
                        pressure(i, j, k) = p_real;
                    }
                }
            }
        }

        //print first and last value of pressure
        std::cout << "First value of pressure: " << pressure(0, 0, 0) << std::endl;
        std::cout << "Last value of pressure: " << pressure(Nx - 1, Ny - 1, Nz - 1) << std::endl;
        //unmap file
        munmap(data, size);
    }
}

