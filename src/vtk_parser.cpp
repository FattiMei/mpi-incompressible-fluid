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
               Tensor<> &pressure) { /*TODO:
            change this to create the tensors inside the function,
         since we shouldn't know the size of the tensors before parsing the file*/
        //open file
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
        for (std::size_t i = 0; i < Nx - 1; i++) {
            for (std::size_t j = 0; j < Ny - 1; j++) {
                for (std::size_t k = 0; k < Nz - 1; k++) {
                    //data position
                    size_t pos = (i * Ny * Nz + j * Nz + k) * 3 * type_size + global_pos;
                    //parse x component
                    if (type_size == 4) {
                        uint32_t x = *(uint32_t *) (data + pos);
                        x = bswap_32(x);
                        uint32_t xplusone = *(uint32_t *) (data + pos + 4 * 3);
                        xplusone = bswap_32(xplusone);
                        Real x_real = static_cast<Real>(std::bit_cast<float>(x));
                        Real xplusone_real = static_cast<Real>(std::bit_cast<float>(xplusone));
                        Real avg = (x_real + xplusone_real) * 0.5;
                        velocity_u(i, j,
                                   k) = avg;
                    } else {
                        uint64_t x = *(uint64_t *) (data + pos);
                        x = bswap_64(x);
                        uint64_t xplusone = *(uint64_t *) (data + pos + 8 * 3);
                        xplusone = bswap_64(xplusone);
                        Real x_real = static_cast<Real>(std::bit_cast<double>(x));
                        Real xplusone_real = static_cast<Real>(std::bit_cast<double>(xplusone));
                        Real avg = (x_real + xplusone_real) * 0.5;
                        velocity_u(i, j, k) = avg;

                    }
                    //parse y component
                    if (type_size == 4) {
                        uint32_t y = *(uint32_t *) (data + pos + 4);
                        y = bswap_32(y);
                        uint32_t yplusone = *(uint32_t *) (data + pos + 4 * 3 + 4);
                        yplusone = bswap_32(yplusone);
                        Real y_real = static_cast<Real>(std::bit_cast<float>(y));
                        Real yplusone_real = static_cast<Real>(std::bit_cast<float>(yplusone));
                        Real avg = (y_real + yplusone_real) * 0.5;
                        velocity_v(i, j, k) = avg;
                    } else {
                        uint64_t y = *(uint64_t *) (data + pos + 8);
                        y = bswap_64(y);
                        uint64_t yplusone = *(uint64_t *) (data + pos + 8 * 3 + 8);
                        yplusone = bswap_64(yplusone);
                        Real y_real = static_cast<Real>(std::bit_cast<double>(y));
                        Real yplusone_real = static_cast<Real>(std::bit_cast<double>(yplusone));
                        Real avg = (y_real + yplusone_real) * 0.5;
                        velocity_v(i, j, k) = avg;

                    }
                    //parse z component
                    if (type_size == 4) {
                        uint32_t z = *(uint32_t *) (data + pos + 8);
                        z = bswap_32(z);
                        uint32_t zplusone = *(uint32_t *) (data + pos + 4 * 3 + 8);
                        zplusone = bswap_32(zplusone);
                        Real z_real = static_cast<Real>(std::bit_cast<float>(z));
                        Real zplusone_real = static_cast<Real>(std::bit_cast<float>(zplusone));
                        Real avg = (z_real + zplusone_real) * 0.5;
                        velocity_w(i, j, k) = avg;

                    } else {
                        uint64_t z = *(uint64_t *) (data + pos + 16);
                        z = bswap_64(z);
                        uint64_t zplusone = *(uint64_t *) (data + pos + 8 * 3 + 16);
                        zplusone = bswap_64(zplusone);
                        Real z_real = static_cast<Real>(std::bit_cast<double>(z));
                        Real zplusone_real = static_cast<Real>(std::bit_cast<double>(zplusone));
                        Real avg = (z_real + zplusone_real) * 0.5;
                        velocity_w(i, j, k) = avg;

                    }
                }
            }
        }
        //last pass for u component
        for (std::size_t i = Nx - 2; i < Nx - 1; ++i) {
            for (std::size_t j = 0; j < Ny; j++) {
                for (std::size_t k = 0; k < Nz; k++) {
                    size_t pos = (i * Ny * Nz + j * Nz + k) * 3 * type_size + global_pos;
                    //parse x component
                    if (type_size == 4) {
                        uint32_t x = *(uint32_t *) (data + pos);
                        x = bswap_32(x);
                        uint32_t xplusone = *(uint32_t *) (data + pos + 4 * 3);
                        xplusone = bswap_32(xplusone);
                        Real x_real = static_cast<Real>(std::bit_cast<float>(x));
                        Real xplusone_real = static_cast<Real>(std::bit_cast<float>(xplusone));
                        Real avg = (x_real + xplusone_real) * 0.5;


                        velocity_u(i, j,
                                   k) = avg;
                    } else {
                        uint64_t x = *(uint64_t *) (data + pos);
                        x = bswap_64(x);
                        uint64_t xplusone = *(uint64_t *) (data + pos + 8 * 3);
                        xplusone = bswap_64(xplusone);
                        Real x_real = static_cast<Real>(std::bit_cast<double>(x));
                        Real xplusone_real = static_cast<Real>(std::bit_cast<double>(xplusone));
                        Real avg = (x_real + xplusone_real) * 0.5;
                        velocity_u(i, j, k) = avg;
                    }
                }
            }
        }

        //last pass for v component
        for (std::size_t i = 0; i < Nx; ++i) {
            for (std::size_t j = Ny - 2; j < Ny - 1; j++) {
                for (std::size_t k = 0; k < Nz; k++) {
                    size_t pos = (i * Ny * Nz + j * Nz + k) * 3 * type_size + global_pos;
                    //parse y component
                    if (type_size == 4) {
                        uint32_t y = *(uint32_t *) (data + pos + 4);
                        y = bswap_32(y);
                        uint32_t yplusone = *(uint32_t *) (data + pos + 4 * 3 + 4);
                        yplusone = bswap_32(yplusone);
                        Real y_real = static_cast<Real>(std::bit_cast<float>(y));
                        Real yplusone_real = static_cast<Real>(std::bit_cast<float>(yplusone));
                        Real avg = (y_real + yplusone_real) * 0.5;
                        velocity_v(i, j, k) = avg;
                    } else {
                        uint64_t y = *(uint64_t *) (data + pos + 8);
                        y = bswap_64(y);
                        uint64_t yplusone = *(uint64_t *) (data + pos + 8 * 3 + 8);
                        yplusone = bswap_64(yplusone);
                        Real y_real = static_cast<Real>(std::bit_cast<double>(y));
                        Real yplusone_real = static_cast<Real>(*reinterpret_cast<double *>(&yplusone));
                        Real avg = (y_real + yplusone_real) * 0.5;
                        velocity_v(i, j, k) = avg;

                    }
                }
            }
        }

        //last pass for w component
        for (std::size_t i = 0; i < Nx; ++i) {
            for (std::size_t j = 0; j < Ny; j++) {
                for (std::size_t k = Nz - 2; k < Nz - 1; k++) {
                    size_t pos = (i * Ny * Nz + j * Nz + k) * 3 * type_size + global_pos;
                    //parse z component
                    if (type_size == 4) {
                        uint32_t z = *(uint32_t *) (data + pos + 8);
                        z = bswap_32(z);
                        uint32_t zplusone = *(uint32_t *) (data + pos + 4 * 3 + 8);
                        zplusone = bswap_32(zplusone);
                        Real z_real = static_cast<Real>(std::bit_cast<float>(z));
                        Real zplusone_real = static_cast<Real>(std::bit_cast<float>(zplusone));
                        Real avg = (z_real + zplusone_real) * 0.5;
                        velocity_w(i, j, k) = avg;

                    } else {
                        uint64_t z = *(uint64_t *) (data + pos + 16);
                        z = bswap_64(z);
                        uint64_t zplusone = *(uint64_t *) (data + pos + 8 * 3 + 16);
                        zplusone = bswap_64(zplusone);
                        Real z_real = static_cast<Real>(std::bit_cast<double>(z));
                        Real zplusone_real = static_cast<Real>(std::bit_cast<double>(zplusone));
                        Real avg = (z_real + zplusone_real) * 0.5;
                        velocity_w(i, j, k) = avg;

                    }
                }
            }
        }

        #else
        //TODO: implement big endian, probably not needed as the target machine will be le
        throw std::runtime_error("Big endian not implemented");
        #endif
        size_t pos = global_pos + 3 * type_size * (Nx) * (Ny) * (Nz) + 1;
        printf("Next line: %s\n", &data[pos]);
        //TODO: parse pressure, should be easier since we don't need to average the values
        //unmap file
        munmap(data, size);
    }
}

