//
// Created by giorgio on 25/10/2024.
//

#ifndef MPI_INCOMPRESSIBLE_FLUID_VTK_PARSER_H
#define MPI_INCOMPRESSIBLE_FLUID_VTK_PARSER_H

#include "vtk_parser.h"
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <Tensor.h>
#include <cstring>
#include <sys/mman.h>

namespace vtk {
    using namespace mif;

    void parse(const std::string &filename, Tensor<> &velocity_u, Tensor<> &velocity_v, Tensor<> &velocity_w,
               Tensor<> &pressure);
}
#endif //MPI_INCOMPRESSIBLE_FLUID_VTK_PARSER_H
