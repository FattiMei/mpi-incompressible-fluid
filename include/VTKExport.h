#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H


#include <string>
#include "Tensor.h"
#include "VelocityTensor.h"


namespace mif{
    void writeVTK(
        const std::string& filename,
        const VelocityTensor& velocity,
        const Constants& constants,
        const StaggeredTensor& pressure,
        const int rank,
        const int size

    );

    void writeDat(
        const std::string& filename,
        const VelocityTensor& velocity,
        const Constants& constants,
        const StaggeredTensor& pressure,
        const int rank,
        const int mpisize,
        const int direction,
        const Real x, const Real y, const Real z
    );
};


#endif
