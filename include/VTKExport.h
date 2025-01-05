#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

#include <string>
#include "VelocityTensor.h"

namespace mif {

void writeVTK(const std::string&     filename,
			  const VelocityTensor&  velocity,
			  const StaggeredTensor& pressure);

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

    // only works for runs with only one processor, used for validating the line exporting
    void writeVTKFullMesh(const std::string&     filename,
                          const mif::VelocityTensor& velocity,
                          const StaggeredTensor& pressure);

} // mif

#endif // VTK_EXPORT_H
