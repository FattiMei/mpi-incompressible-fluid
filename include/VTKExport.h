#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

#include <string>
#include "VelocityTensor.h"

namespace mif {

// Export the requested planes of the tensors in VTK.
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

    // Export the entire mesh in VTK.
    // Only works for runs serially, used for validating the first function.
    void writeVTKFullMesh(const std::string&     filename,
                          const mif::VelocityTensor& velocity,
                          const StaggeredTensor& pressure);

} // mif

#endif // VTK_EXPORT_H
