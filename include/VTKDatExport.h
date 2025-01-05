#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

#include <string>
#include "VelocityTensor.h"

namespace mif {

// Export the requested planes of the tensors in VTK.
void writeVTK(const std::string&     filename,
			  const VelocityTensor&  velocity,
			  const StaggeredTensor& pressure);

// Write a profile in dat format.
// Direction is 0 for x, 1 for y, 2 for z. This is the axis which the line is parallel to.
// x,y,z are the coordinates of a point contained in the line.
void writeDat(const std::string& filename,
              const VelocityTensor& velocity,
              const StaggeredTensor& pressure,
              const int direction,
              const Real x, const Real y, const Real z);

// Export the entire mesh in VTK.
// Only works for runs serially, used for validating the first function.
void writeVTKFullMesh(const std::string&     filename,
                      const mif::VelocityTensor& velocity,
                      const StaggeredTensor& pressure);

} // mif

#endif // VTK_EXPORT_H
