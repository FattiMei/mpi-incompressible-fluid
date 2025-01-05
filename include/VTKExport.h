#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

#include <string>
#include "VelocityTensor.h"

namespace mif {

// Export the requested planes of the tensors in VTK.
void writeVTK(const std::string&     filename,
			  const VelocityTensor&  velocity,
			  const StaggeredTensor& pressure);

// Export the entire mesh in VTK.
// Only works for runs serially, used for validating the other function.
void writeVTKFullMesh(const std::string&     filename,
					  const VelocityTensor&  velocity,
					  const StaggeredTensor& pressure);

} // mif

#endif // VTK_EXPORT_H
