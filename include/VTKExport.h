#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

#include <string>
#include "VelocityTensor.h"

namespace mif {

void writeVTK(const std::string&     filename,
			  const VelocityTensor&  velocity,
			  const StaggeredTensor& pressure);

// only works for runs with only one processor, used for validating the line exporting
void writeVTKFullMesh(const std::string&     filename,
			  const VelocityTensor&  velocity,
			  const StaggeredTensor& pressure);

} // mif

#endif // VTK_EXPORT_H
