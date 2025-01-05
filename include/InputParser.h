#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H


#include <string>
#include "Real.h"

namespace mif {

void parse_input_file(const std::string &filename, 
                        size_t &Nx_global, size_t &Ny_global, size_t &Nz_global,
                        Real &dt, unsigned int &num_time_steps, 
                        int &Py, int &Pz, bool &test_case_2);

} // mif

#endif // INPUT_PARSER_H
