#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <array>
#include "Real.h"
void parse_input_file(const std::string &filename, 
                       size_t &Nx_global,  size_t &Ny_global,  size_t &Nz_global,
                       Real &dt,  unsigned int &num_time_steps, 
                       int &Pz,  int &Py,  bool &test_case_2);