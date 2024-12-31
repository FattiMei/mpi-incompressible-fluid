#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <array>

void parse_input_file(const std::string &filename, 
                      size_t &Nx_global, size_t &Ny_global, size_t &Nz_global,
                      double &dt, unsigned int &num_time_steps, 
                      int &Pz, int &Py, bool &test_case_2);