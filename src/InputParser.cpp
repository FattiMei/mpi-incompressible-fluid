#include "InputParser.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mif {

void parse_input_file(const std::string &filename, 
                      size_t &Nx_global, size_t &Ny_global, size_t &Nz_global,
                      Real &dt, unsigned int &num_time_steps, 
                      int &Py, int &Pz, bool &test_case_2) {
                        
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening input file: " + filename);
    }

    bool found_Nx = false;
    bool found_Ny = false;
    bool found_Nz = false;
    bool found_dt = false;
    bool found_Nt = false;
    bool found_Py = false;
    bool found_Pz = false;
    bool found_test_case = false;

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string key, value;

        if (line.find(':') != std::string::npos) {
            size_t pos = line.find(':');
            key = line.substr(0, pos);
            value = line.substr(pos + 1);
            // Remove whitespace
            key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);
            value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
        } else {
            continue;
        }

        if (key == "Nx" && !found_Nx) {
            Nx_global = std::stoul(value);
            found_Nx = true;
        } else if (key == "Ny" && !found_Ny) {
            Ny_global = std::stoul(value);
            found_Ny = true;
        } else if (key == "Nz" && !found_Nz) {
            Nz_global = std::stoul(value);
            found_Nz = true;
        } else if (key == "dt" && !found_dt) {
            dt = std::stod(value);
            found_dt = true;
        } else if (key == "Nt" && !found_Nt) {
            num_time_steps = std::stoul(value);
            found_Nt = true;
        } else if (key == "Py" && !found_Py) {
            Py = std::stoi(value);
            found_Py = true;
        } else if (key == "Pz" && !found_Pz) {
            Pz = std::stoi(value);
            found_Pz = true;
        } else if (key == "test_case_2" && !found_test_case) {
            test_case_2 = (value == "true");
            found_test_case = true;
        } else {
            throw std::runtime_error("Unknown or duplicate key: " + key);
        }
    }

    infile.close();

    if (!found_Nx || !found_Ny || !found_Nz || !found_dt || !found_Nt || !found_Py || !found_Pz || !found_test_case) {
        throw std::runtime_error("Missing one or more inputs. Required inputs: Nx, Ny, Nz, dt, Nt, Py, Pz, test_case_2.");
    }
}

} // mif