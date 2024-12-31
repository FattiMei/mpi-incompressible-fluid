#include "inputParser.h"

void parse_input_file(const std::string &filename, 
                      size_t &Nx_global, size_t &Ny_global, size_t &Nz_global,
                      double &dt, unsigned int &num_time_steps, 
                      int &Pz, int &Py, bool &test_case_2) {
                        
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening input file: " + filename);
    }

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

        if (key == "Nx") {
            Nx_global = std::stoul(value);
        } else if (key == "Ny") {
            Ny_global = std::stoul(value);
        } else if (key == "Nz") {
            Nz_global = std::stoul(value);
        } else if (key == "dt") {
            dt = std::stod(value);
        } else if (key == "Nt") {
            num_time_steps = std::stoul(value);
        } else if (key == "Px") {
            Pz = std::stoi(value);
        } else if (key == "Py") {
            Py = std::stoi(value);
        } else if (key == "test_case_2") {
            test_case_2 = (value == "true");
        } else {
            std::cerr << "Unknown key: " << key << std::endl;
        }
    }

    infile.close();
}