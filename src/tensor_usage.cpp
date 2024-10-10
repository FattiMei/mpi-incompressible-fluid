#include <Tensor.h>

#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char *argv[]) {
  const std::size_t depth = 2;
  const std::size_t rows = 2;
  const std::size_t cols = 2;
  // Create a tensor
  mif::Tensor<double, 3> t({rows, cols, depth});
  // Load data
  if (!t.load("in.txt")) {
    std::cerr << "Failed to load tensor from file." << std::endl;
    return 1;
  }
  // Display data
  std::cout << t << std::endl;
  // Write data
  if (!t.dump("out.txt")) {
    std::cerr << "Failed to dump tensor to file." << std::endl;
    return 1;
  }

  // Access to an element in the space
  std::cout << "value at {0,0} of depth 0: " << T_VALUE_3D(t, 0, 0, 0)
            << std::endl;
  std::cout << "value at {1,1} of depth 1: " << T_VALUE_3D(t, 0, 1, 1)
            << std::endl;

  return 0;
}
