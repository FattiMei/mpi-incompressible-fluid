#include <Tensor.h>

#include <iostream>

int main(int argc, char *argv[]) {
  const std::size_t depth = 2;
  const std::size_t rows = 2;
  const std::size_t cols = 2;
  std::cout << std::endl
            << "Testing tensor load, dump and indexing" << std::endl;
  // Create a tensor
  mif::Tensor<double, 3> t({depth, rows, cols});
  // Load data
  if (!t.load("in.txt")) {
    std::cerr << "Failed to load tensor from file." << std::endl;
    return 1;
  }
  std::cout << "Tensor loaded from file" << std::endl;
  // Display data
  std::cout << t << std::endl;
  // Write data
  if (!t.dump("out.txt")) {
    std::cerr << "Failed to dump tensor to file." << std::endl;
    return 1;
  }
  std::cout << "Tensor dumped to file" << std::endl << std::endl;

  std::cout << "Reading all tensor elements:" << std::endl;
  // Access to an element in the space
  for (std::size_t i = 0; i < depth; i++) {
    for (std::size_t j = 0; j < rows; j++) {
      for (std::size_t k = 0; k < cols; k++) {
        std::cout << "value at {" << i << "," << j << "," << k << "}"
                  << t(i, j, k) << std::endl;
      }
    }
  }

  // Modify element
  std::cout << std::endl
            << "Modifying vlaue at {0, 0, 0} with constexpr indexing"
            << std::endl;
  // This is a compile time dispatched access
  t(std::integer_sequence<unsigned, 0, 0, 0>{}) = 3.14;
  std::cout << "value at {0,0,0}: "
            << t(std::integer_sequence<unsigned, 0, 0, 0>{}) << std::endl;

  // The one before was constexpr indexes, a wonderful word...
  // We need to manage indexing with runtime values also
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t k = 0;
  // Modify element
  t(i, j, k) = 9.8;
  std::cout << std::endl << "value at {0,0,0}: " << t(i, j, k) << std::endl;

  // Boundary conditions
  std::cout << std::endl << "Testing boundary conditions" << std::endl;
  // Create a tensor
  mif::Tensor<double, 3> tt({depth, rows, cols});
  // Load data
  if (!tt.load("zero.txt")) {
    std::cerr << "Failed to load tensor from file." << std::endl;
    return 1;
  }
  auto fn = [](const unsigned i) -> double {
    if (i < 2) {
      return 2.0;
    }
    return 3.0;
  };
  for (unsigned i = 0; i < 6; ++i) {
    if (i > 0) {
      tt.apply_dirichlet_boundary_face(i - 1, 0.0);
    }
    tt.apply_dirichlet_boundary_face(i, fn, i * 0.5);
    std::cout << "After BC on face " << i << ":" << std::endl
              << tt << std::endl;
  }
  // We can also assign a specific point to a boudnary value.
  for (unsigned i = 0; i < 6; ++i) {
    tt.apply_dirichlet_boundary_face(i, 0.0);
  }
  tt.apply_dirichlet_boundary_point<int, int, int>(0, 0, 0, fn, 0);
  std::cout << "After BC on point {0,0,0}: " << std::endl << tt << std::endl;
  tt.apply_dirichlet_boundary_point<int, int, int>(1, 1, 1, fn, 10);
  std::cout << "After BC on point {1,1,1}: " << std::endl << tt << std::endl;

  return 0;
}
