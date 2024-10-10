/**
 * @file Tensor.h
 * @brief Header file containing the Tensor class.
 * @author Giorgio
 * @author Frizzy
 * @author Kaixi
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

#ifndef MPI_INCOMPRESSIBLE_FLUID_TENSOR_H
#define MPI_INCOMPRESSIBLE_FLUID_TENSOR_H

#define T_VALUE_1D(t, i) (t[i])
#define T_VALUE_2D(t, i, j) (t[i * t.strides[0] + j])
#define T_VALUE_3D(t, i, j, k) (t[i * t.strides[0] + j * t.strides[1] + k])

namespace mif {

/*!
 * @class Tensor defined by row major
 * @brief A flatten multidimensional tensor
 * @tparam Type The type of the element
 * @tparam SpaceDim The physical space dimension
 * @tparam dimensionsize The tensor dimensions data type
 */
template <typename Type = double, uint8_t SpaceDim = 3,
          typename dimensionsType = std::size_t>
class Tensor {
  static_assert(SpaceDim <= 3,
                "Space dimension bigger than 3 is not supported");
  static_assert(SpaceDim >= 1,
                "Space dimension smaller than 1 is not supported");

private:
  /*!
   * The data buffer
   */
  std::array<dimensionsType, SpaceDim> dimensions;
  /*!
   * The tensor dimensions
   */
  std::vector<Type> data;

public:
  /*!
   * Cache the strindes for indexing
   */
  std::array<dimensionsType, SpaceDim - 1> strides;
  /*!
   * A flag to retrieve the total data count contained in the underlying buffer
   */
  static constexpr unsigned TOTAL_DATA_COUNT = UINT32_MAX;
  /*!
   * Constructor
   * @param _dimensions The tensor dimensions defining its dimension
   */
  Tensor(const std::array<dimensionsType, SpaceDim> &_dimensions)
      : dimensions(_dimensions),
        data(std::accumulate(_dimensions.begin(), _dimensions.end(),
                             static_cast<dimensionsType>(1),
                             std::multiplies<dimensionsType>()),
             static_cast<Type>(0)) {
    if constexpr (SpaceDim == 2) {
      strides[0] = _dimensions[1];
    }
    if constexpr (SpaceDim == 3) {
      strides[0] = _dimensions[1] * _dimensions[2];
      strides[1] = _dimensions[1];
    }
  }

  template <typename F> void set(F lambda);
  /*!
   * Resize the tensor dimension
   * Currently this has to be used with caution as data will be messed up if we
   * enlarge or shrink a initialized tensor
   * @param _dimensions The new tensor dimensions
   */
  void resize(const std::array<dimensionsType, SpaceDim> &_dimensions) {
    data.resize(std::accumulate(_dimensions.begin(), _dimensions.end(),
                                static_cast<dimensionsType>(1),
                                std::multiplies<dimensionsType>()));
  }
  /*!
   * Retrieve the dimensions dimensions
   * @return The stride values
   */
  std::array<dimensionsType, SpaceDim> const &size() const {
    return dimensions;
  }
  /*!
   * Retrieve a stride dimensions
   * @param dim The desired dimension
   * @return The stride values
   */
  dimensionsType size(const unsigned dim) const {
    if (dim == TOTAL_DATA_COUNT) {
      return std::accumulate(dimensions.begin(), dimensions.end(),
                             static_cast<dimensionsType>(1),
                             std::multiplies<dimensionsType>());
    }
    return dimensions[dim];
  }

  /*!
   * Get a copy of a data
   * @param i The data index
   * @return The requested data
   */
  inline Type operator[](const size_t i) const { return data[i]; }
  /*!
   * Get a reference of a data
   * @param i The data index
   * @return A reference to the requested data
   */
  inline Type &operator[](const size_t i) { return data[i]; }
  /*!
   * Load tensor content from file
   * @param file_name The input file name
   */
  uint8_t load(std::string const &file_name) {
    auto trim = [](std::string &str) {
      str.erase(std::remove(str.begin(), str.end(), '\n'), str.cend());
    };

    std::ifstream input_f(file_name);
    if (!input_f) {
      std::cerr << "Error opening " << file_name << " file." << std::endl;
      return 0;
    }

    std::string line;
    std::string input;
    while (std::getline(input_f, line)) {
      trim(line);
      input += (line + " ");
    }

    std::istringstream input_stream(input);
    input_stream >> *this;

    input_f.close();
    return 1;
  }
  /*!
   * Dump tensor content to file
   * @param file_name The output file name
   */
  uint8_t dump(std::string const &file_name) const {
    std::ofstream out_f(file_name);
    if (!out_f) {
      std::cerr << "Error opening " << file_name << " file." << std::endl;
      return 0;
    }

    out_f << *this;
    out_f.close();

    return 1;
  }
};

template <typename Type, uint8_t SpaceDim>
std::ostream &operator<<(std::ostream &out,
                         Tensor<Type, SpaceDim> const &tensor);

template <typename Type, uint8_t SpaceDim>
std::istream &operator<<(std::istream &in,
                         Tensor<Type, SpaceDim> const &tensor);

template <typename Type, uint8_t SpaceDim>
std::ostream &operator<<(std::ostream &out,
                         Tensor<Type, SpaceDim> const &tensor) {
  // Recover the runtime stride type
  using Stride = typename std::decay<decltype(tensor.size())>::type::value_type;
  out << "Tensor (row major). Dimensions: [ ";
  for (auto &v : tensor.size()) {
    out << v << " ";
  }
  out << "]";
  out << std::endl << "[ ";
  for (Stride i = 0; i < tensor.size(Tensor<Type, SpaceDim>::TOTAL_DATA_COUNT);
       ++i) {
    out << tensor[i] << " ";
  }
  out << "]" << std::endl;

  return out;
}
template <typename Type, uint8_t SpaceDim>
std::istream &operator>>(std::istream &in, Tensor<Type, SpaceDim> &tensor) {
  // Recover the runtime stride type
  using Stride = typename std::decay<decltype(tensor.size())>::type::value_type;
  for (Stride i = 0; i < tensor.size(Tensor<Type, SpaceDim>::TOTAL_DATA_COUNT);
       ++i) {
    in >> tensor[i];
    // if the istream has not enough data we will leave the other data as they
    // are in memory (hopefully zero by tensor construction)
    if (!in) {
      break;
    }
  }
  return in;
}

} // namespace mif

#endif // MPI_INCOMPRESSIBLE_FLUID_TENSOR_H
