/**
 * @file Tensor.h
 * @brief Header file containing the Tensor class.
 * @author Giorgio
 * @author Frizzy
 * @author Kaixi
 */
#include <MetaHelpers.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
#ifdef DEBUG_MODE
#include <cassert>
#endif

#ifndef MPI_INCOMPRESSIBLE_FLUID_TENSOR_H
#define MPI_INCOMPRESSIBLE_FLUID_TENSOR_H

namespace mif {

/*!
 * @class Tensor defined by row major
 * @brief A flatten multidimensional tensor
 * @tparam Type The type of the element
 * @tparam SpaceDim The physical space dimension
 * @tparam DimensionsType The tensor dimensions data type. By default
 * std::size_t as tensors shapes may be very large
 */
template <typename Type = double, uint8_t SpaceDim = 3,
          typename DimensionsType = std::size_t>
class Tensor {
  static_assert(SpaceDim <= 3,
                "Space dimension bigger than 3 is not supported");
  static_assert(SpaceDim >= 1,
                "Space dimension smaller than 1 is not supported");

#define DISPATCH(size, index_storage)                                  \
  do {                                                                 \
    if constexpr (size == 1) {                                         \
      return _data[index_storage[0]];                                  \
    } else if constexpr (size == 2) {                                  \
      return _data[index_storage[0] * _strides[0] + index_storage[1]]; \
    } else if constexpr (size == 3) {                                  \
      return _data[index_storage[0] * _strides[0] +                    \
                   index_storage[1] * _strides[1] + index_storage[2]]; \
    }                                                                  \
  } while (0)

 private:
  /*!
   * The data buffer
   */
  std::array<DimensionsType, SpaceDim> _dimensions;
  /*!
   * The tensor dimensions
   */
  std::vector<Type> _data;
  /*!
   * Cache the strindes for indexing
   * Not saving the 1D stride as it can be retrieved with the dimensions
   */
  std::array<DimensionsType, SpaceDim - 1> _strides;
  /*!
   * Initialization status
   */
  bool initialized = false;

 public:
  /*!
   * A flag to retrieve the total data count contained in the underlying buffer
   */
  static constexpr unsigned TOTAL_DATA_COUNT = UINT32_MAX;
  /*!
   * Constructor
   * @param in_dimensions The tensor dimensions defining its dimension
   */
  Tensor(const std::array<DimensionsType, SpaceDim> &in_dimensions)
      : _dimensions(in_dimensions),
        _data(std::accumulate(in_dimensions.begin(), in_dimensions.end(),
                              static_cast<DimensionsType>(1),
                              std::multiplies<DimensionsType>()),
              static_cast<Type>(0)) {
    if constexpr (SpaceDim == 2) {
      _strides[0] = in_dimensions[1];
    }
    if constexpr (SpaceDim == 3) {
      _strides[0] = in_dimensions[1] * in_dimensions[2];
      _strides[1] = in_dimensions[2];
    }
  }
  /*!
   * Resize the tensor dimension
   * Currently this has to be used with caution as data will be messed up if we
   * enlarge or shrink an initialized tensor
   * @tparam WithAllocation A flag to control if buffer memory has to be changed
   * or not
   * @param in_dimensions The new tensor dimensions
   */
  void resize(const std::array<DimensionsType, SpaceDim> &in_dimensions) {
    // Backup existing data
    std::vector<Type> buffer(std::accumulate(
        in_dimensions.begin(), in_dimensions.end(),
        static_cast<DimensionsType>(1), std::multiplies<DimensionsType>()));
    std::array<DimensionsType, SpaceDim> old_dimensions = _dimensions;
    std::array<DimensionsType, SpaceDim - 1> old_strides = _strides;

    _dimensions = in_dimensions;
    if constexpr (SpaceDim == 2) {
      _strides[0] = in_dimensions[1];
      // Copy the data
      for (DimensionsType i = 0;
           i < std::min(old_dimensions[0], in_dimensions[0]); i++) {
        for (DimensionsType j = 0;
             j < std::min(old_dimensions[1], in_dimensions[1]); j++) {
          buffer[i * _strides[0] + j] = _data[i * old_strides[0] + j];
        }
      }
    }
    if constexpr (SpaceDim == 3) {
      _strides[0] = in_dimensions[1] * in_dimensions[2];
      _strides[1] = in_dimensions[2];

      // copy the data
      for (DimensionsType i = 0;
           i < std::min(old_dimensions[0], in_dimensions[0]); i++) {
        for (DimensionsType j = 0;
             j < std::min(old_dimensions[1], in_dimensions[1]); j++) {
          for (DimensionsType k = 0;
               k < std::min(old_dimensions[2], in_dimensions[2]); k++) {
            buffer[i * _strides[0] + j * _strides[1] + k] =
                _data[i * old_strides[0] + j * old_strides[1] + k];
          }
        }
      }
    }
    _data.swap(buffer);
  }
  /*!
   * Retrieve the tensor dimensions
   * @return A reference to the dimensions vector
   */
  std::array<DimensionsType, SpaceDim> const &sizes() const {
    return _dimensions;
  }
  /*!
   * Retrieve a stride dimensions
   * The overall data discretized space size can be retrieved with the flag
   * TOTAL_DATA_COUNT
   * @param dim The desired dimension
   * @return The stride values
   */
  DimensionsType size(const unsigned dim) const {
    if (dim == TOTAL_DATA_COUNT) {
      return _data.size();
    }
    return _dimensions[dim];
  }
  /*!
   * Retrieve a copy of an element stored inside the tensor
   * Indexes dispatching is performed at compile time
   */
  template <typename T, T... Values>
  constexpr Type operator()(const std::integer_sequence<T, Values...>) const {
    constexpr unsigned size = integer_sequence_size<T, Values...>::value;
    static_assert(size > 0 && size < 4,
                  "Minimum one index and maximum 3 indexes allowed");
    constexpr std::array<T, size> index_storage = {Values...};
    DISPATCH(size, index_storage);
  }
  /*!
   * Retrieve a reference of an element stored inside the tensor
   * Indexes dispatching is performed at compile time
   */
  template <typename T, T... Values>
  constexpr Type &operator()(const std::integer_sequence<T, Values...>) {
    constexpr unsigned size = integer_sequence_size<T, Values...>::value;
    static_assert(size > 0 && size < 4,
                  "Minimum one index and maximum 3 indexes allowed");
    constexpr std::array<T, size> index_storage = {Values...};
    DISPATCH(size, index_storage);
  }
  /*!
   * Retrieve a copy of an element stored inside the tensor
   * @param i first dimension index The index
   */
  constexpr Type operator()(const DimensionsType i) const { return _data[i]; }
  /*!
   * Retrieve a reference of an element stored inside the tensor
   * @param i first dimension index The index
   */
  constexpr Type &operator()(const DimensionsType i) { return _data[i]; }
  /*!
   * Retrieve a copy of an element stored inside the tensor
   * @param i first dimension index The index
   * @param j second dimension index The index
   */
  constexpr Type operator()(const DimensionsType i,
                            const DimensionsType j) const {
    return _data[i * _strides[0] + j];
  }
  /*!
   * Retrieve a reference of an element stored inside the tensor
   * @param i first dimension index The index
   * @param j second dimension index The index
   */
  constexpr Type &operator()(const DimensionsType i, const DimensionsType j) {
    return _data[i * _strides[0] + j];
  }
  /*!
   * Retrieve a copy of an element stored inside the tensor
   * @param i first dimension index The index
   * @param j second dimension index The index
   * @param k third dimension index
   */
  constexpr Type operator()(const DimensionsType i, const DimensionsType j,
                            const DimensionsType k) const {
    return _data[i * _strides[0] + j * _strides[1] + k];
  }
  /*!
   * Retrieve a reference of an element stored inside the tensor
   * @param i first dimension index The index
   * @param j second dimension index The index
   * @param k third dimension index
   */
  constexpr Type &operator()(const DimensionsType i, const DimensionsType j,
                             const DimensionsType k) {
    return _data[i * _strides[0] + j * _strides[1] + k];
  }
  /*!
   * Apply Dirichlet boundary conditions
   * This will apply a value on a given point. Checks that the point lies on a
   * face is left to the caller
   * TODO: Integrate the assertion that the point lies on a face
   * TODO: Currently when this method is called you have to specify the template
   * arguments
   * @param face The boundary face
   * @param value The boundary value
   */
  template <typename... Indexes, typename MaybeCallable,
            typename... MaybeCallableArgs>
  typename std::enable_if<((sizeof...(Indexes) == SpaceDim)), void>::type
  apply_dirichlet_boundary_point(Indexes... indexes, const MaybeCallable &input,
                                 MaybeCallableArgs... args) {
    Type value = {0};
    if constexpr (is_callable<MaybeCallable>::value) {
      // This static assertion checks that arguments provided to the caller are
      // compatible
      static_assert(
          std::is_invocable<MaybeCallable, MaybeCallableArgs...>::value,
          "Callable cannot be invoked with the given arguments.");
      // Perfect forwarding
      value = input(std::forward<MaybeCallableArgs>(args)...);
    } else {
      value = input;
    }

    // Recover the point the space and assign the requested value
    std::array<DimensionsType, sizeof...(Indexes)> point = {
        static_cast<DimensionsType>(indexes)...};
    if constexpr (SpaceDim == 1) {
      operator()(point[0]) = value;
    } else if constexpr (SpaceDim == 2) {
      operator()(point[0], point[1]) = value;
    } else if constexpr (SpaceDim == 3) {
      operator()(point[0], point[1], point[2]) = value;
    }
  }
  /*!
   * Apply Dirichlet boundary conditions
   * This will apply the same value on an entire face
   * @param face The boundary face
   * @param value The boundary value
   */
  template <typename MaybeCallable, typename... MaybeCallableArgs>
  void apply_dirichlet_boundary_face(const uint8_t face,
                                     const MaybeCallable &input,
                                     MaybeCallableArgs... args) {
    /*
        Boundary tags

        1D:
        0+-------------------+1

        2D:
                  0
        +-------------------+
        |                   |
        |                   |
       1|                   |3
        |                   |
        |                   |
        +-------------------+
                  2

        3D:
            6+-------------------+7
           /|                  /  |
          / |                 /   |
         /  |                /    |
        4+------------------+5    |
        |   |                |    |
        |   |                |    |
        |   3+-------------------+2
        |  /                 |   /
        | /                  | /
        0+-------------------+1
        0154: 0
        4576: 1
        3276: 2
        0123: 3
        1275: 4
        0364: 5
     */
#ifdef DEBUG_MODE
    if constexpr (SpaceDim == 1) {
      assert(face < 2);
    } else if constexpr (SpaceDim == 2) {
      assert(face < 4);
    } else {
      assert(face < 6);
    }
#endif
    Type value = {0};
    if constexpr (is_callable<MaybeCallable>::value) {
      // This static assertion checks that arguments provided to the caller are
      // compatible
      static_assert(
          std::is_invocable<MaybeCallable, MaybeCallableArgs...>::value,
          "Callable cannot be invoked with the given arguments.");
      // Perfect forwarding
      value = input(std::forward<MaybeCallableArgs>(args)...);
    } else {
      value = input;
    }

    if constexpr (SpaceDim == 1) {
      switch (face) {
        case 0:
          operator()(0) = value;
          break;
        case 1:
          operator()(_dimensions[0] - 1) = value;
          break;
      }
    } else if constexpr (SpaceDim == 2) {
      switch (face) {
        case 0:
          for (DimensionsType i = 0; i < _dimensions[1]; ++i) {
            operator()(0, i) = value;
          }
          break;
        case 1:
          for (DimensionsType i = 0; i < _dimensions[0]; ++i) {
            operator()(i, 0) = value;
          }
          break;
        case 2:
          for (DimensionsType i = 0; i < _dimensions[1]; ++i) {
            operator()(_dimensions[0] - 1, i) = value;
          }
          break;
        case 3:
          for (DimensionsType i = 0; i < _dimensions[0]; ++i) {
            operator()(i, _dimensions[1] - 1) = value;
          }
          break;
      }
    } else if constexpr (SpaceDim == 3) {
      switch (face) {
        case 0:
          // First depth, loop over row and col
          for (DimensionsType i = 0; i < _dimensions[1]; ++i) {
            for (DimensionsType j = 0; j < _dimensions[2]; ++j) {
              operator()(0, i, j) = value;
            }
          }
          break;
        case 1:
          // First row, loop over depth and col
          for (DimensionsType i = 0; i < _dimensions[0]; ++i) {
            for (DimensionsType j = 0; j < _dimensions[2]; ++j) {
              operator()(i, 0, j) = value;
            }
          }
          break;
        case 2:
          // Last depth, loop over row and col
          for (DimensionsType i = 0; i < _dimensions[1]; ++i) {
            for (DimensionsType j = 0; j < _dimensions[2]; ++j) {
              operator()(_dimensions[0] - 1, i, j) = value;
            }
          }
          break;
        case 3:
          // Last row, loop depth row and col
          for (DimensionsType i = 0; i < _dimensions[0]; ++i) {
            for (DimensionsType j = 0; j < _dimensions[2]; ++j) {
              operator()(i, _dimensions[1] - 1, j) = value;
            }
          }
          break;
        case 4:
          // Last col, loop over depth and row
          for (DimensionsType i = 0; i < _dimensions[0]; ++i) {
            for (DimensionsType j = 0; j < _dimensions[1]; ++j) {
              operator()(i, j, _dimensions[2] - 1) = value;
            }
          }
          break;
        case 5:
          // First col, loop over depth and row
          for (DimensionsType i = 0; i < _dimensions[0]; ++i) {
            for (DimensionsType j = 0; j < _dimensions[1]; ++j) {
              operator()(i, j, 0) = value;
            }
          }
          break;
      }
    }
  }

  /*!
   * Load tensor content from file
   * TODO: Currently this does not support a second read (we have to resize the
   * tensor)
   * @param file_name The input file name
   */
  uint8_t load(std::string const &file_name) {
    if (initialized) {
      std::cerr << "Can not load file into an initialized tensor" << std::endl;
      return 0;
    }
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
    input_stream >> (*this);

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

    out_f << (*this);
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
  using Stride =
      typename std::decay<decltype(tensor.sizes())>::type::value_type;
  out << "Tensor (row major). Dimensions: [ ";
  for (auto &v : tensor.sizes()) {
    out << v << " ";
  }
  out << "]";
  out << std::endl << "[ ";
  for (Stride i = 0; i < tensor.size(Tensor<Type, SpaceDim>::TOTAL_DATA_COUNT);
       ++i) {
    out << tensor(i) << " ";
  }
  out << "]" << std::endl;

  return out;
}
template <typename Type, uint8_t SpaceDim>
std::istream &operator>>(std::istream &in, Tensor<Type, SpaceDim> &tensor) {
  // Recover the runtime stride type
  using Stride =
      typename std::decay<decltype(tensor.sizes())>::type::value_type;
  for (Stride i = 0; i < tensor.size(Tensor<Type, SpaceDim>::TOTAL_DATA_COUNT);
       ++i) {
    in >> tensor(i);
    // if the istream has not enough data we will leave the other memory cells
    // as they are in memory (hopefully zero by tensor construction)
    if (!in) {
      break;
    }
  }
  return in;
}

}  // namespace mif

#endif  // MPI_INCOMPRESSIBLE_FLUID_TENSOR_H
