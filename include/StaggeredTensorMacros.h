#ifndef VELOCITY_TENSOR_MACROS_H
#define VELOCITY_TENSOR_MACROS_H

// Execute "CODE" over all points for a staggered tensor, or over all internal 
// points if include_border is false.
#define STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(tensor, include_border, CODE) \
  {                                                                            \
    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;           \
    const std::array<size_t, 3> &sizes = tensor.sizes();                       \
    if constexpr (include_border) {                                            \
      lower_limit = 0;                                                         \
    } else {                                                                   \
      lower_limit = 1;                                                         \
    };                                                                         \
    if constexpr (include_border) {                                            \
      upper_limit_x = sizes[0];                                                \
    } else {                                                                   \
      upper_limit_x = sizes[0] - 1;                                            \
    };                                                                         \
    if constexpr (include_border) {                                            \
      upper_limit_y = sizes[1];                                                \
    } else {                                                                   \
      upper_limit_y = sizes[1] - 1;                                            \
    };                                                                         \
    if constexpr (include_border) {                                            \
      upper_limit_z = sizes[2];                                                \
    } else {                                                                   \
      upper_limit_z = sizes[2] - 1;                                            \
    };                                                                         \
    for (size_t k = lower_limit; k < upper_limit_z; k++) {                     \
      for (size_t j = lower_limit; j < upper_limit_y; j++) {                   \
        for (size_t i = lower_limit; i < upper_limit_x; i++) {                 \
          CODE                                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

// Execute "CODE" over all points for a staggered tensor, excluding points 
// that belong to another processor, but including true borders.
#define STAGGERED_TENSOR_ITERATE_OVER_ALL_OWNER_POINTS(tensor, CODE)                 \
  {                                                                                  \
    size_t lower_limit_y, upper_limit_y, lower_limit_z, upper_limit_z;               \
    const std::array<size_t, 3> &sizes = tensor.sizes();                             \
    if (tensor.constants.prev_proc_y != -1) {                                        \
      lower_limit_y = 1;                                                             \
    } else {                                                                         \
      lower_limit_y = 0;                                                             \
    };                                                                               \
    if (tensor.constants.next_proc_y != -1) {                                        \
      upper_limit_y = sizes[1] - 1;                                                  \
    } else {                                                                         \
      upper_limit_y = sizes[1];                                                      \
    };                                                                               \
    if (tensor.constants.prev_proc_z != -1) {                                        \
      lower_limit_z = 1;                                                             \
    } else {                                                                         \
      lower_limit_z = 0;                                                             \
    };                                                                               \
    if (tensor.constants.next_proc_z != -1) {                                        \
      upper_limit_z = sizes[2] - 1;                                                  \
    } else {                                                                         \
      upper_limit_z = sizes[2];                                                      \
    };                                                                               \
    for (size_t k = lower_limit_z; k < upper_limit_z; k++) {                         \
      for (size_t j = lower_limit_y; j < upper_limit_y; j++) {                       \
        for (size_t i = 0; i < sizes[0]; i++) {                                      \
          CODE                                                                       \
        }                                                                            \
      }                                                                              \
    }                                                                                \
  }

// Set the values of a staggered tensor calculating a function over
// all of its points, or all internal points if include_border is false.
#define STAGGERED_TENSOR_FUNCTION_ON_ALL_POINTS(tensor, function,              \
                                               include_border, args...)        \
  STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(tensor, include_border,             \
  tensor(i, j, k) = function(args);)

// Set all components of the tensor in all points using the respective
// components of the function.
#define VELOCITY_TENSOR_SET_FOR_ALL_POINTS(velocity, f_u, f_v, f_w,            \
                                           include_border, args...)            \
  {                                                                            \
    STAGGERED_TENSOR_FUNCTION_ON_ALL_POINTS(velocity.u, f_u, include_border,   \
                                            args)                              \
    STAGGERED_TENSOR_FUNCTION_ON_ALL_POINTS(velocity.v, f_v, include_border,   \
                                            args)                              \
    STAGGERED_TENSOR_FUNCTION_ON_ALL_POINTS(velocity.w, f_w, include_border,   \
                                            args)                              \
  }

#endif // VELOCITY_TENSOR_MACROS_H