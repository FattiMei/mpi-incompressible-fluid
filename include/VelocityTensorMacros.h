#ifndef VELOCITY_TENSOR_MACROS_H
#define VELOCITY_TENSOR_MACROS_H

// Execute "CODE" over all points for a velocity component, or over all internal 
// points if include_border is false.
#define VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(tensor, include_border, CODE)  \
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
    };                                                                                                 \
    for (size_t i = lower_limit; i < upper_limit_x; i++) {                     \
      for (size_t j = lower_limit; j < upper_limit_y; j++) {                   \
        for (size_t k = lower_limit; k < upper_limit_z; k++) {                 \
          CODE                                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

// Set the values of a component of velocity calculating a function over
// all of its points, or all internal points if include_border is false.
#define VELOCITY_TENSOR_FUNCTION_ON_ALL_POINTS(tensor, function,               \
                                               include_border, args...)        \
  VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(tensor, include_border,              \
  tensor(i, j, k) = function(args);)

// Set all components of the tensor in all points using the respective
// components of the function.
#define VELOCITY_TENSOR_SET_FOR_ALL_POINTS(velocity, f_u, f_v, f_w,            \
                                           include_border, args...)            \
  {                                                                            \
    VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.u, f_u, include_border,   \
                                            args)                              \
    VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.v, f_v, include_border,   \
                                            args)                              \
    VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.w, f_w, include_border,   \
                                            args)                              \
  }

#endif // VELOCITY_TENSOR_MACROS_H