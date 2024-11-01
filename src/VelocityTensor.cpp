#include "VelocityTensor.h"
#include "Manufactured.h"
#include <iostream>

namespace mif {

void StaggeredTensor::print() const {
  const std::array<size_t, 3> &sizes = this->sizes();
  for (size_t i = 0; i < sizes[0]; i++) {
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t k = 0; k < sizes[2]; k++) {
        std::cout << this->operator()(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void StaggeredTensor::print(const std::function<bool(Real)> &filter) const {
  const std::array<size_t, 3> &sizes = this->sizes();
  for (size_t i = 0; i < sizes[0]; i++) {
    for (size_t j = 0; j < sizes[1]; j++) {
      for (size_t k = 0; k < sizes[2]; k++) {
        const Real value = this->operator()(i, j, k);
        if (filter(value)) {
          std::cout << "(" << i << "," << j << "," << k << "): " << value
                    << std::endl;
        }
      }
    }
  }
  std::cout << std::endl;
}

    VelocityTensor::VelocityTensor(const Constants &constants): 
        u(constants),
        v(constants),
        w(constants),
        components({&this->u,&this->v,&this->w}),
        constants(constants) {}

void VelocityTensor::swap_data(VelocityTensor &other) noexcept{
  for (size_t component = 0; component < 3U; component++) {
    components[component]->swap_data(*(other.components[component]));
  }
}

void VelocityTensor::set(const VectorFunction &f, bool include_border) {
  const size_t lower_limit = include_border ? 0 : 1;
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> &sizes = tensor->sizes();
    const auto *func = f.components[component];
    const size_t upper_limit_i = include_border ? sizes[0] : sizes[0] - 1;
    const size_t upper_limit_j = include_border ? sizes[1] : sizes[1] - 1;
    const size_t upper_limit_k = include_border ? sizes[2] : sizes[2] - 1;

    for (size_t i = lower_limit; i < upper_limit_i; i++) {
      for (size_t j = lower_limit; j < upper_limit_j; j++) {
        for (size_t k = lower_limit; k < upper_limit_k; k++) {
          (*tensor)(i, j, k) =
              tensor->evaluate_function_at_index(i, j, k, *func);
        }
      }
    }
  }
}

  void VelocityTensor::set(const TimeVectorFunction &f, bool include_border,Real time) {
  const size_t lower_limit = include_border ? 0 : 1;
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> &sizes = tensor->sizes();
    const auto *func = f.components[component];
    const size_t upper_limit_i = include_border ? sizes[0] : sizes[0] - 1;
    const size_t upper_limit_j = include_border ? sizes[1] : sizes[1] - 1;
    const size_t upper_limit_k = include_border ? sizes[2] : sizes[2] - 1;

    for (size_t i = lower_limit; i < upper_limit_i; i++) {
      for (size_t j = lower_limit; j < upper_limit_j; j++) {
 #pragma GCC ivdep
        for (size_t k = lower_limit; k < upper_limit_k; k++) {
          (*tensor)(i, j, k) =
              tensor->evaluate_function_at_index(time,i, j, k, *func);
        }
      }
    }
  }
}

void VelocityTensor::apply_all_dirichlet_bc(Real time) {
  for (size_t component = 0; component < 3U; component++) {
    StaggeredTensor *tensor = components[component];
    const std::array<size_t, 3> sizes = tensor->sizes();
   const Real (*func)(Real, Real, Real,Real) noexcept=
        (component == 0) ? u_exact : (component == 1 ? v_exact : w_exact);

    // Face 1: z=0
    if (component == 2) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
 #pragma GCC ivdep
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          const Real w_at_boundary =
              func(time, i * constants.dx, j * constants.dy, 0);
          const Real du_dx =
              (u.evaluate_function_at_index(time, i, j, 0, u_exact) -
               u.evaluate_function_at_index(time, i - 1, j, 0, u_exact)) *
              constants.one_over_dx;
          const Real dv_dy =
              (v.evaluate_function_at_index(time, i, j, 0, v_exact) -
               v.evaluate_function_at_index(time, i, j - 1, 0, v_exact)) *
              constants.one_over_dy;
          w(i, j, 0) = w_at_boundary - constants.dz_over_2 * (du_dx + dv_dy);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
 #pragma GCC ivdep
        for (size_t j = 0; j < sizes[1]; j++) {
          (*tensor)(i, j, 0) =
              tensor->evaluate_function_at_index(time, i, j, 0, func);
        }
      }
    }

    // Face 2: z=z_max
    if (component == 2) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
 #pragma GCC ivdep
        for (size_t j = 1; j < constants.Ny - 1; j++) {
          const Real w_at_boundary =
              func(time, i * constants.dx, j * constants.dy, constants.z_size);
          const Real du_dx = (u.evaluate_function_at_index(
                                  time, i, j, constants.Nz - 1, u_exact) -
                              u.evaluate_function_at_index(
                                  time, i - 1, j, constants.Nz - 1, u_exact)) *
                             constants.one_over_dx;
          const Real dv_dy = (v.evaluate_function_at_index(
                                  time, i, j, constants.Nz - 1, v_exact) -
                              v.evaluate_function_at_index(
                                  time, i, j - 1, constants.Nz - 1, v_exact)) *
                             constants.one_over_dy;
          w(i, j, constants.Nz - 2) =
              w_at_boundary + constants.dz_over_2 * (du_dx + dv_dy);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
 #pragma GCC ivdep
        for (size_t j = 0; j < sizes[1]; j++) {
          (*tensor)(i, j, constants.Nz - 1) =
              tensor->evaluate_function_at_index(time, i, j, constants.Nz - 1,
                                                 func);
        }
      }
    }

    // Face 3: y=0
    if (component == 1) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
 #pragma GCC ivdep
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real v_at_boundary =
              func(time, i * constants.dx, 0, k * constants.dz);
          const Real du_dx =
              (u.evaluate_function_at_index(time, i, 0, k, u_exact) -
               u.evaluate_function_at_index(time, i - 1, 0, k, u_exact)) *
              constants.one_over_dx;
          const Real dw_dz =
              (w.evaluate_function_at_index(time, i, 0, k, w_exact) -
               w.evaluate_function_at_index(time, i, 0, k - 1, w_exact)) *
              constants.one_over_dz;
          v(i, 0, k) = v_at_boundary - constants.dy_over_2 * (du_dx + dw_dz);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
 #pragma GCC ivdep
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(i, 0, k) =
              tensor->evaluate_function_at_index(time, i, 0, k, func);
        }
      }
    }

    // Face 4: y=y_max
    if (component == 1) {
      for (size_t i = 1; i < constants.Nx - 1; i++) {
 #pragma GCC ivdep
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real v_at_boundary =
              func(time, i * constants.dx, constants.y_size, k * constants.dz);
          const Real du_dx = (u.evaluate_function_at_index(
                                  time, i, constants.Ny - 1, k, u_exact) -
                              u.evaluate_function_at_index(
                                  time, i - 1, constants.Ny - 1, k, u_exact)) *
                             constants.one_over_dx;
          const Real dw_dz = (w.evaluate_function_at_index(
                                  time, i, constants.Ny - 1, k, w_exact) -
                              w.evaluate_function_at_index(
                                  time, i, constants.Ny - 1, k - 1, w_exact)) *
                             constants.one_over_dz;
          v(i, constants.Ny - 2, k) =
              v_at_boundary + constants.dy_over_2 * (du_dx + dw_dz);
        }
      }
    } else {
      for (size_t i = 0; i < sizes[0]; i++) {
 #pragma GCC ivdep
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(i, constants.Ny - 1, k) =
              tensor->evaluate_function_at_index(time, i, constants.Ny - 1, k,
                                                 func);
        }
      }
    }

    // Face 5: x=0
    if (component == 0) {
      for (size_t j = 1; j < constants.Ny - 1; j++) {
 #pragma GCC ivdep
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real u_at_boundary =
              func(time, 0, j * constants.dy, k * constants.dz);
          const Real dv_dy =
              (v.evaluate_function_at_index(time, 0, j, k, v_exact) -
               v.evaluate_function_at_index(time, 0, j - 1, k, v_exact)) *
              constants.one_over_dy;
          const Real dw_dz =
              (w.evaluate_function_at_index(time, 0, j, k, w_exact) -
               w.evaluate_function_at_index(time, 0, j, k - 1, w_exact)) *
              constants.one_over_dz;
          u(0, j, k) = u_at_boundary - constants.dx_over_2 * (dv_dy + dw_dz);
        }
      }
    } else {
      for (size_t j = 0; j < sizes[1]; j++) {
 #pragma GCC ivdep
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(0, j, k) =
              tensor->evaluate_function_at_index(time, 0, j, k, func);
        }
      }
    }

    // Face 6: x=x_max
    if (component == 0) {
      for (size_t j = 1; j < constants.Ny - 1; j++) {
 #pragma GCC ivdep
        for (size_t k = 1; k < constants.Nz - 1; k++) {
          const Real u_at_boundary =
              func(time, constants.x_size, j * constants.dy, k * constants.dz);
          const Real dv_dy = (v.evaluate_function_at_index(
                                  time, constants.Nx - 1, j, k, v_exact) -
                              v.evaluate_function_at_index(
                                  time, constants.Nx - 1, j - 1, k, v_exact)) *
                             constants.one_over_dy;
          const Real dw_dz = (w.evaluate_function_at_index(
                                  time, constants.Nx - 1, j, k, w_exact) -
                              w.evaluate_function_at_index(
                                  time, constants.Nx - 1, j, k - 1, w_exact)) *
                             constants.one_over_dz;
          u(constants.Nx - 2, j, k) =
              u_at_boundary + constants.dx_over_2 * (dv_dy + dw_dz);
        }
      }
    } else {
      for (size_t j = 0; j < sizes[1]; j++) {
 #pragma GCC ivdep
        for (size_t k = 0; k < sizes[2]; k++) {
          (*tensor)(constants.Nx - 1, j, k) =
              tensor->evaluate_function_at_index(time, constants.Nx - 1, j, k,
                                                 func);
        }
      }
    }
  }
}
} // namespace mif