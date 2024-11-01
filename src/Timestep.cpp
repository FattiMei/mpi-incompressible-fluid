//
// Created by giorgio on 10/10/2024.
//

#include "Timestep.h"
#include "MomentumEquation.h"
#include "VelocityTensorMacros.h"
#include <cmath>
#include "immintrin.h"

namespace mif {
    constexpr Real c2 = (8.0 / 15.0);
    constexpr Real a21 = (8.0 / 15.0);
    constexpr Real c3 = (2.0 / 3.0);
    constexpr Real a31 = (1.0 / 4.0);
    constexpr Real a32 = (5.0 / 12.0);
    constexpr Real b1 = (1.0 / 4.0);
    constexpr Real b3 = (3.0 / 4.0);


    Real timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer,
                  VelocityTensor &rhs_buffer, Real t_n,Real target_cfl,Real last_dt) {
        const Constants &constants = velocity.constants;

        float max_velocity = -1.0;
        for (size_t i = 0; i < constants.Nx - 2; i++) {
            for (size_t j = 0; j < constants.Ny - 2; j++) {
#pragma GCC ivdep
                for (size_t k = 0; k < constants.Nz - 2; k++) {
                    max_velocity = std::max(max_velocity, std::abs(velocity.u(i, j, k)));
                    max_velocity = std::max(max_velocity, std::abs(velocity.v(i, j, k)));
                    max_velocity = std::max(max_velocity, std::abs(velocity.w(i, j, k)));
                }
            }
        }

        Real target_dt = constants.dx * target_cfl / max_velocity;
        Real dt = std::clamp(target_dt, 0.5f * last_dt, 2.0f * last_dt);

        const Real time_1 = t_n + c2 * dt;
        const Real time_2 = t_n + c3 * dt;
        const Real final_time = t_n + dt;


        // Stage 1.
        // Apply Dirichlet boundary conditions.
        velocity_buffer.apply_all_dirichlet_bc(time_1);

        // Compute the solution inside the domain.
        {
            {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.u.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = calculate_momentum_rhs_with_forcing_u(velocity, i, j, k, t_n);
                                rhs_buffer.u(i, j, k) = rhs;
                                velocity_buffer.u(i, j, k) = velocity.u(i, j, k) + dt * a21 * rhs;
                            }
                        }
                    }
                }
            } {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.v.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = calculate_momentum_rhs_with_forcing_v(velocity, i, j, k, t_n);
                                rhs_buffer.v(i, j, k) = rhs;
                                velocity_buffer.v(i, j, k) = velocity.v(i, j, k) + dt * a21 * rhs;
                            }
                        }
                    }
                }
            } {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.w.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = calculate_momentum_rhs_with_forcing_w(velocity, i, j, k, t_n);
                                rhs_buffer.w(i, j, k) = rhs;
                                velocity_buffer.w(i, j, k) = velocity.w(i, j, k) + dt * a21 * rhs;
                            }
                        }
                    }
                }
            }
        }

        // Stage 2.
        // Apply Dirichlet boundary conditions.
        velocity.apply_all_dirichlet_bc(time_2);

        // Compute the solution inside the domain.
        {
            {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.u.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = rhs_buffer.u(i, j, k);
                                rhs_buffer.u(i, j, k) = velocity.u(i, j, k) + dt * (b1 * rhs);
                                velocity.u(i, j, k) =
                                        velocity.u(i, j, k) + dt * (
                                            a31 * rhs + a32 * calculate_momentum_rhs_with_forcing_u(
                                                velocity_buffer, i, j, k, time_1));
                            }
                        }
                    }
                }
            } {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.v.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = rhs_buffer.v(i, j, k);
                                rhs_buffer.v(i, j, k) = velocity.v(i, j, k) + dt * (b1 * rhs);
                                velocity.v(i, j, k) =
                                        velocity.v(i, j, k) + dt * (
                                            a31 * rhs + a32 * calculate_momentum_rhs_with_forcing_v(
                                                velocity_buffer, i, j, k, time_1));
                            }
                        }
                    }
                }
            } {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.w.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = rhs_buffer.w(i, j, k);
                                rhs_buffer.w(i, j, k) = velocity.w(i, j, k) + dt * (b1 * rhs);
                                velocity.w(i, j, k) =
                                        velocity.w(i, j, k) + dt * (
                                            a31 * rhs + a32 * calculate_momentum_rhs_with_forcing_w(
                                                velocity_buffer, i, j, k, time_1));
                            }
                        }
                    }
                }
            }
        }

        // Stage 3. u_n
        // Apply Dirichlet boundary conmy rk do not converge n time, t dfiverge the more timestep putditions.
        velocity_buffer.apply_all_dirichlet_bc(final_time);

        // Compute the solution inside the domain.
        {
            {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.u.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = rhs_buffer.u(i, j, k);
                                velocity_buffer.u(i, j, k) = rhs + dt * (b3 * calculate_momentum_rhs_with_forcing_u(
                                                                             velocity, i, j, k, time_2));
                            }
                        }
                    }
                }
            } {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.v.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = rhs_buffer.v(i, j, k);
                                velocity_buffer.v(i, j, k) = rhs + dt * (b3 * calculate_momentum_rhs_with_forcing_v(
                                                                             velocity, i, j, k, time_2));
                            }
                        }
                    }
                }
            } {
                {
                    size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                    const std::array<size_t, 3> &sizes = velocity.w.sizes();
                    lower_limit = 1;
                    upper_limit_x = sizes[0] - 1;
                    upper_limit_y = sizes[1] - 1;
                    upper_limit_z = sizes[2] - 1;
                    for (size_t i = lower_limit; i < upper_limit_x; i++) {
                        for (size_t j = lower_limit; j < upper_limit_y; j++) {
#pragma GCC ivdep
                            for (size_t k = lower_limit; k < upper_limit_z; k++) {
                                const Real rhs = rhs_buffer.w(i, j, k);
                                velocity_buffer.w(i, j, k) = rhs + dt * (b3 * calculate_momentum_rhs_with_forcing_w(
                                                                             velocity, i, j, k, time_2));
                            }
                        }
                    }
                }
            }
        }

        // Put the solution in the original tensors.
        velocity.swap_data(velocity_buffer);
        return dt;
    }
} // namespace mif
