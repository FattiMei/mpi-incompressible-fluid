//
// Created by giorgio on 10/10/2024.
//

#include "Timestep.h"
#include "MomentumEquation.h"

namespace mif {

    constexpr Real c2 = (8.0 / 15.0);
    constexpr Real a21 = (8.0 / 15.0);
    constexpr Real c3 = (2.0 / 3.0);
    constexpr Real a31 = (1.0 / 4.0);
    constexpr Real a32 = (5.0 / 12.0);
    constexpr Real b1 = (1.0 / 4.0);
    constexpr Real b3 = (3.0 / 4.0);


    void
    timestep(VelocityTensor &velocity, VelocityTensor &velocity_buffer1,
             std::vector<std::array<Real, 3>> rhs_buffer, Real t_n) {
        const Constants &constants = velocity.constants;
        const Real time_1 = t_n + c2 * constants.dt;
        const Real time_2 = t_n + c3 * constants.dt;
        const Real final_time = t_n + constants.dt;

        // Stage 1.
        // Apply Dirichlet boundary conditions.
        velocity_buffer1.apply_all_dirichlet_bc(time_1);

        // Compute the solution inside the domain.
        {
            const auto &vel = velocity;
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity_buffer1.u.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = velocity.constants.dt;
                            const Real rhs = calculate_momentum_rhs_with_forcing_u(
                                    vel, i, j, k, t_n);
                            rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][0] = rhs;
                            velocity_buffer1.u(i, j, k) = vel.u(i, j, k) + dt * a21 *
                                                                                rhs;
                        }
                    }
                }
            }
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity_buffer1.v.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = velocity.constants.dt;
                            const Real rhs = calculate_momentum_rhs_with_forcing_v(
                                    vel, i, j, k, t_n);
                            rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][1] = rhs;
                            velocity_buffer1.v(i, j, k) = vel.v(i, j, k) + dt * a21 *
                                                                                rhs;
                        }
                    }
                }
            }
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity_buffer1.w.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = vel.constants.dt;;
                            const Real rhs = calculate_momentum_rhs_with_forcing_w(
                                    vel, i, j, k, t_n);
                            rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][2] = rhs;
                            velocity_buffer1.w(i, j, k) = vel.w(i, j, k) + dt * a21 * rhs;
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


            const auto &vel = velocity_buffer1;
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity.u.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = velocity.constants.dt;
                            const Real rhs = rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][0];
                            rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][0] =
                                    velocity.u(i, j, k) + dt * (b1 *
                                                                rhs);
                            velocity.u(i, j, k) = velocity.u(i, j, k) + dt * (a31 *
                                                                              rhs +
                                                                              a32 *
                                                                              calculate_momentum_rhs_with_forcing_u(
                                                                                      vel, i, j, k,
                                                                                      time_1));
                        }
                    }
                }
            }
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity.v.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = velocity.constants.dt;
                            const Real rhs = rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][1];
                            rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][1] =
                                    velocity.v(i, j, k) + dt * (b1 *
                                                                rhs);
                            velocity.v(i, j, k) = velocity.v(i, j, k) + dt * (a31 *
                                                                              rhs +
                                                                              a32 *
                                                                              calculate_momentum_rhs_with_forcing_v(
                                                                                      vel, i, j, k,
                                                                                      time_1));

                        }
                    }
                }
            }
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity.w.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = velocity.constants.dt;
                            const Real rhs = rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][2];
                            rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][2] =
                                    velocity.w(i, j, k) + dt * (b1 *
                                                                rhs);
                            velocity.w(i, j, k) = velocity.w(i, j, k) + dt * (a31 *
                                                                              rhs +
                                                                              a32 *
                                                                              calculate_momentum_rhs_with_forcing_w(
                                                                                      vel, i, j, k,
                                                                                      time_1));

                        }
                    }
                }
            }
        }

        // Stage 3. u_n
        // Apply Dirichlet boundary conditions.
        velocity_buffer1.apply_all_dirichlet_bc(final_time);

        // Compute the solution inside the domain.
        {

            const auto &vel = velocity;
            {
                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity_buffer1.u.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = vel.constants.dt;
                            const Real rhs = rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][0];
                            velocity_buffer1.u(i, j, k) = rhs + dt * (
                                    b3 *
                                    calculate_momentum_rhs_with_forcing_u(
                                            vel, i, j, k,
                                            time_2));
                        }
                    }
                }
            }
            {

                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity_buffer1.v.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = vel.constants.dt;
                            const Real rhs = rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][1];
                            velocity_buffer1.v(i, j, k) = rhs + dt * (
                                    b3 *
                                    calculate_momentum_rhs_with_forcing_v(
                                            vel, i, j, k,
                                            time_2));
                        }
                    }
                }
            }
            {
                //in this region velocity is constant

                size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;
                const std::array<size_t, 3> &sizes = velocity_buffer1.w.sizes();
                lower_limit = 1;
                upper_limit_x = sizes[0] - 1;
                upper_limit_y = sizes[1] - 1;
                upper_limit_z = sizes[2] - 1;
                #pragma GCC ivdep
                for (size_t i = lower_limit; i < upper_limit_x; i++) {
                    #pragma GCC ivdep
                    for (size_t j = lower_limit; j < upper_limit_y; j++) {
                        #pragma GCC ivdep
                        for (size_t k = lower_limit; k < upper_limit_z; k++) {
                            const Real dt = vel.constants.dt;
                            const Real rhs = rhs_buffer[k + j * constants.Nz + i * constants.Nz * constants.Ny][2];
                            velocity_buffer1.w(i, j, k) = rhs + dt * (
                                    b3 *
                                    calculate_momentum_rhs_with_forcing_w(
                                            vel, i, j, k,
                                            time_2));
                        }
                    }
                }
            }
        }

        // Put the solution in the original tensors.
        velocity.swap_data(velocity_buffer1);
    }

}
