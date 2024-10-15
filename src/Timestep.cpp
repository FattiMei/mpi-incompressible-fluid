#include <cassert>
#include "BoundaryConditions.h"
#include "Constants.h"
#include "MomentumEquation.h"
#include "FunctionHelpers.h"
#include "Tensor.h"
#include "Timestep.h"
#include "VelocityComponent.h"

namespace mif {
    constexpr Real c2 = (8.0 / 15.0);
    constexpr Real a21 = (8.0 / 15.0);
    constexpr Real c3 = (2.0 / 3.0);
    constexpr Real a31 = (1.0 / 4.0);
    constexpr Real a32 = (5.0 / 12.0);
    constexpr Real b1 = (1.0 / 4.0);
    constexpr Real b3 = (3.0 / 4.0);

    // Perform the first stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_Y2(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const std::function<Real(Real, Real, Real, Real)> &forcing_term,
               Real current_time, const Constants &constants, 
               size_t i, size_t j, size_t k) {
        const Real initial_term = choose_component<component>(u, v, w)(i,j,k);
        return initial_term + 
               constants.dt * a21 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, function_at_time(forcing_term, current_time), constants);
    }

    // Perform the second stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_Y3(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w, 
               const Tensor<> &Y2_u, const Tensor<> &Y2_v, const Tensor<> &Y2_w,
               const std::function<Real(Real, Real, Real, Real)> &forcing_term,
               Real current_time, const Constants &constants, 
               size_t i, size_t j, size_t k) {
        const Real time_1 = current_time + c2*constants.dt;
        const Real initial_term = choose_component<component>(u, v, w)(i,j,k);
        const Real fY2 = calculate_momentum_rhs_with_forcing<component>(Y2_u, Y2_v, Y2_w, i, j, k,
                                                                        function_at_time(forcing_term, time_1),
                                                                        constants);
        const Real fU = calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k,
                                                                       function_at_time(forcing_term, current_time),
                                                                       constants);

        return initial_term +
               constants.dt * a31 * fU +
               constants.dt * a32 * fY2;
    }

    // Perform the third stage of a single step of an explicit RK3 method for a single point of a single component.
    template <VelocityComponent component> inline Real
    compute_new_u(const Tensor<> &u, const Tensor<> &v, const Tensor<> &w,
                  const Tensor<> &Y3_u, const Tensor<> &Y3_v, const Tensor<> &Y3_w,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term,
                  Real current_time, const Constants &constants, 
                  size_t i, size_t j, size_t k) {
        const Real time_2 = current_time + c3*constants.dt;
        const Real initial_term = choose_component<component>(u, v, w)(i,j,k);
        return initial_term + 
               constants.dt * b1 * calculate_momentum_rhs_with_forcing<component>(u, v, w, i, j, k, function_at_time(forcing_term, current_time), constants) + 
               constants.dt * b3 * calculate_momentum_rhs_with_forcing<component>(Y3_u, Y3_v, Y3_w, i, j, k, function_at_time(forcing_term, time_2), constants);
    }

    void timestep(Tensor<> &u, Tensor<> &v, Tensor<> &w, 
                  Tensor<> &u_buffer1, Tensor<> &v_buffer1, Tensor<> &w_buffer1,
                  Tensor<> &u_buffer2, Tensor<> &v_buffer2, Tensor<> &w_buffer2, 
                  const std::function<Real(Real, Real, Real, Real)> &u_exact,
                  const std::function<Real(Real, Real, Real, Real)> &v_exact,
                  const std::function<Real(Real, Real, Real, Real)> &w_exact, 
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_u,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_v,
                  const std::function<Real(Real, Real, Real, Real)> &forcing_term_w,
                  Real current_time,
                  std::vector<std::array<Real, 3>> &rhs,
                  const Constants &constants) {
        // Precompute constants to avoid repetitive calculations inside the loops
        const Real time_1 = current_time + c2 * constants.dt;
        const Real time_2 = current_time + c3 * constants.dt;
        const Real final_time = current_time + constants.dt;
        const Real dt_a21 = constants.dt * a21;
        const Real dt_a31 = constants.dt * a31;
        const Real dt_a32 = constants.dt * a32;
        const Real dt_b1 = constants.dt * b1;
        const Real dt_b3 = constants.dt * b3;

        // Precompute forcing terms for current and future times to avoid recomputation
        auto forcing_term_u_at_time = function_at_time(forcing_term_u, current_time);
        auto forcing_term_v_at_time = function_at_time(forcing_term_v, current_time);
        auto forcing_term_w_at_time = function_at_time(forcing_term_w, current_time);

        auto forcing_term_u_at_time_1 = function_at_time(forcing_term_u, time_1);
        auto forcing_term_v_at_time_1 = function_at_time(forcing_term_v, time_1);
        auto forcing_term_w_at_time_1 = function_at_time(forcing_term_w, time_1);

        auto forcing_term_u_at_time_2 = function_at_time(forcing_term_u, time_2);
        auto forcing_term_v_at_time_2 = function_at_time(forcing_term_v, time_2);
        auto forcing_term_w_at_time_2 = function_at_time(forcing_term_w, time_2);



        // Apply Dirichlet boundary conditions for the first intermediate time (Stage 1)
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer1, function_at_time(u_exact, time_1), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer1, function_at_time(v_exact, time_1), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer1, function_at_time(w_exact, time_1), constants);

        // --- Stage 1: Compute Y2 for each velocity component ---
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                #pragma ivdep
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    //Compiler does not seam to like to inline the function calls here, so we will do it manually (we could also use a pragma to force inlining)
                    const Real rhs_u = calculate_momentum_rhs_with_forcing<VelocityComponent::u>(u, v, w, i, j, k,
                                                                                                 forcing_term_u_at_time,
                                                                                                 constants);
                    const Real rhs_v = calculate_momentum_rhs_with_forcing<VelocityComponent::v>(u, v, w, i, j, k,
                                                                                                 forcing_term_v_at_time,
                                                                                                 constants);
                    const Real rhs_w = calculate_momentum_rhs_with_forcing<VelocityComponent::w>(u, v, w, i, j, k,
                                                                                                 forcing_term_w_at_time,
                                                                                                 constants);
                    rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k] = {rhs_u, rhs_v, rhs_w};
                    u_buffer1(i, j, k) = u(i, j, k) + dt_a21 * rhs_u;
                    v_buffer1(i, j, k) = v(i, j, k) + dt_a21 * rhs_v;
                    w_buffer1(i, j, k) = w(i, j, k) + dt_a21 * rhs_w;
                }
            }
        }

        // Apply Dirichlet boundary conditions for the second intermediate time (Stage 2)
        apply_all_dirichlet_bc<VelocityComponent::u>(u_buffer2, function_at_time(u_exact, time_2), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v_buffer2, function_at_time(v_exact, time_2), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w_buffer2, function_at_time(w_exact, time_2), constants);


        // --- Stage 2: Compute Y3 for each velocity component ---
        for (size_t i = 1; i < constants.Nx - 1; i++) {
            for (size_t j = 1; j < constants.Ny - 1; j++) {
                #pragma ivdep
                for (size_t k = 1; k < constants.Nz - 1; k++) {
                    /*const Real rhs_u = calculate_momentum_rhs_with_forcing<VelocityComponent::u>(u, v, w, i, j, k, forcing_term_u_at_time, constants);
                    const Real rhs_v = calculate_momentum_rhs_with_forcing<VelocityComponent::v>(u, v, w, i, j, k, forcing_term_v_at_time, constants);
                    const Real rhs_w = calculate_momentum_rhs_with_forcing<VelocityComponent::w>(u, v, w, i, j, k, forcing_term_w_at_time, constants);
*/
                    const Real rhs_u = rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k][0];
                    const Real rhs_v = rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k][1];
                    const Real rhs_w = rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k][2];


                    const Real rhs_u_buffer1 = calculate_momentum_rhs_with_forcing<VelocityComponent::u>(u_buffer1,
                                                                                                         v_buffer1,
                                                                                                         w_buffer1, i,
                                                                                                         j, k,
                                                                                                         forcing_term_u_at_time_1,
                                                                                                         constants);
                    const Real rhs_v_buffer1 = calculate_momentum_rhs_with_forcing<VelocityComponent::v>(u_buffer1,
                                                                                                         v_buffer1,
                                                                                                         w_buffer1, i,
                                                                                                         j, k,
                                                                                                         forcing_term_v_at_time_1,
                                                                                                         constants);
                    const Real rhs_w_buffer1 = calculate_momentum_rhs_with_forcing<VelocityComponent::w>(u_buffer1,
                                                                                                         v_buffer1,
                                                                                                         w_buffer1, i,
                                                                                                         j, k,
                                                                                                         forcing_term_w_at_time_1,
                                                                                                         constants);

                    u_buffer2(i, j, k) = u(i, j, k) + dt_a31 * rhs_u + dt_a32 * rhs_u_buffer1;
                    v_buffer2(i, j, k) = v(i, j, k) + dt_a31 * rhs_v + dt_a32 * rhs_v_buffer1;
                    w_buffer2(i, j, k) = w(i, j, k) + dt_a31 * rhs_w + dt_a32 * rhs_w_buffer1;
                }
            }
        }

        // Apply Dirichlet boundary conditions for the final time (Stage 3)
        apply_all_dirichlet_bc<VelocityComponent::u>(u, function_at_time(u_exact, final_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::v>(v, function_at_time(v_exact, final_time), constants);
        apply_all_dirichlet_bc<VelocityComponent::w>(w, function_at_time(w_exact, final_time), constants);

        // --- Stage 3: Final computation for each velocity component ---
        for (size_t i = 1; i < constants.Nx - 1; ++i) {
            for (size_t j = 1; j < constants.Ny - 1; ++j) {
                #pragma ivdep
                for (size_t k = 1; k < constants.Nz - 1; ++k) {
                    /*const Real rhs_u = calculate_momentum_rhs_with_forcing<VelocityComponent::u>(u, v, w, i, j, k, forcing_term_u_at_time, constants);
                    const Real rhs_v = calculate_momentum_rhs_with_forcing<VelocityComponent::v>(u, v, w, i, j, k, forcing_term_v_at_time, constants);
                    const Real rhs_w = calculate_momentum_rhs_with_forcing<VelocityComponent::w>(u, v, w, i, j, k, forcing_term_w_at_time, constants);
*/
                    const Real rhs_u = rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k][0];
                    const Real rhs_v = rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k][1];
                    const Real rhs_w = rhs[i * constants.Ny * constants.Nz + j * constants.Nz + k][2];
                    const Real lu = u(i, j, k);
                    const Real lv = v(i, j, k);
                    const Real lw = w(i, j, k);


                    const Real rhs_u_buffer2 = calculate_momentum_rhs_with_forcing<VelocityComponent::u>(u_buffer2,
                                                                                                         v_buffer2,
                                                                                                         w_buffer2, i,
                                                                                                         j, k,
                                                                                                         forcing_term_u_at_time_2,
                                                                                                         constants);
                    const Real rhs_v_buffer2 = calculate_momentum_rhs_with_forcing<VelocityComponent::v>(u_buffer2,
                                                                                                         v_buffer2,
                                                                                                         w_buffer2, i,
                                                                                                         j, k,
                                                                                                         forcing_term_v_at_time_2,
                                                                                                         constants);
                    const Real rhs_w_buffer2 = calculate_momentum_rhs_with_forcing<VelocityComponent::w>(u_buffer2,
                                                                                                         v_buffer2,
                                                                                                         w_buffer2, i,
                                                                                                         j, k,
                                                                                                         forcing_term_w_at_time_2,
                                                                                                         constants);

                    u(i, j, k) = lu + dt_b1 * rhs_u + dt_b3 * rhs_u_buffer2;
                    v(i, j, k) = lv + dt_b1 * rhs_v + dt_b3 * rhs_v_buffer2;
                    w(i, j, k) = lw + dt_b1 * rhs_w + dt_b3 * rhs_w_buffer2;
                }
            }
        }

        // Swap buffers to finalize the step
    }
}
