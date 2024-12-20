#include "PressureEquation.h"
#include "PressureSolverStructures.h"
#include "StaggeredTensorMacros.h"
#include "VelocityDivergence.h"

namespace mif {

// Add the non-homogeneous term to the boundaries of the rhs.
void apply_non_homogeneous_neumann(StaggeredTensor &rhs, const VectorFunction &exact_pressure_gradient) {
    const Constants &constants = rhs.constants;
    std::array<size_t, 3> sizes = rhs.sizes();

    // Face 1: z=z_min
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, j, 0) += rhs.evaluate_function_at_index(i, j, 0, exact_pressure_gradient.f_w) * 2.0 * constants.one_over_dz;
        }
    }

    // Face 2: z=z_max
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, j, sizes[2]-1) -= rhs.evaluate_function_at_index(i, j, sizes[2]-1, exact_pressure_gradient.f_w) * 2.0 * constants.one_over_dz;
        }
    }

    // Face 3: y=y_min
    for (size_t k = 0; k < sizes[2]; k++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, 0, k) += rhs.evaluate_function_at_index(i, 0, k, exact_pressure_gradient.f_v) * 2.0 * constants.one_over_dy;
        }
    }

    // Face 4: y=y_max
    for (size_t k = 0; k < sizes[2]; k++) {
        for (size_t i = 0; i < sizes[0]; i++) {
            rhs(i, sizes[1]-1, k) -= rhs.evaluate_function_at_index(i, sizes[1]-1, k, exact_pressure_gradient.f_v) * 2.0 * constants.one_over_dy;
        }
    }

    // Face 5: x=x_min
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
            rhs(0, j, k) += rhs.evaluate_function_at_index(0, j, k, exact_pressure_gradient.f_u) * 2.0 * constants.one_over_dx;
        }
    }

    // Face 6: x=x_max
    for (size_t j = 0; j < sizes[1]; j++) {
        for (size_t k = 0; k < sizes[2]; k++) {
            rhs(sizes[0]-1, j, k) -= rhs.evaluate_function_at_index(sizes[0]-1, j, k, exact_pressure_gradient.f_u) * 2.0 * constants.one_over_dx;
        }
    }
}

// Compute the rhs for homogeneous Neumann boundary conditions.
void compute_rhs_homogeneous_neumann(StaggeredTensor &rhs, const VelocityTensor &velocity, Real dt) {
    STAGGERED_TENSOR_ITERATE_OVER_ALL_POINTS(rhs, true, rhs(i,j,k) = calculate_velocity_divergence(velocity,i,j,k) / dt;)
}
    
inline Real compute_eigenvalue_neumann(size_t index, Real delta, size_t N_domains) {
    return 2.0 * (std::cos(M_PI * index / N_domains) - 1.0) / (delta*delta);
}

// Solve the pressure equation with Neumann boundary conditions.
// "pressure" contains the rhs, and will be replaced with the pressure.
void solve_pressure_equation_neumann(StaggeredTensor &pressure,
                                     const VelocityTensor &velocity,
                                     PressureSolverStructures &structures) {
    const Constants &constants = velocity.constants;

    // Execute type 1 DCT along direction x and transpose from (z,y,x) to (x,z,y).
    // Note: major refers to the last index of the triple, the triple is in the iterating order.
    for (size_t k = 0; k < constants.Nz; k++) {
        for (size_t j = 0; j < constants.Ny; j++) {
            const size_t base_index = constants.Nx*constants.Ny * k + constants.Nx * j;

            // Copy the original data.
            memcpy(structures.buffer_x, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*constants.Nx);

            // Execute the fft.
            fftw_execute(structures.fft_plan_x);

            // Copy the transformed data.
            memcpy(static_cast<Real *>(pressure.raw_data())+base_index, structures.buffer_x, sizeof(Real)*constants.Nx);
        }
    }
    structures.c2d.transposeX2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 DCT along direction y and transpose from (x,z,y) to (y,x,z).
    for (size_t i = 0; i < constants.Nx; i++) {
        for (size_t k = 0; k < constants.Nz; k++) {
            const size_t base_index = constants.Nz*constants.Ny * i + constants.Ny * k;

            // Copy the original data.
            memcpy(structures.buffer_y, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*constants.Ny);

            // Execute the fft.
            fftw_execute(structures.fft_plan_y);

            // Copy the transformed data.
            memcpy(static_cast<Real *>(pressure.raw_data())+base_index, structures.buffer_y, sizeof(Real)*constants.Ny);
        }
    }
    structures.c2d.transposeY2Z_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 DCT along direction z while in indexing (y,x,z), do not transpose.
    for (size_t j = 0; j < constants.Ny; j++) {
        for (size_t i = 0; i < constants.Nx; i++) {
            const size_t base_index = constants.Nz*constants.Nx * j + constants.Nz * i;

            // Copy the original data.
            memcpy(structures.buffer_z, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*constants.Nz);

            // Execute the fft.
            fftw_execute(structures.fft_plan_z);

            // Copy the transformed data.
            memcpy(static_cast<Real *>(pressure.raw_data())+base_index, structures.buffer_z, sizeof(Real)*constants.Nz);
        }
    }


    // Divide by eigenvalues.
    for (size_t j = 0; j < constants.Ny; j++) {
        const Real lambda_2 = compute_eigenvalue_neumann(j, constants.dy, constants.Ny_domains_global);
        const Real base_index_1 = j*constants.Nx*constants.Nz;
        for (size_t i = 0; i < constants.Nx; i++) {
            const Real base_index_2 = base_index_1 + i*constants.Nz;
            const Real lambda_1 = compute_eigenvalue_neumann(i, constants.dx, constants.Nx_domains);
            for (size_t k = 0; k < constants.Nz; k++) {
                const Real lambda_3 = compute_eigenvalue_neumann(k, constants.dz, constants.Nz_domains_global);
                pressure(base_index_2 + k) /= (lambda_1 + lambda_2 + lambda_3);
            }
        }
    }
    pressure(0,0,0) = 0;


    // Execute type 1 IDCT along direction z, transpose from (y,x,z) to (x,z,y).
    for (size_t j = 0; j < constants.Ny; j++) {
        for (size_t i = 0; i < constants.Nx; i++) {
            const size_t base_index = constants.Nz*constants.Nx * j + constants.Nz * i;

            // Copy the original data.
            memcpy(structures.buffer_z, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*constants.Nz);

            // Execute the fft.
            fftw_execute(structures.fft_plan_z);

            // Copy the transformed data.
            for (size_t k = 0; k < constants.Nz; k++) {
                pressure(base_index+k) = structures.buffer_z[k] / (2.0*constants.Nz_domains_global);
            }
        }
    }
    structures.c2d.transposeZ2Y_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 IDCT along direction y and transpose from (x,z,y) to (z,y,x).
    for (size_t i = 0; i < constants.Nx; i++) {
        for (size_t k = 0; k < constants.Nz; k++) {
            const size_t base_index = constants.Nz*constants.Ny * i + constants.Ny * k;

            // Copy the original data.
            memcpy(structures.buffer_y, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*constants.Ny);

            // Execute the fft.
            fftw_execute(structures.fft_plan_y);

            // Copy the transformed data.
            for (size_t j = 0; j < constants.Ny; j++){
                pressure(base_index+j) = structures.buffer_y[j] / (2.0*constants.Ny_domains_global);
            }
        }
    }
    structures.c2d.transposeY2X_MajorIndex(static_cast<Real *>(pressure.raw_data()), static_cast<Real *>(pressure.raw_data()));


    // Execute type 1 IDCT along direction x while in indexing (z,y,x), do not transpose.
    for (size_t k = 0; k < constants.Nz; k++) {
        for (size_t j = 0; j < constants.Ny; j++) {
            const size_t base_index = constants.Nx*constants.Ny * k + constants.Nx * j;

            // Copy the original data.
            memcpy(structures.buffer_x, static_cast<Real *>(pressure.raw_data())+base_index, sizeof(Real)*constants.Nx);

            // Execute the fft.
            fftw_execute(structures.fft_plan_x);

            // Copy the transformed data.
            for (size_t i = 0; i < constants.Nx; i++) {
                pressure(base_index+i) = structures.buffer_x[i] / (2.0*constants.Nx_domains);
            }
        }
    }
}

void solve_pressure_equation_homogeneous_neumann(StaggeredTensor &pressure,
                                                 const VelocityTensor &velocity,
                                                 PressureSolverStructures &structures,
                                                 Real dt) {
    compute_rhs_homogeneous_neumann(pressure, velocity, dt);
    solve_pressure_equation_neumann(pressure, velocity, structures);
}

void solve_pressure_equation_non_homogeneous_neumann(StaggeredTensor &pressure, 
                                                     const VelocityTensor &velocity,
                                                     const VectorFunction &exact_pressure_gradient,
                                                     PressureSolverStructures &structures,
                                                     Real dt) {
    compute_rhs_homogeneous_neumann(pressure, velocity, dt);
    apply_non_homogeneous_neumann(pressure, exact_pressure_gradient);
    solve_pressure_equation_neumann(pressure, velocity, structures);
}

void adjust_pressure(StaggeredTensor &pressure,
                     const std::function<Real(Real, Real, Real)> &exact_pressure) {
    // Compute the constant difference.
    const Constants &constants = pressure.constants;
    const Real difference = exact_pressure(constants.min_x_global, constants.min_y_global, constants.min_z_global) - pressure(0,0,0);

    // Remove the difference from all points.
    for (size_t k = 0; k < constants.Nz; k++) {
        for (size_t j = 0; j < constants.Ny; j++) {
            for (size_t i = 0; i < constants.Nx; i++) {
                pressure(i,j,k) += difference;
            }
        }
    }
}

} // mif