#include "VelocityTensor.h"

namespace mif {

    VelocityTensor::VelocityTensor(const Constants &constants): 
        constants(constants),
        u(constants),
        v(constants),
        w(constants),
        components({&this->u,&this->v,&this->w}) {}

    void VelocityTensor::swap_data(VelocityTensor &other) {
        for (size_t component = 0; component < components.size(); component++) {
            components[component]->swap_data(*(other.components[component]));
        }
    }

    void VelocityTensor::set(const IndexVectorFunction &f, bool include_border) {
        const size_t lower_limit = include_border ? 0 : 1;
        for (size_t component = 0; component < components.size(); component++) {
            StaggeredTensor *tensor = components[component];
            const std::array<size_t, 3> sizes = tensor->sizes();
            const std::function<Real(size_t, size_t, size_t)> *func = f.components[component];
            const size_t upper_limit_i = include_border ? sizes[0] : sizes[0]-1;
            const size_t upper_limit_j = include_border ? sizes[1] : sizes[1]-1;
            const size_t upper_limit_k = include_border ? sizes[2] : sizes[2]-1;

            for (size_t i = lower_limit; i < upper_limit_i; i++) {
                for (size_t j = lower_limit; j < upper_limit_j; j++) {
                    for (size_t k = lower_limit; k < upper_limit_k; k++) {
                        (*tensor)(i,j,k) = (*func)(i, j, k);
                    }
                }
            }
        }
    }

    void VelocityTensor::set(const VectorFunction &f, bool include_border) {
        std::array<std::function<Real(size_t, size_t, size_t)>, 3> functions{};
        for (size_t component = 0; component < components.size(); component++) {
            StaggeredTensor *tensor = components[component];
            const std::function<Real(Real, Real, Real)> *func = f.components[component];
            functions[component] = [func, tensor](size_t i, size_t j, size_t k) {
                    return tensor->evaluate_function_at_index(i, j, k, *func);
                };
        }
        IndexVectorFunction f_index(functions[0], functions[1], functions[2]);
        set(f_index, include_border);
    }
    
    void VelocityTensor::apply_all_dirichlet_bc(const VectorFunction &exact_solution) {
        for (size_t component = 0; component < components.size(); component++) {
            StaggeredTensor *tensor = components[component];
            const std::array<size_t, 3> sizes = tensor->sizes();
            const std::function<Real(Real, Real, Real)> *func = exact_solution.components[component];

            // Face 1: z=0
            if (component == 2) {
                for (size_t i = 1; i < constants.Nx-1; i++) {
                    for (size_t j = 1; j < constants.Ny-1; j++) {
                        const Real w_at_boundary = (*func)(i*constants.dx, j*constants.dy, 0);
                        const Real du_dx = (u.evaluate_function_at_index(i,j,0,*exact_solution.components[0]) - u.evaluate_function_at_index(i-1,j,0,*exact_solution.components[0])) * constants.one_over_dx;
                        const Real dv_dy = (v.evaluate_function_at_index(i,j,0,*exact_solution.components[1]) - v.evaluate_function_at_index(i,j-1,0,*exact_solution.components[1])) * constants.one_over_dy;
                        w(i, j, 0) = w_at_boundary - constants.dz_over_2 * (du_dx + dv_dy);
                    }
                }
            } else {
                for (size_t i = 0; i < sizes[0]; i++) {
                    for (size_t j = 0; j < sizes[1]; j++) {
                        (*tensor)(i, j, 0) = tensor->evaluate_function_at_index(i, j, 0, *func);
                    }
                }
            }

            // Face 2: z=z_max
            if (component == 2) {
                for (size_t i = 1; i < constants.Nx-1; i++) {
                    for (size_t j = 1; j < constants.Ny-1; j++) {
                        const Real w_at_boundary = (*func)(i*constants.dx, j*constants.dy, constants.z_size);
                        const Real du_dx = (u.evaluate_function_at_index(i,j,constants.Nz-1,*exact_solution.components[0]) - u.evaluate_function_at_index(i-1,j,constants.Nz-1,*exact_solution.components[0])) * constants.one_over_dx;
                        const Real dv_dy = (v.evaluate_function_at_index(i,j,constants.Nz-1,*exact_solution.components[1]) - v.evaluate_function_at_index(i,j-1,constants.Nz-1,*exact_solution.components[1])) * constants.one_over_dy;
                        w(i, j, constants.Nz-2) = w_at_boundary + constants.dz_over_2 * (du_dx + dv_dy);
                    }
                }
            } else {
                for (size_t i = 0; i < sizes[0]; i++) {
                    for (size_t j = 0; j < sizes[1]; j++) {
                        (*tensor)(i, j, constants.Nz-1) = tensor->evaluate_function_at_index(i, j, constants.Nz-1, *func);
                    }
                }
            }

            // Face 3: y=0
            if (component == 1) {
                for (size_t i = 1; i < constants.Nx-1; i++) {
                    for (size_t k = 1; k < constants.Nz-1; k++) {
                        const Real v_at_boundary = (*func)(i*constants.dx, 0, k*constants.dz);
                        const Real du_dx = (u.evaluate_function_at_index(i,0,k,*exact_solution.components[0]) - u.evaluate_function_at_index(i-1,0,k,*exact_solution.components[0])) * constants.one_over_dx;
                        const Real dw_dz = (w.evaluate_function_at_index(i,0,k,*exact_solution.components[2]) - w.evaluate_function_at_index(i,0,k-1,*exact_solution.components[2])) * constants.one_over_dz;
                        v(i, 0, k) = v_at_boundary - constants.dy_over_2 * (du_dx + dw_dz);
                    }
                }
            } else {
                for (size_t i = 0; i < sizes[0]; i++) {
                    for (size_t k = 0; k < sizes[2]; k++) {
                        (*tensor)(i, 0, k) = tensor->evaluate_function_at_index(i, 0, k, *func);
                    }
                }
            }

            // Face 4: y=y_max
            if (component == 1) {
                for (size_t i = 1; i < constants.Nx-1; i++) {
                    for (size_t k = 1; k < constants.Nz-1; k++) {
                        const Real v_at_boundary = (*func)(i*constants.dx, constants.y_size, k*constants.dz);
                        const Real du_dx = (u.evaluate_function_at_index(i,constants.Ny-1,k,*exact_solution.components[0]) - u.evaluate_function_at_index(i-1,constants.Ny-1,k,*exact_solution.components[0])) * constants.one_over_dx;
                        const Real dw_dz = (w.evaluate_function_at_index(i,constants.Ny-1,k,*exact_solution.components[2]) - w.evaluate_function_at_index(i,constants.Ny-1,k-1,*exact_solution.components[2])) * constants.one_over_dz;
                        v(i, constants.Ny-2, k) = v_at_boundary + constants.dy_over_2 * (du_dx + dw_dz);
                    }
                }
            } else {
                for (size_t i = 0; i < sizes[0]; i++) {
                    for (size_t k = 0; k < sizes[2]; k++) {
                        (*tensor)(i, constants.Ny-1, k) = tensor->evaluate_function_at_index(i, constants.Ny-1, k, *func);
                    }
                }
            }

            // Face 5: x=0
            if (component == 0) {
                for (size_t j = 1; j < constants.Ny-1; j++) {
                    for (size_t k = 1; k < constants.Nz-1; k++) {
                        const Real u_at_boundary = (*func)(0, j*constants.dy, k*constants.dz);
                        const Real dv_dy = (v.evaluate_function_at_index(0,j,k,*exact_solution.components[1]) - v.evaluate_function_at_index(0,j-1,k,*exact_solution.components[1])) * constants.one_over_dy;
                        const Real dw_dz = (w.evaluate_function_at_index(0,j,k,*exact_solution.components[2]) - w.evaluate_function_at_index(0,j,k-1,*exact_solution.components[2])) * constants.one_over_dz;
                        u(0, j, k) = u_at_boundary - constants.dx_over_2 * (dv_dy + dw_dz);
                    }
                }
            } else {
                for (size_t j = 0; j < sizes[1]; j++) {
                    for (size_t k = 0; k < sizes[2]; k++) {
                        (*tensor)(0, j, k) = tensor->evaluate_function_at_index(0, j, k, *func);
                    }
                }
            }

            // Face 6: x=x_max
            if (component == 0) {
                for (size_t j = 1; j < constants.Ny-1; j++) {
                    for (size_t k = 1; k < constants.Nz-1; k++) {
                        const Real u_at_boundary = (*func)(constants.x_size, j*constants.dy, k*constants.dz);
                        const Real dv_dy = (v.evaluate_function_at_index(constants.Nx-1,j,k,*exact_solution.components[1]) - v.evaluate_function_at_index(constants.Nx-1,j-1,k,*exact_solution.components[1])) * constants.one_over_dy;
                        const Real dw_dz = (w.evaluate_function_at_index(constants.Nx-1,j,k,*exact_solution.components[2]) - w.evaluate_function_at_index(constants.Nx-1,j,k-1,*exact_solution.components[2])) * constants.one_over_dz;
                        u(constants.Nx-2, j, k) = u_at_boundary + constants.dx_over_2 * (dv_dy + dw_dz);
                    }
                }
            } else {
                for (size_t j = 0; j < sizes[1]; j++) {
                    for (size_t k = 0; k < sizes[2]; k++) {
                        (*tensor)(constants.Nx-1, j, k) = tensor->evaluate_function_at_index(constants.Nx-1, j, k, *func);
                    }
                }
            }
        }
    }
}