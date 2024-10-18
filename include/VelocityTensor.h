#ifndef VELOCITY_TENSOR_H
#define VELOCITY_TENSOR_H

#include "Constants.h"
#include "Tensor.h"
#include "VectorFunction.h"

namespace mif {

    enum StaggeringStatus {x, y, z, none};

    // Abstract class for a staggered tensor.
    class StaggeredTensor: public Tensor<Real, 3U, size_t> {
      public:
        StaggeredTensor(const Constants &constants, const std::array<size_t, 3U> &in_dimensions, const StaggeringStatus &staggering):
                constants(constants),
                staggering(staggering),
                Tensor(in_dimensions) {}

        const Constants &constants;

        /*!
        * Evaluate the function f, depending on x,y,z, on an index of this tensor.
        * @param i The index for the x direction.
        * @param j The index for the y direction.
        * @param k The index for the z direction.
        * @param f The function to evaluate.
        * @param constants An object containing information on the domain.
        */
        virtual inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                                       const std::function<Real(Real, Real, Real)> &f) const = 0;

        virtual inline Real evaluate_function_at_index(Real time, size_t i, size_t j, size_t k,
                                                       const std::function<Real(Real, Real, Real, Real)> &f) const = 0;

        void print() const;
        void print(const std::function<bool(Real)> &filter) const;

      private:
        const StaggeringStatus staggering;
    };

    // Tensor staggered in the x direction.
    class UTensor: public StaggeredTensor {
      public:
        UTensor(const Constants &constants): 
            StaggeredTensor(constants, {constants.Nx-1, constants.Ny, constants.Nz}, StaggeringStatus::x) {}

        inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                               const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i + constants.dx_over_2, constants.dy * j, constants.dz * k);
        }

        inline Real evaluate_function_at_index(Real time, size_t i, size_t j, size_t k, 
                                               const std::function<Real(Real, Real, Real, Real)> &f) const override {
            return f(time, constants.dx * i + constants.dx_over_2, constants.dy * j, constants.dz * k);
        }
    };

    // Tensor staggered in the y direction.
    class VTensor: public StaggeredTensor {
      public:
        VTensor(const Constants &constants): 
            StaggeredTensor(constants, {constants.Nx, constants.Ny-1, constants.Nz}, StaggeringStatus::y) {}

        inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                               const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i, constants.dy * j + constants.dy_over_2, constants.dz * k);
        }

        inline Real evaluate_function_at_index(Real time, size_t i, size_t j, size_t k, 
                                               const std::function<Real(Real, Real, Real, Real)> &f) const override {
            return f(time, constants.dx * i, constants.dy * j + constants.dy_over_2, constants.dz * k);
        }
    };

    // Tensor staggered in the z direction.
    class WTensor: public StaggeredTensor {
      public:
        WTensor(const Constants &constants):  
            StaggeredTensor(constants, {constants.Nx, constants.Ny, constants.Nz-1}, StaggeringStatus::z) {}

        inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                               const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i, constants.dy * j, constants.dz * k + constants.dz_over_2);
        }

        inline Real evaluate_function_at_index(Real time, size_t i, size_t j, size_t k, 
                                               const std::function<Real(Real, Real, Real, Real)> &f) const override {
            return f(time, constants.dx * i, constants.dy * j, constants.dz * k + constants.dz_over_2);
        }
    };

    // A collection of 3 tensors representing the 3 velocity components.
    class VelocityTensor {
      public:

        UTensor u;
        VTensor v;
        WTensor w;
        std::array<StaggeredTensor*, 3> components;
        const Constants constants;   

        VelocityTensor(const Constants &constants);

        // Swap this tensor's data with another's. Done in constant time.
        void swap_data(VelocityTensor &other);

        // Set the values of a component of velocity calculating a function over all of its points, or 
        // all internal points if include_border is false.
        #define VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(tensor, function, include_border, args...) {        \
            size_t lower_limit, upper_limit_x, upper_limit_y, upper_limit_z;                                \
            const std::array<size_t, 3> &sizes = tensor.sizes();                                            \
            if constexpr (include_border) {lower_limit = 0;} else {lower_limit = 1;};                       \
            if constexpr (include_border) {upper_limit_x = sizes[0];} else {upper_limit_x = sizes[0] - 1;}; \
            if constexpr (include_border) {upper_limit_y = sizes[1];} else {upper_limit_y = sizes[1] - 1;}; \
            if constexpr (include_border) {upper_limit_z = sizes[2];} else {upper_limit_z = sizes[2] - 1;}; \
            for (size_t i = lower_limit; i < upper_limit_x; i++) {                                          \
                for (size_t j = lower_limit; j < upper_limit_y; j++) {                                      \
                    for (size_t k = lower_limit; k < upper_limit_z; k++) {                                  \
                        tensor(i,j,k) = function(args);                                                     \
                    }                                                                                       \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \

        // Set all components of the tensor in all points using the respective components of the function.
        #define VELOCITY_TENSOR_SET_FOR_ALL_POINTS(velocity, f_u, f_v, f_w, include_border, args...) {  \
            VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.u, f_u, include_border, args)              \
            VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.v, f_v, include_border, args)              \
            VELOCITY_TENSOR_ITERATE_OVER_ALL_POINTS(velocity.w, f_w, include_border, args)              \
        }                                                                                               \

        // Set all components of the tensor in all points using the respective components of the function.
        void set(const VectorFunction &f, bool include_border);  

        // Apply Dirichlet boundary conditions to all components of the velocity on all boundaries.
        // The function assumes the velocity field is divergence free.
        void apply_all_dirichlet_bc(Real time);                                       
    };
        

} // mif

#endif // VELOCITY_TENSOR_H