#ifndef VELOCITY_TENSOR_H
#define VELOCITY_TENSOR_H

#include "Constants.h"
#include "Tensor.h"
#include "VectorFunction.h"

namespace mif {

    /*!
    * @class StaggeredTensor
    * @brief A tensor with staggered components.
    * 
    * This class represents a tensor with staggered components, i.e., components
    * that are offset by half a grid cell in one of the directions. We further
    * inherit from this class to create tensors staggered in the x, y, and z
    * directions, see UTensor, VTensor, and WTensor.
    * 
    * @param constants An object containing information on the domain.
    * @param in_dimensions The dimensions of the tensor.
    */
    class StaggeredTensor: public Tensor<Real, 3U, size_t> {
      public:
        StaggeredTensor(const Constants &constants, const std::array<size_t, 3U> &in_dimensions):
                constants(constants),
                Tensor(in_dimensions) {}

        const Constants &constants;

        /*!
        * Evaluate the function f, depending on x,y,z, on an index of this
        * tensor.
        * 
        * @param i The index for the x direction.
        * @param j The index for the y direction.
        * @param k The index for the z direction.
        * @param f The function to evaluate.
        * @param constants An object containing information on the domain.
        */
        virtual inline Real evaluate_function_at_index(
            size_t i, size_t j, size_t k,
            const std::function<Real(Real, Real, Real)> &f) const = 0;

        virtual inline Real evaluate_function_at_index(
            Real time, size_t i, size_t j, size_t k,
            const std::function<Real(Real, Real, Real, Real)> &f) const = 0;

        // A debug function to print the tensor.
        void print() const;
        void print(const std::function<bool(Real)> &filter) const;
    };


    // Tensor staggered in the x direction.
    class UTensor: public StaggeredTensor {
      public:
        UTensor(const Constants &constants): 
            StaggeredTensor(constants, {constants.Nx-1, constants.Ny, constants.Nz}) {}

        inline Real evaluate_function_at_index(
                size_t i, size_t j, size_t k,
                const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i + constants.dx_over_2,
                     constants.dy * j, constants.dz * k);
        }

        inline Real evaluate_function_at_index(
                Real time, size_t i, size_t j, size_t k,
                const std::function<Real(Real, Real, Real, Real)> &f) const override {
            return f(time, constants.dx * i + constants.dx_over_2,
                     constants.dy * j, constants.dz * k);
        }
    };

    // Tensor staggered in the y direction.
    class VTensor: public StaggeredTensor {
      public:
        VTensor(const Constants &constants): 
            StaggeredTensor(constants, {constants.Nx, constants.Ny-1, constants.Nz}) {}

        inline Real evaluate_function_at_index(
                size_t i, size_t j, size_t k,
                const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i, constants.dy * j +
                     constants.dy_over_2, constants.dz * k);
        }

        inline Real evaluate_function_at_index(
                Real time, size_t i, size_t j, size_t k,
                const std::function<Real(Real, Real, Real, Real)> &f) const override {
            return f(time, constants.dx * i, constants.dy * j +
                     constants.dy_over_2, constants.dz * k);
        }
    };

    // Tensor staggered in the z direction.
    class WTensor: public StaggeredTensor {
      public:
        WTensor(const Constants &constants):  
            StaggeredTensor(constants, {constants.Nx, constants.Ny, constants.Nz-1}) {}

        inline Real evaluate_function_at_index(
                size_t i, size_t j, size_t k, 
                const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i, constants.dy * j,
                     constants.dz * k + constants.dz_over_2);
        }

        inline Real evaluate_function_at_index(
                Real time, size_t i, size_t j, size_t k,
                const std::function<Real(Real, Real, Real, Real)> &f) const override {
            return f(time, constants.dx * i, constants.dy * j,
                     constants.dz * k + constants.dz_over_2);
        }
    };


    /*!
    * @class VelocityTensor
    * @brief A collection of 3 tensors representing the 3 velocity components.
    * 
    * This class is mainly used to abstract the velocity field into a single
    * object, which can be easily manipulated and passed around.
    */
    class VelocityTensor {
      public:
        UTensor u;
        VTensor v;
        WTensor w;
        std::array<StaggeredTensor*, 3> components;
        const Constants constants;   

        VelocityTensor(const Constants &constants);

        // Swap this tensor's data with another's in constant time by swapping
        // pointers.
        void swap_data(VelocityTensor &other);

        // Set all components of the tensor in all points using the respective
        // components of the function.
        void set(const VectorFunction &f, bool include_border);  

        // Apply Dirichlet boundary conditions to all components of the velocity
        // on all boundaries. The function assumes the velocity field is
        // divergence free.
        void apply_all_dirichlet_bc(Real time);                                       
    };
        

} // mif

#endif // VELOCITY_TENSOR_H