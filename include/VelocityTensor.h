#ifndef VELOCITY_TENSOR_H
#define VELOCITY_TENSOR_H

#include <cassert>
#include "Constants.h"
#include "StaggeringStatus.h"
#include "Tensor.h"
#include "VectorFunction.h"

namespace mif {

    // Abstract class.
    class StaggeredTensor: public Tensor<Real, 3U, size_t> {
      public:
        StaggeredTensor(const Constants &constants, const std::array<size_t, 3U> &in_dimensions, const StaggeringStatus &staggering):
                constants(constants),
                staggering(staggering),
                Tensor(in_dimensions) {}

        /*!
        * Return the staggering direction of this tensor.
        */
        StaggeringStatus get_staggering() const {
            return staggering;
        }

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

      private:
        const StaggeringStatus staggering;
      protected:
        const Constants &constants;
    };

    class UTensor: public StaggeredTensor {
      public:
        UTensor(const Constants &constants): 
            StaggeredTensor(constants, {constants.Nx-1, constants.Ny, constants.Nz}, StaggeringStatus::x) {}

        inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                                const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i + constants.dx_over_2, constants.dy * j, constants.dz * k);
        }
    };

    class VTensor: public StaggeredTensor {
      public:
        VTensor(const Constants &constants): 
            StaggeredTensor(constants, {constants.Nx, constants.Ny-1, constants.Nz}, StaggeringStatus::y) {}

        inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                                const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i, constants.dy * j + constants.dy_over_2, constants.dz * k);
        }
    };

    class WTensor: public StaggeredTensor {
      public:
        WTensor(const Constants &constants):  
            StaggeredTensor(constants, {constants.Nx, constants.Ny, constants.Nz-1}, StaggeringStatus::z) {}

        inline Real evaluate_function_at_index(size_t i, size_t j, size_t k, 
                                                const std::function<Real(Real, Real, Real)> &f) const override {
            return f(constants.dx * i, constants.dy * j, constants.dz * k + constants.dz_over_2);
        }
    };

    class VelocityTensor {
      public:

        class IndexVectorFunction {
          public:

            IndexVectorFunction(const std::function<Real(size_t, size_t, size_t)> f_u,
                                const std::function<Real(size_t, size_t, size_t)> f_v,
                                const std::function<Real(size_t, size_t, size_t)> f_w);

            const std::function<Real(size_t, size_t, size_t)> f_u;
            const std::function<Real(size_t, size_t, size_t)> f_v;
            const std::function<Real(size_t, size_t, size_t)> f_w;
            const std::array<const std::function<Real(size_t, size_t, size_t)>*, 3> components;

            static IndexVectorFunction identity(const VelocityTensor &tensor);

            IndexVectorFunction operator+(const IndexVectorFunction other);
            IndexVectorFunction operator*(Real scalar);
        };

        UTensor u;
        VTensor v;
        WTensor w;
        std::array<StaggeredTensor*, 3> components;
        const Constants constants;   

        VelocityTensor(const Constants &constants);

        void swap_data(VelocityTensor &other);

        // Set all components of the tensor using the respective components of the function.
        void set(const IndexVectorFunction &f, bool include_border);

        // Set all components of the tensor using the respective components of the function.
        void set(const VectorFunction &f, bool include_border);   

        // Apply Dirichlet boundary conditions to all components of the velocity on all boundaries.
        // The function assumes the velocity field is divergence free.
        void apply_all_dirichlet_bc(const VectorFunction &exact_solution);                                            
    };
        

} // mif

#endif // VELOCITY_TENSOR_H