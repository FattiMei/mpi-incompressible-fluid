#include "VelocityTensor.h"

namespace mif {
    
    VelocityTensor::IndexVectorFunction::IndexVectorFunction(const std::function<Real(size_t, size_t, size_t)> f_u,
                                                             const std::function<Real(size_t, size_t, size_t)> f_v,
                                                             const std::function<Real(size_t, size_t, size_t)> f_w):
            f_u(f_u), f_v(f_v), f_w(f_w),
            components{&this->f_u, &this->f_v, &this->f_w} {};

    VelocityTensor::IndexVectorFunction VelocityTensor::IndexVectorFunction::identity(const VelocityTensor &tensor) {
        const std::function<Real(size_t, size_t, size_t)> &identity_u =
            [&tensor](size_t i, size_t j, size_t k) {
                return tensor.u(i,j,k);
            };
        const std::function<Real(size_t, size_t, size_t)> &identity_v =
            [&tensor](size_t i, size_t j, size_t k) {
                return tensor.v(i,j,k);
            };
        const std::function<Real(size_t, size_t, size_t)> &identity_w =
            [&tensor](size_t i, size_t j, size_t k) {
                return tensor.w(i,j,k);
            };
        return IndexVectorFunction(identity_u, identity_v, identity_w);
    }

    VelocityTensor::IndexVectorFunction VelocityTensor::IndexVectorFunction::operator+(const IndexVectorFunction other) {
        const std::function<Real(size_t, size_t, size_t)> &sum_u =
            [*this, other](size_t i, size_t j, size_t k) {
                return f_u(i,j,k) + other.f_u(i,j,k);
            };
        const std::function<Real(size_t, size_t, size_t)> &sum_v =
            [*this, other](size_t i, size_t j, size_t k) {
                return f_v(i,j,k) + other.f_v(i,j,k);
            };
        const std::function<Real(size_t, size_t, size_t)> &sum_w =
            [*this, other](size_t i, size_t j, size_t k) {
                return f_w(i,j,k) + other.f_w(i,j,k);
            };
        return IndexVectorFunction(sum_u,sum_v,sum_w);
    }

    VelocityTensor::IndexVectorFunction  VelocityTensor::IndexVectorFunction::operator*(Real scalar) {
        const std::function<Real(size_t, size_t, size_t)> &scaled_u =
            [*this, scalar](size_t i, size_t j, size_t k) {
                return f_u(i,j,k) * scalar;
            };
        const std::function<Real(size_t, size_t, size_t)> &scaled_v =
            [*this, scalar](size_t i, size_t j, size_t k) {
                return f_v(i,j,k) * scalar;
            };
        const std::function<Real(size_t, size_t, size_t)> &scaled_w =
            [*this, scalar](size_t i, size_t j, size_t k) {
                return f_w(i,j,k) * scalar;
            };
        return IndexVectorFunction(scaled_u,scaled_v,scaled_w);
    }

} // mif