#pragma once
#include <cmath>

// Backend implementations of the opaque placeholder kernels the zoomy_core C++
// printer emits (REQ-38). The generated Model.H calls these by bare name, so
// this header must be included BEFORE Model.H in every translation unit that
// instantiates the kernels (unqualified lookup at the template definition point
// then resolves them).
//
// compute_derivative(value, dx, dy, dz): the NON-LOCAL spatial derivative of a
// field. In zoomy_foam this is a field/mesh-level operation (fvc::grad); here
// Model<T>::update_aux_variables is evaluated PER CELL, so a true mesh-aware
// derivative is not available at this call site.
//
// For SME level 0 (the current target) the gradient-aux it fills (dq/dx, dh/dx,
// db/dx, …) are NOT consumed by the hyperbolic update: flux/eigenvalues use only
// Q, the bed slope enters through the nonconservative matrix / well-balanced
// reconstruction, and the source uses only the closure-stress aux. So returning
// 0 here is physically correct for SME(0).
//
// A real mesh-aware compute_derivative (LSQ stencil via Gradient.hpp, evaluated
// at field level before the flux/source sweep) is required by higher SME levels
// and diffusion — tracked as task 0025 Phase C. When added, route the aux
// derivative slots through that field-level path instead of this per-cell stub.
template <typename T>
inline T compute_derivative(T /*value*/, int /*dx*/, int /*dy*/, int /*dz*/) {
    return T(0);
}
