"""
This script generates the C++ code for a given manufactured solution.
It uses the sympy library to symbolically manipulate the expressions and
generate the code through the codegen utility. The produced code is then
printed to the standard output.

See FORCING_TERM_GENERATOR and FORCING_TERM_ARTIFACT from the CMakelists.txt
file for more information on how this script is used in the project.
"""

import sys
import sympy as sp
from sympy import diff
from sympy import simplify as simp

# The codegen callable is not in the sympy namespace automatically.
from sympy.utilities.codegen import codegen


# Create the required symbols.
t, x, y, z = sp.symbols("t, x, y, z")


# Define the laplacian operator.
def lap(p):
    return diff(p, x, 2) + diff(p, y, 2) + diff(p, z, 2)


# Define the divergence operator.
def div(u, v, w):
    return diff(u, x, 1) + diff(v, y, 1) + diff(w, z, 1)


if __name__ == "__main__":
    # This is the manufactured solution we want to generate the code for.
    u = -sp.sin(x) * sp.cos(y) * sp.cos(z) * t
    v = -sp.cos(x) * sp.sin(y) * sp.cos(z) * t
    w = -sp.cos(x) * sp.cos(y) * sp.sin(z) * t
    p = sp.cos(x) * sp.cos(y) * sp.cos(z) * t

    # Check if the laplacian of the pressure is equal to the divergence of the velocity.
    if simp(div(u, v, w) - lap(p)) != 0:
        sys.stderr.write(
            "[WARNING]: the proposed manufactured solution is inconsistent. The result is meaningless.\n"
        )

    dp_dx = simp(diff(p, x))
    dp_dy = simp(diff(p, y))
    dp_dz = simp(diff(p, z))

    if (
        simp(dp_dx.subs(x, 0)) != 0
        or simp(dp_dx.subs(x, 2 * sp.pi)) != 0
        or simp(dp_dy.subs(y, 0)) != 0
        or simp(dp_dy.subs(y, 2 * sp.pi)) != 0
        or simp(dp_dz.subs(z, 0)) != 0
        or simp(dp_dz.subs(z, 2 * sp.pi)) != 0
    ):
        sys.stderr.write(
            "[WARNING]: the proposed manufactured solution does not have homogeneous Neumann boundary conditions on [0, 2pi]^3.\n"
        )

    # Generate the C code through sympy's codegen utility.
    [(c_name, c_code), (h_name, c_header)] = codegen(
        [
            ("u_exact_p_test", u),
            ("v_exact_p_test", v),
            ("w_exact_p_test", w),
            ("p_exact_p_test", p),
            ("dp_dx_exact_p_test", dp_dx),
            ("dp_dy_exact_p_test", dp_dy),
            ("dp_dz_exact_p_test", dp_dz),
        ],
        language="C99",
        prefix="ManufacturedPressure",
        project="mif",
        header=True,
        empty=True,
        argument_sequence=(t, x, y, z),
    )

    # This code will be fed to the `manufsol.cpp` file.
    print(c_code)
