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
t, x, y, z, Re = sp.symbols("t, x, y, z, Reynolds")


# Define the laplacian operator.
def lap(u):
    return diff(u, x, 2) + diff(u, y, 2) + diff(u, z, 2)


# Define the manufactured solution function.
# Note: the asterisk (*) in the function signature indicates that the arguments
# are keyword-only. This is to minimize the risk of passing wrong arguments.
def manufsol(*, u, v, w):
    # Safety check: the proposed manufactured solution must be divergence free.
    if diff(u, x) + diff(v, y) + diff(w, z) != 0:
        w = sp.integrate(simp(sp.diff(u, x) + sp.diff(v, y)), z)
        sys.stderr.write(
            "[WARNING]: the proposed manufactured solution was \
                         not divergence free, so the w term was overridden.\n"
        )

    fx = simp(
        diff(u, t) + u * diff(u, x) + v * diff(u, y) + w * diff(u, z) - lap(u) / Re
    )
    fy = simp(
        diff(v, t) + u * diff(v, x) + v * diff(v, y) + w * diff(v, z) - lap(v) / Re
    )
    fz = simp(
        diff(w, t) + u * diff(w, x) + v * diff(w, y) + w * diff(w, z) - lap(w) / Re
    )

    return (u, v, w), (fx, fy, fz)


if __name__ == "__main__":
    # This is the manufactured solution we want to generate the code for.
    u = sp.sin(x) * sp.cos(y) * sp.sin(z) * sp.sin(t)
    v = sp.cos(x) * sp.sin(y) * sp.sin(z) * sp.sin(t)
    w = 2 * sp.cos(x) * sp.cos(y) * sp.cos(z) * sp.sin(t)

    # Until we tackle the pressure term, we can ignore it.
    (u, v, w), (fx, fy, fz) = manufsol(u=u, v=v, w=w)

    # Generate the C code through sympy's codegen utility.
    [(c_name, c_code), (h_name, c_header)] = codegen(
        [
            ("u_exact", u),
            ("v_exact", v),
            ("w_exact", w),
            ("forcing_x", fx),
            ("forcing_y", fy),
            ("forcing_z", fz),
        ],
        language="C99",
        prefix="ManufacturedVelocity",
        project="mif",
        header=True,
        empty=True,
        argument_sequence=(t, x, y, z),
        global_vars=[Re],
    )

    # This code will be fed to the `manufsol.cpp` file.
    print(c_code)
