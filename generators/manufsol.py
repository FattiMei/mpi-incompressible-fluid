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


if __name__ == "__main__":
    # Parameters.
    a = sp.pi / 4
    d = sp.pi / 2

    # This is the manufactured solution we want to generate the code for.
    u = (
        -a
        * (
            sp.exp(a * x) * sp.sin(a * y + d * z)
            + sp.exp(a * z) * sp.cos(a * x + d * y)
        )
        * sp.exp(-d * d * t / Re)
    )
    v = (
        -a
        * (
            sp.exp(a * y) * sp.sin(a * z + d * x)
            + sp.exp(a * x) * sp.cos(a * y + d * z)
        )
        * sp.exp(-d * d * t / Re)
    )
    w = (
        -a
        * (
            sp.exp(a * z) * sp.sin(a * x + d * y)
            + sp.exp(a * y) * sp.cos(a * z + d * x)
        )
        * sp.exp(-d * d * t / Re)
    )
    p = (
        -a
        * a
        / 2
        * (
            sp.exp(2 * a * x)
            + sp.exp(2 * a * y)
            + sp.exp(2 * a * z)
            + 2 * sp.sin(a * x + d * y) * sp.cos(a * z + d * x) * sp.exp(a * (y + z))
            + 2 * sp.sin(a * y + d * z) * sp.cos(a * x + d * y) * sp.exp(a * (z + x))
            + 2 * sp.sin(a * z + d * x) * sp.cos(a * y + d * z) * sp.exp(a * (x + y))
        )
        * sp.exp(-2 * d * d * t / Re)
    )

    if simp(diff(u, x) + diff(v, y) + diff(w, z)) != 0:
        sys.stderr.write(
            "[WARNING]: the proposed manufactured solution is not divergence free, the solution will be wrong.\n"
        )

    if (
        simp(
            diff(u, t)
            + (u * diff(u, x) + v * diff(u, y) + w * diff(u, z))
            - 1 / Re * lap(u)
            + diff(p, x)
        )
        != 0
        or simp(
            diff(v, t)
            + (u * diff(v, x) + v * diff(v, y) + w * diff(v, z))
            - 1 / Re * lap(v)
            + diff(p, y)
        )
        != 0
        or simp(
            diff(w, t)
            + (u * diff(w, x) + v * diff(w, y) + w * diff(w, z))
            - 1 / Re * lap(w)
            + diff(p, z)
        )
        != 0
    ):
        sys.stderr.write(
            "[WARNING]: the proposed manufactured solution does not satisfy the momentum equation, the solution will be wrong.\n"
        )

    dp_dx = simp(diff(p, x))
    dp_dy = simp(diff(p, y))
    dp_dz = simp(diff(p, z))

    # Generate the C code through sympy's codegen utility.
    [(c_name, c_code), (h_name, c_header)] = codegen(
        [
            ("u_exact", u),
            ("v_exact", v),
            ("w_exact", w),
            ("p_exact", p),
            ("dp_dx_exact", dp_dx),
            ("dp_dy_exact", dp_dy),
            ("dp_dz_exact", dp_dz),
        ],
        language="C99",
        prefix="Manufactured",
        project="mif",
        header=True,
        empty=True,
        argument_sequence=(t, x, y, z),
        global_vars=[Re],
    )

    # This code will be fed to the `manufsol.cpp` file.
    print(c_code)
