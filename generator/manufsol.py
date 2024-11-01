import sys
import sympy as sp
from sympy import diff
from sympy import simplify as simp
from sympy.utilities.codegen import codegen

# Create the required symbols.
t, x, y, z, Re = sp.symbols('t, x, y, z, Reynolds')

# Define the laplacian operator.
def lap(u):
    return diff(u, x, 2) + diff(u, y, 2) + diff(u, z, 2)

# Define the manufactured solution function.
def manufsol(*, u, v, w, p, ignore_pressure=False):
    if ignore_pressure:
        p = 0

    if diff(u, x) + diff(v, y) + diff(w, z) != 0:
        w = sp.integrate(simp(sp.diff(u, x) + sp.diff(v, y)), z)
        sys.stderr.write(
            '[WARNING]: the proposed manufactured solution was not divergence free, so the w term was overridden.\n')

    fx = simp(diff(u, t) + u * diff(u, x) + v * diff(u, y) + w * diff(u, z) + diff(p, x) - lap(u) / Re)
    fy = simp(diff(v, t) + u * diff(v, x) + v * diff(v, y) + w * diff(v, z) + diff(p, y) - lap(v) / Re)
    fz = simp(diff(w, t) + u * diff(w, x) + v * diff(w, y) + w * diff(w, z) + diff(p, z) - lap(w) / Re)

    return (u, v, w), (fx, fy, fz)

if __name__ == '__main__':
    u = sp.sin(x) * sp.cos(y) * sp.sin(z) * sp.sin(t)
    v = sp.cos(x) * sp.sin(y) * sp.sin(z) * sp.sin(t)
    w = 2 * sp.cos(x) * sp.cos(y) * sp.cos(z) * sp.sin(t)
    p = sp.sin(t) * x * y * z

    (u, v, w), (fx, fy, fz) = manufsol(u=u, v=v, w=w, p=p, ignore_pressure=True)

    # Generate the C code through sympy's codegen utility.
    signatures = [
        ('u_exact', u),
        ('v_exact', v),
        ('w_exact', w),
        ('p_exact', p),
        ('forcing_x', fx),
        ('forcing_y', fy),
        ('forcing_z', fz)
    ]

    codegen_result = codegen(
        signatures,
        language='C99',
        prefix='Manufactured',
        project='mif',
        header=True,
        empty=True,
        argument_sequence=(t, x, y, z),
        global_vars=[Re]
    )

    # Prepare the output strings for C and header files

    # Change the return type and argument types in the generated C code
    # Modify the generated code for compatibility with the specified format
    output_code = ""

    for function_name, code in codegen_result:
        # Change the function signature to use `const Real` and `noexcept`
        modified_code = str(code).replace('double', 'Real').replace('Real z)', 'Real z)noexcept')
        output_code += modified_code + '\n\n'

    # Print the modified C code
    print(output_code)
