import sympy as sp
from sympy import diff
from sympy.utilities.codegen import codegen


t, x, y, z, Re = sp.symbols('t x y z Reynolds')


def laplacian(u):
    return diff(u,x,2) + diff(u,y,2) + diff(u,z,2)


def manufsol(u, v, w, p, ignore_pressure=False):
    if diff(u,x) + diff(v,y) + diff(w,z) != 0:
        print("[WARNING]: the proposed manufactured solution is not divergence free, correcting")
        w = -z * (diff(u,x) + diff(v,y))

    if ignore_pressure:
        p = 0

    fx = sp.simplify(diff(u,t) + u*diff(u,x) + v*diff(u,y) + w*diff(u,z) + diff(p,x) - laplacian(u)/Re + diff(p,x))
    fy = sp.simplify(diff(v,t) + u*diff(v,x) + v*diff(v,y) + w*diff(v,z) + diff(p,y) - laplacian(v)/Re + diff(p,y))
    fz = sp.simplify(diff(w,t) + u*diff(w,x) + v*diff(w,y) + w*diff(w,z) + diff(p,z) - laplacian(w)/Re + diff(p,z))

    return (u,v,w), (fx,fy,fz)


if __name__ == '__main__':
    u = x**3
    v = x**3
    w = x**3
    p = x*y

    # for current testing purpuoses we ignore the pressure term
    (u,v,w), (fx,fy,fz) = manufsol(u,v,w,p,ignore_pressure=True)

    [(c_name, c_code), (h_name, c_header)] = codegen(
        [
            ('u_exact'  ,  u),
            ('v_exact'  ,  v),
            ('w_exact'  ,  w),
            ('p_exact'  ,  p),
            ('forcing_x', fx),
            ('forcing_y', fy),
            ('forcing_z', fz),
        ],
        language='C99',
        prefix='Manifactured',
        project='mif',
        header=True,
        empty=True,
        argument_sequence=(t,x,y,z),
        global_vars=[Re]
    )

    print(c_code)
