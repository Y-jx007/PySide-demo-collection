def vectorize_func(func):
            """将mp函数向量化"""
            def vectorized_func(z_array,*args, **kwargs):
                if np.isscalar(z_array):
                    return complex(func(z_array,*args, **kwargs))
                
                result = np.empty(z_array.shape, dtype=complex)
                flat_z = z_array.flat
                flat_result = result.flat
                
                for i in range(len(flat_z)):
                    flat_result[i] = complex(func(flat_z[i],*args, **kwargs))
                
                return result
            return vectorized_func

def make_safe_expression(expr: str) -> str:
    """将 sin, cos 等替换为 math.sin, math.cos，便于 eval 使用"""
    funcs = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tanh']
    for f in funcs:
        expr = re.sub(rf'\b{f}\b', f'math.{f}', expr)
    return expr

def integrate_custom_python(x0, y0, z0, params, dt, n, dx_code, dy_code, dz_code):
    """使用预编译的表达式进行 RK4 积分"""
    out = np.zeros((n, 3), dtype=np.float64)
    x, y, z = x0, y0, z0
    p = params
    ns = {'x': x, 'y': y, 'z': z,
          'p0': p[0], 'p1': p[1], 'p2': p[2],
          'p3': p[3], 'p4': p[4], 'p5': p[5],
          'math': math}
    for i in range(n):
        ns['x'], ns['y'], ns['z'] = x, y, z
        dx1 = eval(dx_code, ns)
        dy1 = eval(dy_code, ns)
        dz1 = eval(dz_code, ns)

        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        z2 = z + 0.5 * dt * dz1
        ns['x'], ns['y'], ns['z'] = x2, y2, z2
        dx2 = eval(dx_code, ns)
        dy2 = eval(dy_code, ns)
        dz2 = eval(dz_code, ns)

        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        z3 = z + 0.5 * dt * dz2
        ns['x'], ns['y'], ns['z'] = x3, y3, z3
        dx3 = eval(dx_code, ns)
        dy3 = eval(dy_code, ns)
        dz3 = eval(dz_code, ns)

        x4 = x + dt * dx3
        y4 = y + dt * dy3
        z4 = z + dt * dz3
        ns['x'], ns['y'], ns['z'] = x4, y4, z4
        dx4 = eval(dx_code, ns)
        dy4 = eval(dy_code, ns)
        dz4 = eval(dz_code, ns)

        x += dt / 6.0 * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt / 6.0 * (dy1 + 2*dy2 + 2*dy3 + dy4)
        z += dt / 6.0 * (dz1 + 2*dz2 + 2*dz3 + dz4)
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out, x, y, z