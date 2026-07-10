import math
import re
import numpy as np
import taichi_forge as ti

def vectorize_func(func):
    """将mp函数向量化"""
    def vectorized_func(z_array, *args, **kwargs):
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
    """使用预编译的表达式进行 RK4 积分

    安全提示：eval 在限制命名空间 {x,y,z,p0-p5,math} 中执行，
    不包含 __builtins__，可以防止任意代码执行。
    """
    out = np.zeros((n, 3), dtype=np.float64)
    x, y, z = x0, y0, z0
    p = params
    _builtins = {"__builtins__": {}}
    ns = {'x': x, 'y': y, 'z': z,
          'p0': p[0], 'p1': p[1], 'p2': p[2],
          'p3': p[3], 'p4': p[4], 'p5': p[5],
          'math': math}
    for i in range(n):
        ns['x'], ns['y'], ns['z'] = x, y, z
        dx1 = eval(dx_code, _builtins, ns)
        dy1 = eval(dy_code, _builtins, ns)
        dz1 = eval(dz_code, _builtins, ns)

        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        z2 = z + 0.5 * dt * dz1
        ns['x'], ns['y'], ns['z'] = x2, y2, z2
        dx2 = eval(dx_code, _builtins, ns)
        dy2 = eval(dy_code, _builtins, ns)
        dz2 = eval(dz_code, _builtins, ns)

        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        z3 = z + 0.5 * dt * dz2
        ns['x'], ns['y'], ns['z'] = x3, y3, z3
        dx3 = eval(dx_code, _builtins, ns)
        dy3 = eval(dy_code, _builtins, ns)
        dz3 = eval(dz_code, _builtins, ns)

        x4 = x + dt * dx3
        y4 = y + dt * dy3
        z4 = z + dt * dz3
        ns['x'], ns['y'], ns['z'] = x4, y4, z4
        dx4 = eval(dx_code, _builtins, ns)
        dy4 = eval(dy_code, _builtins, ns)
        dz4 = eval(dz_code, _builtins, ns)

        x += dt / 6.0 * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt / 6.0 * (dy1 + 2*dy2 + 2*dy3 + dy4)
        z += dt / 6.0 * (dz1 + 2*dz2 + 2*dz3 + dz4)
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out, x, y, z

# 双精度复数向量类型
c64 = ti.types.vector(2, ti.f64)

# ---------- 复数运算（统一使用安全除零版本） ----------
@ti.func
def csqr(z: c64) -> c64:
    return c64(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)

@ti.func
def cconj(z: c64) -> c64:
    return c64(z.x, -z.y)

@ti.func
def cmul(a: c64, b: c64) -> c64:
    return c64(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def cdiv(a: c64, b: c64) -> c64:
    # 安全除法，避免除零导致 NaN
    denom = b.x * b.x + b.y * b.y
    safe = denom > 1e-12
    inv_denom = 1.0 / ti.max(denom, 1e-12)
    real = (a.x * b.x + a.y * b.y) * inv_denom
    imag = (a.y * b.x - a.x * b.y) * inv_denom
    return c64(ti.select(safe, real, 0.0), ti.select(safe, imag, 0.0))

@ti.func
def csin(z: c64) -> c64:
    ey = ti.exp(z.y)
    e_neg_y = ti.exp(-z.y)
    cosh_y = (ey + e_neg_y) * 0.5
    sinh_y = (ey - e_neg_y) * 0.5
    return c64(ti.sin(z.x) * cosh_y, ti.cos(z.x) * sinh_y)

@ti.func
def ccos(z: c64) -> c64:
    ey = ti.exp(z.y)
    e_neg_y = ti.exp(-z.y)
    cosh_y = (ey + e_neg_y) * 0.5
    sinh_y = (ey - e_neg_y) * 0.5
    return c64(ti.cos(z.x) * cosh_y, -ti.sin(z.x) * sinh_y)

@ti.func
def cexp(z: c64) -> c64:
    r = ti.exp(z.x)
    return c64(r * ti.cos(z.y), r * ti.sin(z.y))

@ti.func
def clog(z: c64) -> c64:
    return c64(ti.log(ti.sqrt(z.x * z.x + z.y * z.y)),
               ti.atan2(z.y, z.x))

@ti.func
def cpow(z: c64, n: ti.f64) -> c64:
    r = ti.sqrt(z.x * z.x + z.y * z.y)
    theta = ti.atan2(z.y, z.x)
    new_r = ti.pow(r, n)
    new_theta = theta * n
    return c64(new_r * ti.cos(new_theta), new_r * ti.sin(new_theta))

# ---------- 颜色打包 ----------
@ti.func
def pack_color(r: ti.f32, g: ti.f32, b: ti.f32) -> ti.u32:
    ri = ti.cast(ti.min(255, ti.max(0, r * 255.0)), ti.u32)
    gi = ti.cast(ti.min(255, ti.max(0, g * 255.0)), ti.u32)
    bi = ti.cast(ti.min(255, ti.max(0, b * 255.0)), ti.u32)
    return (ti.u32(0xff) << 24) | (ri << 16) | (gi << 8) | bi