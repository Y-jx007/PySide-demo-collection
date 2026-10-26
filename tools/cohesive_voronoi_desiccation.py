# -*- coding: utf-8 -*-
"""Voronoi-polygon cohesive network for desiccation cracks.

Seeds are jittered; Voronoi cells are clipped to the square.  Each polygon is
fan-triangulated from its centroid for the bulk, and every shared polygon edge
carries a zero-thickness cohesive element.  Cracks therefore follow polygon
edges, giving natural Y/T junctions instead of a fixed triangle mesh.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def _clip_poly(poly):
    out = [(float(x), float(y)) for x, y in poly]
    for axis, lim, keep in ((0, 0.0, 1), (0, 1.0, -1), (1, 0.0, 1), (1, 1.0, -1)):
        if not out:
            break
        new = []
        prev = out[-1]
        for cur in out:
            def inside(p):
                return (p[axis] - lim) * keep >= -1e-12
            if inside(cur):
                if not inside(prev):
                    t = (lim - prev[axis]) / (cur[axis] - prev[axis]) if abs(cur[axis] - prev[axis]) > 1e-15 else 0.0
                    new.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
                new.append(cur)
            elif inside(prev):
                t = (lim - prev[axis]) / (cur[axis] - prev[axis]) if abs(cur[axis] - prev[axis]) > 1e-15 else 0.0
                new.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            prev = cur
        out = new
    return out


def _cst(coords, E=1.0, nu=0.3):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    B = np.vstack([np.array([b1, 0, b2, 0, b3, 0.0]),
                   np.array([0, c1, 0, c2, 0, c3]),
                   np.array([c1, b1, c2, b2, c3, b3])]) / (2 * A)
    D = E / (1 - nu * nu) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], float)
    return A, B, D


def _Bcoh(nx, ny):
    B = np.zeros((2, 8))
    tt = np.array([-ny, nx])
    for q in (0, 1):
        B[0, 2 * q] = 0.5 * nx
        B[0, 2 * q + 1] = 0.5 * ny
        B[1, 2 * q] = 0.5 * tt[0]
        B[1, 2 * q + 1] = 0.5 * tt[1]
        B[0, 2 * (q + 2)] = -0.5 * nx
        B[0, 2 * (q + 2) + 1] = -0.5 * ny
        B[1, 2 * (q + 2)] = -0.5 * tt[0]
        B[1, 2 * (q + 2) + 1] = -0.5 * tt[1]
    return B


def _trac(w, kn=10.0, kt=5.0, tmax=0.0012, Gc=0.001):
    w0 = tmax / kn
    wf = 2.0 * Gc / tmax
    if w <= 0.0:
        return kn * w, 0.0
    if w <= w0:
        return kn * w, 0.0
    if w < wf:
        tn = tmax * (wf - w) / (wf - w0)
        return tn, float(np.clip(1.0 - tn / max(kn * w, 1e-30), 0.0, 1.0))
    return 0.0, 1.0


def build(m=10, seed=7):
    rng = np.random.default_rng(seed)
    pts = []
    for j in range(m):
        for i in range(m):
            pts.append((i / (m - 1) + (rng.random() - 0.5) * 0.5 / (m - 1),
                        j / (m - 1) + (rng.random() - 0.5) * 0.5 / (m - 1)))
    for i in range(m):
        pts += [(i / (m - 1), -0.25), (i / (m - 1), 1.25)]
    for j in range(m):
        pts += [(-0.25, j / (m - 1)), (1.25, j / (m - 1))]
    vor = Voronoi(np.array(pts))
    polys = []
    for k in range(m * m):
        reg = vor.regions[vor.point_region[k]]
        if not reg or -1 in reg:
            continue
        poly = [vor.vertices[v] for v in reg]
        area = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] -
                   poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
        if area < 0:
            poly = poly[::-1]
        poly = _clip_poly(poly)
        if len(poly) >= 3:
            polys.append(poly)

    nodes = []
    cells = []
    tris = []
    for poly in polys:
        P = np.array(poly)
        c = P.mean(axis=0)
        base = len(nodes)
        nodes.append(c)
        bnodes = []
        for p in P:
            bnodes.append(len(nodes))
            nodes.append(p)
        for k in range(len(P)):
            tris.append((base, bnodes[k], bnodes[(k + 1) % len(P)]))
        cells.append((base, bnodes, P))

    emap = {}
    coh = []
    for base, bnodes, P in cells:
        for k in range(len(P)):
            key = tuple(sorted([tuple(np.round(P[k], 10)),
                                tuple(np.round(P[(k + 1) % len(P)], 10))]))
            if key in emap:
                b0, b1 = emap[key]
                a0, a1 = bnodes[k], bnodes[(k + 1) % len(P)]
                p0 = np.array(nodes[a0])
                p1 = np.array(nodes[a1])
                ex = p1 - p0
                L = np.linalg.norm(ex)
                nx, ny = ex[1] / L, -ex[0] / L
                coh.append((a0, a1, b0, b1, nx, ny, L,
                            p0[0], p0[1], p1[0], p1[1]))
            else:
                emap[key] = (bnodes[k], bnodes[(k + 1) % len(P)])

    nd = 2 * len(nodes)
    rows, cols, vals = [], [], []
    f0 = np.zeros(nd)
    for tri in tris:
        A, B, D = _cst(np.array([nodes[t] for t in tri]))
        K = A * B.T @ D @ B
        fe = A * B.T @ D @ np.array([1.0, 1.0, 0.0])
        for a in range(3):
            for b in range(3):
                for p in range(2):
                    for q in range(2):
                        rows.append(2 * tri[a] + p)
                        cols.append(2 * tri[b] + q)
                        vals.append(K[2 * a + p, 2 * b + q])
            for p in range(2):
                f0[2 * tri[a] + p] += fe[2 * a + p]
    Kb = coo_matrix((vals, (rows, cols)), shape=(nd, nd)).tocsr()
    kf = 0.05
    Kf = coo_matrix((np.full(nd, kf), (np.arange(nd), np.arange(nd))),
                    shape=(nd, nd)).tocsr()
    rng2 = np.random.default_rng(seed + 9)
    tmax = 0.0012 * (1.0 + 0.25 * rng2.normal(size=len(coh)))
    weak = rng2.random(len(coh)) < 0.30
    tmax[weak] *= 0.15
    return nodes, tris, coh, Kb, f0, Kf, tmax


def solve_voronoi(eps=0.003, m=10, seed=7, n_iter=30):
    nodes, tris, coh, Kb, f0, Kf, tmax = build(m, seed)
    nd = 2 * len(nodes)
    d = np.zeros(len(coh))
    u = np.zeros(nd)
    K = Kb + Kf
    for it in range(n_iter):
        rows, cols, vals = [], [], []
        for q, (a0, a1, b0, b1, nx, ny, L, *_) in enumerate(coh):
            B = _Bcoh(nx, ny)
            ke = np.diag([max((1 - d[q]) * 10.0, 1e-6),
                          max((1 - d[q]) * 5.0, 1e-6)])
            Kc = L * (B.T @ ke @ B)
            ids = [a0, a1, b0, b1]
            for aa in range(4):
                for bb in range(4):
                    for p in range(2):
                        for qq in range(2):
                            rows.append(2 * ids[aa] + p)
                            cols.append(2 * ids[bb] + qq)
                            vals.append(Kc[2 * aa + p, 2 * bb + qq])
        K = Kb + Kf + coo_matrix((vals, (rows, cols)),
                                 shape=(nd, nd)).tocsr()
        u = spsolve(K, -f0 * eps)
        dnew = d.copy()
        for q, (a0, a1, b0, b1, nx, ny, L, *_) in enumerate(coh):
            B = _Bcoh(nx, ny)
            ul = np.array([u[2 * a0], u[2 * a0 + 1], u[2 * a1], u[2 * a1 + 1],
                           u[2 * b0], u[2 * b0 + 1], u[2 * b1], u[2 * b1 + 1]])
            # The edge normal is oriented from the second cell to the first;
            # drying opens with a negative sign for this ordering.
            w = -float((B @ ul)[0])
            tn, dd = _trac(w, tmax=tmax[q])
            dnew[q] = max(d[q], 0.5 * d[q] + 0.5 * dd)
        ch = float(np.max(np.abs(dnew - d)))
        d = dnew
        if ch < 1e-9:
            break
    return u, d, coh, nodes


def rasterize(d, coh, res=320, threshold=0.2):
    from PIL import Image, ImageDraw
    img = Image.new("L", (res, res), 0)
    draw = ImageDraw.Draw(img)
    for q, (a0, a1, b0, b1, nx, ny, L, x0, y0, x1, y1) in enumerate(coh):
        if d[q] <= threshold:
            continue
        draw.line([(x0 * res, y0 * res), (x1 * res, y1 * res)],
                  fill=255, width=max(1, int(round(0.8 + 1.8 * float(d[q])))))
    return np.asarray(img, dtype=np.uint8)


if __name__ == "__main__":
    for eps in (0.0015, 0.003, 0.0045):
        u, d, coh, nodes = solve_voronoi(eps)
        print(f"eps={eps:.4f} cells={len(nodes)} coh={len(coh)} "
              f"dmax={d.max():.3f} dmean={d.mean():.3f}")
