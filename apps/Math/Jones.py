# -*- coding: utf-8 -*-
"""Jones knot invariant explorer.

A knot is entered by name (Rolfsen label, English/Chinese alias or T(p, q)).
From that name the app reads a braid word and then *generates* everything:

* the closed braid is swept into a hemp tube (no per-knot curve is stored);
* the Jones polynomial comes from a Temperley-Lieb state sum over the braid;
* determinant = |V(-1)|, Alexander from the catalogue / torus formula.

The Kauffman bracket helpers below (PD code based) are kept as an independent
cross-check used by the tests.
"""
from collections import defaultdict
from fractions import Fraction
import math
import numpy as np

from common.free_camera import FreeCamera
from common.lab_window import LabWindow
from common.shader_canvas import ShaderMeshCanvas
from common.qt_base import (
    QComboBox, QColor, QFont, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPainter, QPen, QPointF, QPushButton, QRectF, QSizePolicy, QVBoxLayout,
    QWidget, LabeledSlider, Signal, Qt, run_app,
)
from support.math.knots import (
    KnotError, braid_components, braid_curves, braid_permutation,
    catalogue_names, conjugate, destabilize, determinant_from_jones,
    jones_from_braid, mirror_jones, r2_insert, r2_remove, r3_apply, resolve,
    stabilize, writhe_of,
)

# Knot Atlas PD codes.  Crossing order is X[a,b,c,d].
TREFOIL_PD = ((1, 4, 2, 5), (3, 6, 4, 1), (5, 2, 6, 3))
FIGURE_EIGHT_PD = ((4, 2, 5, 1), (8, 6, 1, 5), (6, 3, 7, 4), (2, 7, 3, 8))


def kauffman_bracket(pd):
    """Return the Kauffman bracket as a Laurent polynomial in A.

    Key is an integer A exponent, value a Fraction coefficient.
    """
    n = len(pd)
    total = defaultdict(Fraction)
    for mask in range(1 << n):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        a_count = b_count = 0
        for k, (a, b, c, d) in enumerate(pd):
            if (mask >> k) & 1:
                b_count += 1
                pairings = ((a, d), (b, c))
            else:
                a_count += 1
                pairings = ((a, b), (c, d))
            for x, y in pairings:
                union(x, y)

        loops = len({find(x) for x in parent})
        power = a_count - b_count
        poly = {0: Fraction(1)}
        for _ in range(max(0, loops - 1)):
            nxt = defaultdict(Fraction)
            for exponent, coef in poly.items():
                nxt[exponent + 2] -= coef
                nxt[exponent - 2] -= coef
            poly = nxt
        for exponent, coef in poly.items():
            total[exponent + power] += coef
    return {e: c for e, c in total.items() if c}


def jones_polynomial(pd, writhe):
    """Return V(t) as {Fraction exponent: Fraction coefficient}."""
    bracket = kauffman_bracket(pd)
    sign = Fraction(-1) if writhe % 2 else Fraction(1)
    out = defaultdict(Fraction)
    for exponent, coef in bracket.items():
        a_exp = exponent - 3 * writhe
        # t = A^-4  =>  A^a = t^(-a/4)
        out[Fraction(-a_exp, 4)] += coef * sign
    return {e: c for e, c in out.items() if c}


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _pow_t(exp):
    exp = int(exp)
    if exp == 0:
        return ""
    if exp == 1:
        return "t"
    return "t" + str(exp).translate(_SUP)


def format_polynomial(poly):
    if not poly:
        return "0"
    terms = sorted(poly.items(), key=lambda kv: kv[0], reverse=True)
    pieces = []
    for exponent, coef in terms:
        sign = "-" if coef < 0 else "+"
        value = abs(coef)
        if value.denominator == 1:
            number = "" if value.numerator == 1 and exponent != 0 else str(value.numerator)
        else:
            number = f"{value.numerator}/{value.denominator}"
        body = number + _pow_t(exponent)
        if not body:
            body = "1"
        pieces.append((sign, body))
    first_sign, first_body = pieces[0]
    text = ("-" if first_sign == "-" else "") + first_body
    for sign, body in pieces[1:]:
        text += f" {sign} {body}"
    return text


def _tube_one(points, radius=0.052, sides=40, scale=1.0, turns=None, strands=7):
    """One closed curve -> several thin twisted hemp strands around it."""
    pts = np.asarray(points, dtype=np.float64) * scale
    n = len(pts)
    tangents = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tangents, up)
    bad = np.linalg.norm(normals, axis=1) < 1e-6
    normals[bad] = np.cross(tangents[bad], np.array([0.0, 1.0, 0.0]))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    binormals = np.cross(tangents, normals)
    binormals /= np.maximum(np.linalg.norm(binormals, axis=1, keepdims=True), 1e-9)

    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring_cos = np.cos(angles)[None, :]
    ring_sin = np.sin(angles)[None, :]
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    u = np.repeat(u, sides, axis=1)
    v = np.repeat(angles[None, :], n, axis=0) / (2.0 * math.pi)

    if turns is None:
        # tight twist: about 3.6 turns per unit arc length
        length = float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))
        turns = max(10, int(round(length * 3.6)))
    # never wrap faster than the polyline can resolve, or the filaments turn
    # into triangular spikes instead of a smooth helix
    turns = max(6, min(turns, n // 14))

    # Solid rope: filaments overlap slightly so the grooves stay shallow and
    # the silhouette stays round even when the whole rope is only a few pixels.
    overlap = 1.15
    pitch_r = radius / (1.0 + overlap * math.sin(math.pi / strands))
    strand_r = pitch_r * overlap * math.sin(math.pi / strands)
    core_r = max(pitch_r - strand_r, radius * 0.10)

    all_verts, all_normals, all_uvs, all_idx, all_strand = [], [], [], [], []
    offset = 0

    def emit(centers, offs, strand_id, u_coord, v_coord):
        nonlocal offset
        verts = centers[:, None, :] + offs
        all_verts.append(verts.reshape(-1, 3))
        all_normals.append((offs / np.maximum(
            np.linalg.norm(offs, axis=-1, keepdims=True), 1e-9)).reshape(-1, 3))
        uv = np.stack((u_coord, v_coord), axis=-1)
        all_uvs.append(uv.reshape(-1, 2))
        all_strand.append(np.full(n * sides, strand_id, dtype=np.int32))
        ring = np.arange(sides)
        ring_next = (ring + 1) % sides
        rows = np.arange(n)
        rows_next = (rows + 1) % n
        base_a = offset + rows[:, None] * sides
        base_b = offset + rows_next[:, None] * sides
        a = base_a + ring[None, :]
        b = base_a + ring_next[None, :]
        c = base_b + ring_next[None, :]
        d = base_b + ring[None, :]
        tri1 = np.stack((a, b, c), axis=-1).reshape(-1, 3)
        tri2 = np.stack((a, c, d), axis=-1).reshape(-1, 3)
        all_idx.append(np.concatenate((tri1, tri2)).astype(np.uint32))
        offset += n * sides

    for k in range(strands):
        theta = 2.0 * math.pi * (turns * u[:, 0] + k / strands)
        e1 = (normals * np.cos(theta)[:, None]
              + binormals * np.sin(theta)[:, None])
        e2 = (-normals * np.sin(theta)[:, None]
              + binormals * np.cos(theta)[:, None])
        centers = pts + pitch_r * e1
        offs = (strand_r * ring_cos[..., None] * e1[:, None, :]
                + strand_r * ring_sin[..., None] * e2[:, None, :])
        emit(centers, offs, k, u, v + k / strands)

    # central core filament fills the hole of the outer ring
    core_offs = (core_r * ring_cos[..., None] * normals[:, None, :]
                 + core_r * ring_sin[..., None] * binormals[:, None, :])
    emit(pts, core_offs, 0, u, v)

    return (np.concatenate(all_verts).astype(np.float32),
            np.concatenate(all_normals).astype(np.float32),
            np.concatenate(all_uvs).astype(np.float32),
            np.concatenate(all_idx),
            np.concatenate(all_strand))


def tube_mesh(points, radius=0.052, sides=40):
    return _tube_one(points, radius=radius, sides=sides, scale=0.62)


def tube_mesh_multi(curves, radius=0.052, sides=40):
    """Tube for every link component, merged into one mesh."""
    parts = [_tube_one(c, radius=radius, sides=sides, scale=1.0) for c in curves]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("no curve to mesh")
    verts, normals, uvs, idx, strand = [], [], [], [], []
    base = 0
    for v, nm, uv, ix, st in parts:
        verts.append(v)
        normals.append(nm)
        uvs.append(uv)
        idx.append(ix + np.uint32(base))
        strand.append(st)
        base += len(v)
    return (np.concatenate(verts), np.concatenate(normals),
            np.concatenate(uvs), np.concatenate(idx), np.concatenate(strand))


class BraidDiagram(QWidget):
    """Schematic braid diagram: columns are generators, rows are strands."""

    crossingClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setMaximumHeight(112)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.word = []
        self.strands = 2
        self.selected = -1
        self.smoothing = {}

    def set_state(self, word, strands, selected=-1, smoothing=None):
        self.word = list(word)
        self.strands = int(strands)
        self.selected = int(selected)
        self.smoothing = dict(smoothing or {})
        self.update()

    def _layout(self):
        pad = 14.0
        usable = max(1.0, self.width() - 2.0 * pad)
        n = len(self.word)
        cw = min(44.0, usable / max(1, n))
        rowh = max(6.0, (self.height() - 2.0 * pad) / max(1, self.strands))
        return pad, cw, rowh, n

    def column_at(self, x):
        pad, cw, _, n = self._layout()
        if n <= 0:
            return -1
        k = int((x - pad) // cw)
        return k if 0 <= k < n else -1

    def mousePressEvent(self, event):
        k = self.column_at(event.position().x())
        if k >= 0:
            self.crossingClicked.emit(k)
        event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(253, 252, 250))
        pad, cw, rowh, n = self._layout()
        if n == 0:
            p.setPen(QColor(130, 130, 130))
            p.drawText(self.rect(), Qt.AlignCenter, "平凡结 · 无交叉")
            p.end()
            return
        if 0 <= self.selected < n:
            p.fillRect(QRectF(pad + self.selected * cw, 4, cw, self.height() - 8),
                       QColor(230, 242, 255))
        base = QColor(120, 100, 78)
        over_pen = QPen(base, 2.4, Qt.SolidLine, Qt.RoundCap)
        thin_pen = QPen(base, 1.6, Qt.SolidLine, Qt.RoundCap)
        smooth_pen = QPen(QColor(46, 140, 160), 2.0, Qt.DashLine, Qt.RoundCap)
        for k, g in enumerate(self.word):
            i = abs(int(g)) - 1
            x0 = pad + k * cw
            x1 = x0 + cw
            ya = pad + (i + 0.5) * rowh
            yb = pad + (i + 1.5) * rowh
            # strands that do not take part in this crossing stay straight
            p.setPen(thin_pen)
            for r in range(self.strands):
                if r in (i, i + 1):
                    continue
                y = pad + (r + 0.5) * rowh
                p.drawLine(QPointF(x0, y), QPointF(x1, y))
            if k in self.smoothing:
                mode = self.smoothing[k]
                p.setPen(smooth_pen)
                if mode == "A":
                    p.drawLine(QPointF(x0, ya), QPointF(x1, ya))
                    p.drawLine(QPointF(x0, yb), QPointF(x1, yb))
                else:
                    p.drawLine(QPointF(x0, ya), QPointF(x1, yb))
                    p.drawLine(QPointF(x0, yb), QPointF(x1, ya))
                continue
            down_over = int(g) > 0
            # over strand: down (i -> i+1) when g > 0, up otherwise
            p.setPen(over_pen)
            if down_over:
                p.drawLine(QPointF(x0, ya), QPointF(x1, yb))
            else:
                p.drawLine(QPointF(x0, yb), QPointF(x1, ya))
            # under strand, split around the crossing centre
            p.setPen(thin_pen)
            if down_over:
                p.drawLine(QPointF(x0, yb), QPointF(x0 + cw * 0.38, ya + rowh * 0.38))
                p.drawLine(QPointF(x0 + cw * 0.62, ya + rowh * 0.62), QPointF(x1, ya))
            else:
                p.drawLine(QPointF(x0, ya), QPointF(x0 + cw * 0.38, yb - rowh * 0.38))
                p.drawLine(QPointF(x0 + cw * 0.62, yb - rowh * 0.62), QPointF(x1, yb))
        p.setPen(QPen(QColor(210, 210, 205), 1.0))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()


class JonesCanvas(ShaderMeshCanvas):
    VERTEX_SRC = """
    #version 330 core
    layout(location = 0) in vec3 a_pos;
    layout(location = 1) in vec3 a_normal;
    layout(location = 2) in vec2 a_uv;
    layout(location = 3) in float a_strand;
    uniform mat4 u_mvp;
    out vec3 v_world;
    out vec3 v_normal;
    out vec2 v_uv;
    out float v_strand;
    void main() {
        v_world = a_pos;
        v_normal = a_normal;
        v_uv = a_uv;
        v_strand = a_strand;
        gl_Position = u_mvp * vec4(a_pos, 1.0);
    }
    """

    FRAGMENT_SRC = """
    #version 330 core
    in vec3 v_world;
    in vec3 v_normal;
    in vec2 v_uv;
    in float v_strand;
    uniform vec3 u_eye;
    out vec4 frag_color;

    float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
    }

    float hash31(vec3 p) {
        p = fract(p * 0.3183099 + vec3(0.11, 0.27, 0.39));
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }

    float vnoise(vec3 x) {
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(hash31(i + vec3(0, 0, 0)), hash31(i + vec3(1, 0, 0)), f.x),
                       mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y),
                   mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x),
                       mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y), f.z);
    }

    float D_GGX(float NoH, float a) {
        float a2 = a * a;
        float d = NoH * NoH * (a2 - 1.0) + 1.0;
        return a2 / (3.14159265 * d * d + 1e-7);
    }

    float V_Smith(float NoV, float NoL, float a) {
        float a2 = a * a;
        float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
        float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
        return 0.5 / (gv + gl + 1e-7);
    }

    vec3 pbr_light(vec3 N, vec3 V, vec3 L, vec3 radiance,
                   vec3 albedo, float rough, vec3 F0, float spec_mask) {
        vec3 H = normalize(L + V);
        float NoL = max(dot(N, L), 0.0);
        float NoV = max(dot(N, V), 1e-4);
        float NoH = max(dot(N, H), 0.0);
        float VoH = max(dot(V, H), 0.0);
        float a = max(0.05, rough * rough);
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
        vec3 spec = F * D_GGX(NoH, a) * V_Smith(NoV, NoL, a);
        vec3 diffuse = albedo * (1.0 - F) / 3.14159265;
        return (diffuse + spec * spec_mask) * radiance * NoL;
    }

    void main() {
        vec3 N = normalize(v_normal);
        vec3 V = normalize(u_eye - v_world);

        vec3 dpdx = dFdx(v_world);
        vec3 dpdy = dFdy(v_world);
        vec2 duvdx = dFdx(v_uv);
        vec2 duvdy = dFdy(v_uv);
        float det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
        vec3 T = normalize((dpdx * duvdy.y - dpdy * duvdx.y) / max(det, 1e-5));
        vec3 B = normalize(cross(N, T));
        // Only non-periodic, world-space detail.  Deliberately NO stripe
        // pattern: a periodic fibre texture always moires at a distance and
        // reads as zebra stripes up close.  The braided structure comes from
        // the filament geometry instead.
        float foot_w = max(max(fwidth(v_world.x), fwidth(v_world.y)),
                           fwidth(v_world.z)) + 1e-7;
        float n1 = vnoise(v_world * 38.0) - 0.5;
        float n2 = vnoise(v_world * 150.0) - 0.5;
        float n3 = vnoise(v_world * 380.0) - 0.5;
        float b1 = clamp(1.45 - foot_w * 38.0, 0.0, 1.0);
        float b2 = clamp(1.45 - foot_w * 150.0, 0.0, 1.0);
        float b3 = clamp(1.45 - foot_w * 380.0, 0.0, 1.0);
        float grain = n1 * b1 + 0.6 * n2 * b2 + 0.35 * n3 * b3;

        N = normalize(N + 0.55 * grain * B + 0.30 * n3 * b3 * T);

        int sid = int(clamp(v_strand + 0.5, 0.0, 6.0));
        vec3 albedo = vec3(0.44, 0.29, 0.16);
        if (sid == 1) albedo = vec3(0.37, 0.23, 0.12);
        if (sid == 2) albedo = vec3(0.50, 0.34, 0.19);
        if (sid == 3) albedo = vec3(0.41, 0.26, 0.14);
        if (sid == 4) albedo = vec3(0.47, 0.31, 0.17);
        if (sid == 5) albedo = vec3(0.39, 0.25, 0.13);
        if (sid == 6) albedo = vec3(0.45, 0.30, 0.16);
        // gentle tonal variation only (no dark grooves)
        float patch = vnoise(v_world * 10.0) - 0.5;
        albedo *= (0.94 + 0.10 * grain) * (1.0 + 0.16 * patch);

        float rough = 0.90 + 0.06 * (0.5 + n1) + 0.04 * grain;
        vec3 F0 = vec3(0.024);
        vec3 color = vec3(0.0);

        // slight sheen variation, no stripe occlusion
        float spec_mask = 0.92 + 0.16 * grain;

        vec3 L1 = normalize(vec3(-0.65, 0.70, 0.55));
        vec3 L2 = normalize(vec3(0.62, 0.52, 0.58));
        vec3 L3 = normalize(vec3(0.10, -0.65, 0.35));
        color += pbr_light(N, V, L1, vec3(2.3, 1.75, 1.20),
                           albedo, rough, F0, spec_mask);
        color += pbr_light(N, V, L2, vec3(1.15, 1.25, 1.35),
                           albedo, rough, F0, spec_mask);
        color += pbr_light(N, V, L3, vec3(0.30, 0.34, 0.38),
                           albedo, rough, F0, spec_mask);

        // Camera-facing fill opens the dark backside without flattening the form.
        vec3 Lcam = normalize(-V + vec3(0.0, 0.20, 0.0));
        color += pbr_light(N, V, Lcam, vec3(0.42, 0.38, 0.32),
                           albedo, rough, F0, spec_mask);

        float hemi = 0.5 + 0.5 * N.y;
        color += albedo * mix(0.10, 0.24, hemi);
        float rim = pow(1.0 - max(dot(N, V), 0.0), 2.4);
        color += vec3(1.0, 0.80, 0.58) * rim * 0.055;

        color = (color * (2.51 * color + 0.03)) /
                (color * (2.43 * color + 0.59) + 0.14);
        color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
        frag_color = vec4(color, 1.0);
    }
    """

    def __init__(self, parent=None):
        spec = resolve("3_1")
        curves = braid_curves(spec["braid"], spec["strands"], key=spec["key"])
        mesh = tube_mesh_multi(curves)
        super().__init__(
            self.VERTEX_SRC,
            self.FRAGMENT_SRC,
            mesh,
            camera=FreeCamera(target=(0.0, 0.0, 0.0), distance=4.4,
                              yaw=35.0, pitch=24.0),
            overlay="PBR hemp · 程序化纤维 · 拖动自由旋转 · 滚轮缩放 · 双击复位",
            parent=parent,
        )
        self.setFixedSize(600, 600)
        self.curves = curves

    def set_curves(self, curves):
        self.curves = curves
        self.set_mesh(tube_mesh_multi(curves))

    def set_curve(self, points):
        """Backwards-compatible single-component setter."""
        self.set_curves([points])

    def reset_view(self):
        self.camera.reset()
        self.update()


class JonesApp(LabWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.setWindowTitle("Jones · 扭结不变量")
        self.resize(1000, 800)
        self.setMinimumSize(1000, 800)
        self.preset_key = "3_1"
        self.spec = resolve("3_1")
        self.mirror = False
        self.polynomial = {}
        self.word = list(self.spec["braid"])
        self.strands = int(self.spec["strands"])
        self.base_jones = {}
        self.history = []
        self.smoothing = {}
        self.selected = -1
        self.canvas = JonesCanvas()
        self._build_ui()
        self.apply_spec(self.spec)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        presets = QGroupBox("结名")
        pform = QFormLayout(presets)
        pform.setContentsMargins(8, 6, 8, 6)
        pform.setSpacing(5)
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.setInsertPolicy(QComboBox.NoInsert)
        self.preset_combo.addItems(catalogue_names())
        self.preset_combo.setCurrentText(self.spec["pretty"])
        self.preset_combo.lineEdit().setPlaceholderText("3_1 / 5₂ / Trefoil / 三叶结 / T(2,5)")
        self.preset_combo.lineEdit().returnPressed.connect(self.apply_typed)
        self.preset_combo.activated.connect(self._apply_index)
        pform.addRow("结", self.preset_combo)
        self.mirror_check = QPushButton("镜像（切换手性）")
        self.mirror_check.clicked.connect(self.toggle_mirror)
        pform.addRow("", self.mirror_check)
        reset_view = QPushButton("重置视角")
        reset_view.clicked.connect(self.canvas.reset_view)
        pform.addRow("", reset_view)
        self.zoom_slider = LabeledSlider(
            "", 50, 220, 100, formatter=lambda v: f"{v}%", value_width=46
        )
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        pform.addRow("视图缩放", self.zoom_slider)
        col.addWidget(presets)

        invariant = QGroupBox("不变量")
        vbox = QVBoxLayout(invariant)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(4)
        self.lbl_crossings = QLabel("交叉数  –")
        self.lbl_writhe = QLabel("writhe  –")
        self.lbl_components = QLabel("分支数  –")
        self.lbl_det = QLabel("行列式  –")
        for widget in (self.lbl_crossings, self.lbl_writhe,
                       self.lbl_components, self.lbl_det):
            widget.setStyleSheet("font-family: Consolas, monospace;")
            vbox.addWidget(widget)
        vbox.addSpacing(4)
        vbox.addWidget(QLabel("Jones"))
        self.lbl_poly = QLabel("V(t) = –")
        self.lbl_poly.setWordWrap(True)
        self.lbl_poly.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        vbox.addWidget(self.lbl_poly)
        vbox.addWidget(QLabel("Alexander"))
        self.lbl_alexander = QLabel("Δ(t) = –")
        self.lbl_alexander.setWordWrap(True)
        self.lbl_alexander.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        vbox.addWidget(self.lbl_alexander)
        col.addWidget(invariant)

        formula = QGroupBox("公式")
        fbox = QVBoxLayout(formula)
        fbox.setContentsMargins(8, 6, 8, 6)
        fbox.setSpacing(4)
        ftext = QLabel(
            "<b>Kauffman bracket</b><br>"
            "⟨D⟩ = Σ<sub>s</sub> A<sup>a−b</sup>"
            "(−A²−A⁻²)<sup>|s|−1</sup><br>"
            "<b>Jones</b><br>"
            "V<sub>K</sub>(t) = (−A³)<sup>−w(D)</sup>⟨D⟩, "
            "t = A⁻⁴<br>"
            "<b>Alexander</b><br>"
            "Δ<sub>K</sub>(t) ≐ Δ<sub>K</sub>(t⁻¹), Δ<sub>K</sub>(1)=1<br>"
            "<b>行列式</b><br>"
            "det(K) = |Δ<sub>K</sub>(−1)|"
        )
        ftext.setWordWrap(True)
        ftext.setStyleSheet("font-size: 11px; color:#344054;")
        fbox.addWidget(ftext)
        col.addWidget(formula)

        algorithm = QGroupBox("算法")
        abox = QVBoxLayout(algorithm)
        abox.setContentsMargins(8, 6, 8, 6)
        abox.setSpacing(2)
        steps = QLabel(
            "1. 结名 → 辫子词 BR[n, σ…]<br>"
            "2. 辫子词展开成 Temperley–Lieb 状态<br>"
            "3. 闭合辫数环，取 δ = −A²−A⁻²<br>"
            "4. 用 writhe 归一化得 V(t)<br>"
            "5. 同时把闭合辫扫成三维 hemp 绳"
        )
        steps.setWordWrap(True)
        steps.setStyleSheet("font-size: 11px; color:#344054;")
        abox.addWidget(steps)
        smooth = QLabel(
            "A/B 平滑：交叉点换成两条不相交弧线，A 与 B 相差 90°；"
            "每个交叉选一种，2ⁿ 个状态。"
        )
        smooth.setWordWrap(True)
        smooth.setStyleSheet("font-size: 11px; color:#344054;")
        abox.addWidget(smooth)
        abox.addWidget(QLabel("R1/R2/R3 不改变 V(t)。"))
        col.addWidget(algorithm)

        col.addStretch(1)

        root.addWidget(panel)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        rv.addWidget(self.canvas)

        braid_box = QGroupBox("辫子图")
        bv = QVBoxLayout(braid_box)
        bv.setContentsMargins(6, 5, 6, 5)
        bv.setSpacing(4)
        self.diagram = BraidDiagram()
        self.diagram.crossingClicked.connect(self.select_crossing)
        bv.addWidget(self.diagram)
        row = QHBoxLayout()
        row.setSpacing(4)
        actions = (
            ("R2+", self.do_r2_insert),
            ("R2−", self.do_r2_remove),
            ("R3", self.do_r3),
            ("共轭", self.do_conjugate),
            ("稳定+", self.do_stabilize),
            ("稳定−", self.do_destabilize),
            ("撤销", self.undo),
            ("A", lambda: self.set_smoothing("A")),
            ("B", lambda: self.set_smoothing("B")),
        )
        for text, slot in actions:
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        bv.addLayout(row)
        rv.addWidget(braid_box)
        root.addWidget(right)

        self.lbl_status = QLabel("就绪")
        self.statusBar().addWidget(self.lbl_status, 1)

    # ---------------------------------------------------------------- naming
    def apply_typed(self):
        text = self.preset_combo.currentText().strip()
        try:
            spec = resolve(text)
        except KnotError as exc:
            self.lbl_status.setText(f"✗ {exc}")
            self.preset_combo.setCurrentText(self.spec["pretty"])
            return
        self.apply_spec(spec)

    def _apply_index(self, index):
        if index < 0:
            return
        self._apply_text(self.preset_combo.itemText(index))

    def _apply_text(self, text):
        try:
            spec = resolve(text)
        except KnotError as exc:
            self.lbl_status.setText(f"✗ {exc}")
            return
        self.apply_spec(spec)

    def apply_spec(self, spec):
        self.spec = spec
        self.preset_key = spec["key"]
        self.mirror = False
        self.mirror_check.setText("镜像（切换手性）")
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(spec["pretty"])
        self.preset_combo.blockSignals(False)
        self.word = list(spec["braid"])
        self.strands = int(spec["strands"])
        self.base_jones = jones_from_braid(self.word, self.strands)
        self.history = []
        self.smoothing = {}
        self.selected = -1
        self._rebuild()

    def set_zoom(self, value):
        self.canvas.zoom = int(value) / 100.0
        self.canvas.camera.zoom = self.canvas.zoom
        self.canvas.camera.update()
        self.canvas.update()

    def toggle_mirror(self):
        self.mirror = not self.mirror
        self.mirror_check.setText("恢复原手性" if self.mirror else "镜像（切换手性）")
        self._rebuild()

    # ------------------------------------------------------------- braid edit
    def select_crossing(self, index):
        self.selected = int(index)
        self._refresh_diagram()
        self.lbl_status.setText(
            f"选中第 {self.selected + 1} 个交叉 · σ{abs(self.word[self.selected])}"
            f"{'⁻¹' if self.word[self.selected] < 0 else ''}"
            f" · 用 A/B 平滑查看该交叉的两种平滑")

    def _refresh_diagram(self):
        self.diagram.set_state(self.word, self.strands, self.selected, self.smoothing)

    def _apply_word(self, word, strands, note):
        self.history.append((list(self.word), int(self.strands),
                             dict(self.smoothing), int(self.selected)))
        self.word = list(word)
        self.strands = int(strands)
        self.smoothing = {}
        self.selected = min(self.selected, len(self.word) - 1)
        self._rebuild()
        ok = jones_from_braid(self.word, self.strands) == self.base_jones
        self.lbl_status.setText(
            f"{note} · 辫长 {len(self.word)} · {self.strands} 股 · "
            + ("✓ 不变量未变" if ok else "✗ 不变量改变"))

    def _current_gen(self, fallback=1):
        if 0 <= self.selected < len(self.word):
            return abs(int(self.word[self.selected]))
        return fallback

    def do_r2_insert(self):
        gen = self._current_gen()
        pos = self.selected + 1 if self.selected >= 0 else len(self.word)
        self._apply_word(r2_insert(self.word, pos, gen, self.strands),
                         self.strands, f"R2 插入 σ{gen}σ{gen}⁻¹ @{pos + 1}")

    def do_r2_remove(self):
        start = max(0, self.selected)
        result = r2_remove(self.word, start, self.strands)
        if result is None:
            self.lbl_status.setText("没有可删除的 σσ⁻¹ 对（可先插入一对）")
            return
        self._apply_word(result[0], self.strands, f"R2 删除 @{result[1] + 1}")

    def do_r3(self):
        start = max(0, self.selected)
        result = r3_apply(self.word, start, self.strands)
        if result is None:
            self.lbl_status.setText("没有 σaσbσa 模式（可先插入 R2 对或选中别的交叉）")
            return
        self._apply_word(result[0], self.strands, f"R3 @{result[1] + 1}")

    def do_conjugate(self):
        gen = self._current_gen()
        self._apply_word(conjugate(self.word, gen, self.strands),
                         self.strands, f"共轭 σ{gen}·w·σ{gen}⁻¹")

    def do_stabilize(self):
        word, strands = stabilize(self.word, self.strands)
        self._apply_word(word, strands, "稳定化 +1 股")

    def do_destabilize(self):
        result = destabilize(self.word, self.strands)
        if result is None:
            self.lbl_status.setText("当前辫子无法直接去掉一股（需最后一位只出现一次）")
            return
        self._apply_word(result[0], result[1], "去稳定化 −1 股")

    def undo(self):
        if not self.history:
            self.lbl_status.setText("没有可撤销的操作")
            return
        word, strands, smoothing, selected = self.history.pop()
        self.word = list(word)
        self.strands = int(strands)
        self.smoothing = dict(smoothing)
        self.selected = int(selected)
        self._rebuild()
        self.lbl_status.setText(
            f"已撤销 · 辫长 {len(self.word)} · {self.strands} 股")

    def set_smoothing(self, mode):
        if not (0 <= self.selected < len(self.word)):
            self.lbl_status.setText("先点选辫子图里的一个交叉")
            return
        self.smoothing[self.selected] = mode
        self._refresh_diagram()
        self.lbl_status.setText(
            f"第 {self.selected + 1} 个交叉取 {mode} 平滑（仅示意：平滑后是链环，"
            "3D 仍显示原结）")

    def _rebuild(self):
        spec = self.spec
        base = jones_from_braid(self.word, self.strands)
        self.polynomial = mirror_jones(base) if self.mirror else base
        writhe = writhe_of(self.word)
        if self.mirror:
            writhe = -writhe
        components = braid_components(self.word, self.strands)
        alex = {Fraction(e): Fraction(c) for e, c in spec["alex"].items()}
        self.lbl_crossings.setText(f"交叉数  {spec['crossings']}")
        self.lbl_writhe.setText(f"writhe  {writhe:+d}  (辫长 {len(self.word)})")
        self.lbl_components.setText(f"分支数  {components}  ·  {self.strands} 股")
        self.lbl_det.setText(f"行列式  {determinant_from_jones(base)}")
        self.lbl_poly.setText("V(t) = " + format_polynomial(self.polynomial))
        self.lbl_alexander.setText("Δ(t) = " + format_polynomial(alex))
        pristine = list(self.word) == list(spec["braid"])
        curves = braid_curves(self.word, self.strands,
                              key=spec["key"] if pristine else None)
        if self.mirror:
            curves = [[(-x, y, z) for x, y, z in c] for c in curves]
        self.canvas.set_curves(curves)
        self._refresh_diagram()
        same = base == self.base_jones
        self.lbl_status.setText(
            f"Jones · {spec['pretty']} · {spec['key']} · "
            f"{'镜像' if self.mirror else '标准'} · 辫长 {len(self.word)} · "
            + ("✓ 结型不变" if same else "✗ 结型已改变"))


MainWindow = JonesApp


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
