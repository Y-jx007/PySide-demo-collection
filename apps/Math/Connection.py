# -*- coding: utf-8 -*-
"""Connection: parallel transport and covariant derivative on a surface.

Two interactive qualities are deliberate:

* the dragged point follows the pointer through a critically damped spring,
  so it accelerates and settles smoothly at any frame rate instead of
  lagging behind the cursor and snapping to it when the button is released;
* the ambient field carries a smooth length envelope, so the arrows grow and
  shrink continuously while the point travels, and the covariant derivative
  shows both the turning and the stretching of the field.
"""
import math

import numpy as np

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    QApplication, LabeledSlider, QCheckBox, QColor, QComboBox, QFileDialog, QGroupBox,
    setup_gl_format,
    QHBoxLayout, QLabel, QMessageBox, QPainter, QPen, QPointF, QPolygonF,
    QPixmap, QPushButton, QTimer, QVBoxLayout, QWidget, Qt, run_app,
)
from PySide6.QtCore import QElapsedTimer
from PySide6.QtGui import QOpenGLContext, QOffscreenSurface, QPainterPath, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from common.opengl_base import (
    GL_COLOR_ARRAY, GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_MODELVIEW, GL_PROJECTION,
    GL_QUADS, GL_VERTEX_ARRAY,
    glClear, glClearColor, glColorPointer, glDisableClientState,
    glDrawArrays, glEnableClientState, glLoadIdentity, glMatrixMode,
    glOrtho, glVertexPointer, glViewport,
)
from support.math.connection import (
    SURFACES, ambient_directions, field_envelope, make_surface,
    oriented_angle, parallel_transport,
)


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return np.asarray(v, dtype=float) / n


class Camera3D:
    def __init__(self):
        self.target = np.zeros(3)
        self.yaw = -58.0
        self.pitch = 20.0
        self.zoom = 1.0
        self.update()

    def update(self):
        y = math.radians(self.yaw)
        p = math.radians(self.pitch)
        eye_dir = np.array(
            [math.cos(p) * math.cos(y), math.sin(p), math.cos(p) * math.sin(y)]
        )
        self.forward = -eye_dir
        self.eye = self.target + eye_dir * 4.0
        self.right = unit(np.cross(self.forward, np.array([0.0, 1.0, 0.0])))
        self.up = unit(np.cross(self.right, self.forward))

    def project(self, p, width, height):
        rel = np.asarray(p, dtype=float) - self.eye
        scale = min(width, height) * 0.38 * self.zoom
        x = float(rel @ self.right) * scale + width * 0.5
        y = -float(rel @ self.up) * scale + height * 0.5
        depth = float(rel @ self.forward)
        return QPointF(x, y), depth

    def ray(self, x, y, width, height):
        scale = min(width, height) * 0.38 * self.zoom
        ox = (float(x) - width * 0.5) / scale
        oy = -(float(y) - height * 0.5) / scale
        origin = self.eye + self.right * ox + self.up * oy
        return origin, self.forward


class GpuSurface:
    """Rasterises the mesh with OpenGL into a QImage (offscreen FBO).

    The canvas stays a plain QWidget painted with QPainter, so the
    annotations keep their anti-aliased quality; only the thousands of
    shaded quads are handed to the GPU.  Falls back cleanly when no GL
    context can be created (headless tests, remote sessions).
    """

    def __init__(self):
        self.ok = False
        self.context = None
        self.surface = None
        self.fbo = None
        try:
            fmt = QSurfaceFormat.defaultFormat()
            self.surface = QOffscreenSurface()
            self.surface.setFormat(fmt)
            self.surface.create()
            self.context = QOpenGLContext()
            self.context.setFormat(fmt)
            self.ok = bool(self.context.create())
            self.ok = self.ok and bool(self.context.makeCurrent(self.surface))
            if self.ok:
                self.context.doneCurrent()
        except Exception:
            self.ok = False

    def render(self, canvas, pixel_ratio):
        if not self.ok:
            return None
        w = max(1, canvas.width())
        h = max(1, canvas.height())
        iw = max(1, int(round(w * pixel_ratio)))
        ih = max(1, int(round(h * pixel_ratio)))
        if not self.context.makeCurrent(self.surface):
            return None
        try:
            if self.fbo is None or self.fbo.width() != iw or self.fbo.height() != ih:
                fmt = QOpenGLFramebufferObjectFormat()
                fmt.setSamples(0)
                self.fbo = QOpenGLFramebufferObject(iw, ih, fmt)
            if not self.fbo.bind():
                return None
            glViewport(0, 0, iw, ih)
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            canvas.paint_surface_gl(w, h)
            self.fbo.release()
            image = self.fbo.toImage()
            image.setDevicePixelRatio(pixel_ratio)
            return image
        finally:
            self.context.doneCurrent()


class ConnectionCanvasBase:
    """The interactive surface canvas (drawing with QPainter)."""

    FOLLOW_MS = 16          # follow timer period, about 60 fps
    FOLLOW_OMEGA = 46.0     # rad/s, critically damped: no overshoot
    FOLLOW_AMAX = 140.0     # world units per second squared, ease-in bound
    FOLLOW_VMAX = 7.0       # world units per second
    FOLLOW_VSTOP = 0.06     # world units per second, counts as arrived
    FOLLOW_EPS = 1.2e-3     # world units, snap distance
    FIELD_GAIN = 0.62       # arrow length at envelope 1.0
    SHADE_STEPS = 96        # quantised shading levels for the surface
    PAPER_LIGHT = (242, 237, 225)   # lit sheet
    PAPER_DARK = (170, 159, 140)    # folded shadow
    PAPER_GRAIN = 2         # per-face jitter in shading levels
    MESH_CPU = (64, 54)     # quads; dense enough that the grid never shows
    MESH_GPU = (132, 110)   # the GPU can afford a much finer sheet
    PATH_MAX_POINTS = 400   # path samples kept for drawing
    BASE_SNAP = 0.12        # world units: pressing this near the base snaps to it
    CLOSE_TOL = 0.10        # world units: this near the start closes the loop
    CLOSE_HINT = 0.30       # world units: show where the loop started

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 600)
        self.window = window
        self.camera = Camera3D()
        self.surface = make_surface("球面")
        self._gpu = None
        if QApplication.platformName() not in ("offscreen", "minimal", "vnc"):
            gpu = GpuSurface()
            if gpu.ok:
                self._gpu = gpu
        self.mesh_size = self.MESH_GPU if self._gpu is not None else self.MESH_CPU
        self.mesh_fine = self._prepare_mesh(self.surface.mesh(*self.mesh_size))
        self.verts = self.mesh_fine["verts"]
        self.faces = self.mesh_fine["faces"]
        self.shade_colors = self._shade_colors()
        self._shade_rgb_cache = None
        self.count = 12
        self.field_amp = 0.60
        self.field_len = 1.0
        self.mode = "平行移动"
        self.show_ghost = True
        self.show_plane = True
        self.show_normal = True
        self.p_uv = self.surface.initial()
        self.comps = np.zeros((0, 2))
        self.initial_uv = self.p_uv
        self.base_uv = self.p_uv
        self.base_locked = True
        self.initial_comps = np.zeros((0, 2))
        self.path = []
        self.path_len = 0.0
        self.holonomy = None
        self.closure = None
        self.nabla = None
        self.nabla_vec = None
        self.ambient = ambient_directions(self.count)
        self.dragging_point = False
        self.dragging_camera = False
        self.last_mouse = None
        self._surface_cache = None
        self._surface_cache_key = None
        self._path_cache = None
        self._path_cache_key = None
        self._path_cache_head = None
        self._path_cache_n = 0
        self._path_cache_stride = 1
        self._path_vis = []

        self._target_uv = None
        self._finish_pending = False
        self._velocity = np.zeros(2)
        self._clock = QElapsedTimer()
        self._last_tick = 0.0
        self._glide = QTimer(self)
        self._glide.setTimerType(Qt.PreciseTimer)
        self._glide.setInterval(self.FOLLOW_MS)
        self._glide.timeout.connect(self._glide_tick)
        self.setCursor(Qt.CrossCursor)

    # ---------- session ----------
    def reset_session(self, base_uv=None):
        if base_uv is not None and not isinstance(base_uv, (tuple, list, np.ndarray)):
            base_uv = None
        self._glide.stop()
        self._target_uv = None
        self._finish_pending = False
        self._velocity[:] = 0.0
        self.dragging_point = False
        self.base_uv = self.surface.wrap_uv(
            *(self.surface.initial() if base_uv is None else base_uv)
        )
        self.p_uv = self.base_uv
        self.initial_uv = self.p_uv
        self.comps, _ = self.surface.cluster(self.p_uv[0], self.p_uv[1], self.count, 0.24)
        self.initial_comps = self.comps.copy()
        self.path = [self.p_uv]
        self.path_len = 0.0
        self.holonomy = None
        self.closure = None
        self.nabla = None
        self.nabla_vec = None
        self.ambient = ambient_directions(self.count)
        self.field_len = self._field_scale(self.p_uv)
        self._publish()
        self.update()

    def set_surface(self, name):
        self.surface = make_surface(name)
        self.mesh_fine = self._prepare_mesh(self.surface.mesh(*self.mesh_size))
        self.verts = self.mesh_fine["verts"]
        self.faces = self.mesh_fine["faces"]
        self._surface_cache = None
        self.reset_session()

    def set_count(self, count):
        self.count = max(2, int(count))
        if self.path_len < 1e-6:
            self.initial_uv = self.p_uv
            self.comps, _ = self.surface.cluster(self.p_uv[0], self.p_uv[1], self.count, 0.24)
            self.initial_comps = self.comps.copy()
        self.ambient = ambient_directions(self.count)
        self.update()

    def set_mode(self, mode):
        base = self.base_uv if self.base_locked else None
        self.mode = mode
        self.nabla = None
        self.nabla_vec = None
        self.reset_session(base)

    def set_base_locked(self, locked):
        """Pin the base point: every new stroke then starts there."""
        self.base_locked = bool(locked)
        self.update()

    def set_field_amp(self, percent):
        self.field_amp = min(max(float(percent) / 100.0, 0.0), 0.85)
        self.nabla_vec = None
        self.field_len = self._field_scale(self.p_uv)
        self._publish()
        self.update()

    def _publish(self):
        if self.mode == "向量场":
            self.window.set_metrics(
                self.path_len, self.holonomy, self.p_uv, field_len=self.field_len,
            )
        else:
            self.window.set_metrics(
                self.path_len, self.holonomy, self.p_uv, closure=self.closure,
            )

    # ---------- environment field ----------
    def _field_scale(self, uv):
        return field_envelope(self.surface.point(*uv), self.field_amp)

    def _ambient_vectors(self, uv):
        n = self.surface.normal(*uv)
        scale = self.FIELD_GAIN * self._field_scale(uv)
        return [scale * (f - float(f @ n) * n) for f in self.ambient]

    def _ambient_derivative(self, du, dv):
        norm = math.hypot(du, dv)
        if norm < 1e-9:
            return None
        eps = 2e-3
        u1, v1 = self.surface.wrap_uv(
            self.p_uv[0] + du / norm * eps,
            self.p_uv[1] + dv / norm * eps,
        )
        v0 = self._ambient_vectors(self.p_uv)[0]
        v1 = self._ambient_vectors((u1, v1))[0]
        n = self.surface.normal(*self.p_uv)
        d = (v1 - v0) / eps
        return d - float(d @ n) * n

    # ---------- painting ----------
    def _paint_overlays(self, painter):
        """Everything that sits on top of the surface image."""
        self._draw_path(painter)
        if self.show_plane:
            self._draw_tangent_plane(painter)
        self._draw_vectors(painter, ghost=False)
        if self.show_ghost and self.path_len > 0.03:
            self._draw_vectors(painter, ghost=True)
        if self.show_normal:
            self._draw_normal(painter)
        if self.nabla_vec is not None and self.mode == "向量场":
            self._draw_nabla(painter)
        if self.mode == "平行移动" or self.base_locked:
            self._draw_start_marker(painter)
        if self.mode == "平行移动":
            self._draw_holonomy(painter)
        self._draw_point(painter)

    # ---------- surface geometry ----------
    def _shade_rgb(self):
        """Palette as float triples, shaped for a GL vertex-colour array."""
        if self._shade_rgb_cache is None:
            self._shade_rgb_cache = np.array(
                [[c.redF(), c.greenF(), c.blueF()] for c in self.shade_colors],
                dtype=np.float32,
            )
        return self._shade_rgb_cache

    def _surface_geometry(self):
        """Project the mesh: screen corners, far-to-near order, shade level."""
        mesh = self.mesh_fine
        w, h = self.width(), self.height()
        verts = mesh["verts"]
        fi, fj = mesh["fi"], mesh["fj"]
        rel = verts - self.camera.eye
        scale = min(w, h) * 0.38 * self.camera.zoom
        sx = (rel @ self.camera.right) * scale + w * 0.5
        sy = -(rel @ self.camera.up) * scale + h * 0.5
        depth = rel @ self.camera.forward
        px = sx[fi, fj]
        py = sy[fi, fj]
        face_depth = depth[fi, fj].mean(axis=1)
        p0 = verts[fi[:, 0], fj[:, 0]]
        p1 = verts[fi[:, 1], fj[:, 1]]
        p3 = verts[fi[:, 3], fj[:, 3]]
        normal = np.cross(p1 - p0, p3 - p0)
        length = np.linalg.norm(normal, axis=1)
        light = unit(np.array([0.35, -0.75, 0.65]))
        cosang = np.clip((normal @ light) / np.maximum(length, 1e-12), 0.0, 1.0)
        steps = self.SHADE_STEPS
        tone = np.clip(0.22 + 0.78 * cosang, 0.0, 1.0)
        level = np.clip(np.rint(tone * steps).astype(int) + mesh["grain"], 0, steps)
        order = np.argsort(-face_depth)
        return px, py, order, level

    def _surface_pixmap(self):
        key = (
            self.surface.name, round(self.camera.yaw, 2), round(self.camera.pitch, 2),
            round(self.camera.zoom, 3), self.width(), self.height(),
        )
        if self._surface_cache is not None and key == self._surface_cache_key:
            return self._surface_cache
        pix = QPixmap(self.width(), self.height())
        pix.fill(QColor(255, 255, 255))
        image = None
        if self._gpu is not None:
            image = self._gpu.render(self, self.devicePixelRatioF())
        p = QPainter(pix)
        if image is not None:
            p.drawImage(0, 0, image)
        else:
            # no antialiasing on the surface: adjacent quads then tile
            # exactly, which removes the pale seams that read as a grid
            p.setRenderHint(QPainter.Antialiasing, False)
            self._draw_surface(p, self.mesh_fine)
        p.end()
        self._surface_cache = pix
        self._surface_cache_key = key
        return pix

    def _prepare_mesh(self, mesh):
        """Pre-array a mesh so a redraw only does vector arithmetic."""
        verts, faces = mesh
        index = np.asarray(faces, dtype=int)
        rng = np.random.default_rng(20261115)
        grain = np.rint(
            (rng.random(len(index)) - 0.5) * 2.0 * self.PAPER_GRAIN
        ).astype(int)
        verts = np.asarray(verts, dtype=float)
        normals = np.cross(
            np.gradient(verts, axis=0), np.gradient(verts, axis=1)
        )
        length = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / np.maximum(length, 1e-12)
        normals *= self.surface.normal_flip
        nu, nv = verts.shape[0] - 1, verts.shape[1] - 1
        return {
            "verts": verts,
            "fi": index[:, :, 0],
            "fj": index[:, :, 1],
            "faces": faces,
            "grain": grain,
            "normals": normals,
            "span": (nu, nv),
        }

    def _mesh_normal(self, u, v):
        """Nearest grid normal: cheap enough to run for every path sample."""
        mesh = self.mesh_fine
        nu, nv = mesh["span"]
        surf = self.surface
        i = int(round((u - surf.u_range[0]) / (surf.u_range[1] - surf.u_range[0]) * nu))
        j = int(round((v - surf.v_range[0]) / (surf.v_range[1] - surf.v_range[0]) * nv))
        i = max(0, min(nu, i))
        j = max(0, min(nv, j))
        return mesh["normals"][i, j]

    def _shade_colors(self):
        """Paper tones: warm, matte, one hue from shadow to lit sheet."""
        steps = self.SHADE_STEPS
        dark, light = self.PAPER_DARK, self.PAPER_LIGHT
        colors = []
        for k in range(steps + 1):
            t = k / steps
            colors.append(QColor(
                int(round(dark[0] + (light[0] - dark[0]) * t)),
                int(round(dark[1] + (light[1] - dark[1]) * t)),
                int(round(dark[2] + (light[2] - dark[2]) * t)),
                255,
            ))
        return colors

    def _draw_surface(self, painter, mesh=None):
        px, py, order, level = self._surface_geometry()
        palette = self.shade_colors
        painter.setPen(Qt.NoPen)
        draw = painter.drawPolygon
        point = QPointF
        # one reusable polygon: no per-face container allocation
        quad = QPolygonF([point(0.0, 0.0)] * 4)
        for k in order.tolist():
            quad[0] = point(px[k, 0], py[k, 0])
            quad[1] = point(px[k, 1], py[k, 1])
            quad[2] = point(px[k, 2], py[k, 2])
            quad[3] = point(px[k, 3], py[k, 3])
            painter.setBrush(palette[level[k]])
            draw(quad)

    def _path_screen(self):
        """Projected path plus which samples still face the camera.

        Both lists are cached between frames; the camera or a longer path
        invalidates them, and only new samples are projected while dragging.
        """
        head = self.path[0]
        count = len(self.path)
        stride = max(1, int(math.ceil(count / float(self.PATH_MAX_POINTS))))
        camera_key = (
            round(self.camera.yaw, 2), round(self.camera.pitch, 2),
            round(self.camera.zoom, 3), stride,
        )
        cached = self._path_cache
        if (
            cached is None
            or self._path_cache_key != camera_key
            or self._path_cache_head != head
            or count < self._path_cache_n
        ):
            sampled = self.path[::stride]
            if stride > 1 and sampled[-1] != self.path[-1]:
                sampled = list(sampled) + [self.path[-1]]
            project = self.camera.project
            w, h, eye = self.width(), self.height(), self.camera.eye
            point_at = self.surface.point
            normal_at = self._mesh_normal
            cached = []
            visible = []
            for u, v in sampled:
                p = point_at(u, v)
                cached.append(project(p, w, h)[0])
                visible.append(float(normal_at(u, v) @ (eye - p)) > 0.0)
            self._path_cache = cached
            self._path_vis = visible
            self._path_cache_key = camera_key
            self._path_cache_head = head
            self._path_cache_n = count
            self._path_cache_stride = stride
            return cached
        if count > self._path_cache_n and self._path_cache_stride == 1:
            project = self.camera.project
            w, h, eye = self.width(), self.height(), self.camera.eye
            point_at = self.surface.point
            normal_at = self._mesh_normal
            for u, v in self.path[self._path_cache_n:]:
                p = point_at(u, v)
                cached.append(project(p, w, h)[0])
                self._path_vis.append(float(normal_at(u, v) @ (eye - p)) > 0.0)
            self._path_cache_n = count
        return cached

    def _draw_path(self, painter):
        if len(self.path) < 2:
            return
        pts = self._path_screen()
        visible = self._path_vis
        painter.setBrush(Qt.NoBrush)
        n = len(pts)
        start = 0
        while start < n - 1:
            front = visible[start]
            end = start + 1
            while end < n and visible[end] == front:
                end += 1
            # overlap one sample so the pieces stay joined
            run = pts[start:min(end + 1, n)]
            if len(run) >= 2:
                self._stroke_path(painter, run, front)
            start = end

    def _stroke_path(self, painter, pts, front):
        if front:
            painter.setPen(QPen(QColor(39, 111, 168, 170), 2.2))
        else:
            # behind the surface: dashed, as if seen through the paper
            painter.setPen(QPen(QColor(58, 100, 142, 145), 1.6, Qt.DashLine))
        if len(pts) < 3:
            painter.drawPolyline(QPolygonF(pts))
            return
        path = QPainterPath()
        path.moveTo(pts[0])
        for i in range(1, len(pts) - 1):
            mid = QPointF(
                (pts[i].x() + pts[i + 1].x()) * 0.5,
                (pts[i].y() + pts[i + 1].y()) * 0.5,
            )
            path.quadTo(pts[i], mid)
        path.lineTo(pts[-1])
        painter.drawPath(path)

    def _chart_vecs(self):
        p = self.surface.point(*self.p_uv)
        return p, [
            self.surface.components_to_vector(self.p_uv[0], self.p_uv[1], c)
            for c in self.comps
        ]

    def _draw_tangent_plane(self, painter):
        p = self.surface.point(*self.p_uv)
        n = self.surface.normal(*self.p_uv)
        e1, e2 = self.surface.tangent_basis(*self.p_uv)
        pts = []
        for k in range(40):
            a = 2.0 * math.pi * k / 40
            q3 = p + 0.27 * (math.cos(a) * e1 + math.sin(a) * e2) + n * 0.004
            q, _ = self.camera.project(q3, self.width(), self.height())
            pts.append(q)
        painter.setPen(QPen(QColor(60, 120, 150, 95), 0.9))
        painter.setBrush(QColor(90, 160, 190, 28))
        painter.drawPolygon(QPolygonF(pts))

    def _draw_arrow(self, painter, p0, p1, color, width=1.8, head=7.0):
        q0, _ = self.camera.project(p0, self.width(), self.height())
        q1, _ = self.camera.project(p1, self.width(), self.height())
        painter.setPen(QPen(color, width))
        painter.drawLine(q0, q1)
        dx = q1.x() - q0.x()
        dy = q1.y() - q0.y()
        length = math.hypot(dx, dy)
        if length < 2.0:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        base = QPointF(q1.x() - ux * head, q1.y() - uy * head)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([
            q1,
            QPointF(base.x() + px * head * 0.45, base.y() + py * head * 0.45),
            QPointF(base.x() - px * head * 0.45, base.y() - py * head * 0.45),
        ]))

    def _draw_vectors(self, painter, ghost=False):
        if ghost:
            pu = self.initial_uv
            p = self.surface.point(*pu)
            n = self.surface.normal(*pu)
            if self.mode == "向量场":
                vecs = self._ambient_vectors(pu)
            else:
                vecs = [
                    self.surface.components_to_vector(pu[0], pu[1], c)
                    for c in self.initial_comps
                ]
            cols = [QColor(138, 147, 158, 135) for _ in vecs]
        else:
            pu = self.p_uv
            p = self.surface.point(*pu)
            n = self.surface.normal(*pu)
            if self.mode == "向量场":
                vecs = self._ambient_vectors(pu)
            else:
                vecs = [
                    self.surface.components_to_vector(pu[0], pu[1], c)
                    for c in self.comps
                ]
            # one hue, light to dark around the fan
            last = max(1, len(vecs) - 1)
            cols = [
                QColor.fromHsv(
                    206,
                    int(round(130 + 70 * k / last)),
                    int(round(200 - 104 * k / last)),
                )
                for k in range(len(vecs))
            ]
        for vec, col in zip(vecs, cols):
            if float(np.linalg.norm(vec)) < 1e-6:
                continue
            end = p + vec
            self._draw_arrow(painter, p + n * 0.005, end + n * 0.005, col, 1.7 if not ghost else 1.35)
        if not ghost:
            painter.setPen(QPen(QColor(45, 55, 65, 190), 1.0))
            painter.setBrush(QColor(255, 255, 255))
            q, _ = self.camera.project(p, self.width(), self.height())
            painter.drawEllipse(q, 4.2, 4.2)

    def _draw_normal(self, painter):
        p = self.surface.point(*self.p_uv)
        n = self.surface.normal(*self.p_uv)
        self._draw_arrow(painter, p, p + n * 0.30, QColor(110, 120, 130, 150), 1.0, 5.0)

    def _draw_nabla(self, painter):
        p = self.surface.point(*self.p_uv)
        n = self.surface.normal(*self.p_uv)
        vec = self.nabla_vec
        if vec is None:
            return
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            return
        vec = vec / length * 0.34
        self._draw_arrow(painter, p + n * 0.010, p + vec + n * 0.010, QColor(218, 66, 52), 2.5, 8.0)

    def _draw_start_marker(self, painter):
        """Dashed ring on the base point (always, while it is pinned)."""
        near = self.closure is not None and self.closure <= self.CLOSE_HINT
        if not (self.base_locked or near):
            return
        uv = self.base_uv if self.base_locked else self.initial_uv
        q, _ = self.camera.project(
            self.surface.point(*uv), self.width(), self.height()
        )
        painter.setPen(QPen(QColor(122, 134, 148, 180), 1.1, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(q, 6.5, 6.5)

    def _draw_holonomy(self, painter):
        """Arc from the initial vector to the one that came back around."""
        if self.holonomy is None or len(self.initial_comps) == 0:
            return
        uv = self.initial_uv
        p0 = self.surface.point(*uv)
        n = self.surface.normal(*uv)
        e1, e2 = self.surface.tangent_basis(*uv)
        v0 = self.surface.components_to_vector(uv[0], uv[1], self.initial_comps[0])
        length = float(np.linalg.norm(v0))
        if length < 1e-9:
            return
        a0 = math.atan2(float(v0 @ e2), float(v0 @ e1))
        radius = max(0.34, length * 1.35)
        sweep = math.radians(self.holonomy)
        steps = max(8, int(abs(self.holonomy) / 5.0) + 4)
        pts = []
        for k in range(steps + 1):
            a = a0 + sweep * k / steps
            q3 = p0 + radius * (math.cos(a) * e1 + math.sin(a) * e2) + n * 0.008
            q, _ = self.camera.project(q3, self.width(), self.height())
            pts.append(q)
        tick = QPen(QColor(206, 96, 40, 150), 1.0, Qt.DashLine)
        painter.setPen(tick)
        painter.setBrush(Qt.NoBrush)
        for angle in (a0, a0 + sweep):
            inner = p0 + radius * 0.78 * (math.cos(angle) * e1 + math.sin(angle) * e2)
            outer = p0 + radius * (math.cos(angle) * e1 + math.sin(angle) * e2)
            qi, _ = self.camera.project(inner + n * 0.008, self.width(), self.height())
            qo, _ = self.camera.project(outer + n * 0.008, self.width(), self.height())
            painter.drawLine(qi, qo)
        color = QColor(206, 96, 40, 235)
        painter.setPen(QPen(color, 2.1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(QPolygonF(pts))
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(pts[0], 2.4, 2.4)
        if len(pts) < 2:
            return
        end, prev = pts[-1], pts[-2]
        dx, dy = end.x() - prev.x(), end.y() - prev.y()
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        head = 6.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([
            end,
            QPointF(end.x() - ux * head + px * head * 0.45,
                    end.y() - uy * head + py * head * 0.45),
            QPointF(end.x() - ux * head - px * head * 0.45,
                    end.y() - uy * head - py * head * 0.45),
        ]))

    def _draw_point(self, painter):
        p = self.surface.point(*self.p_uv)
        q, _ = self.camera.project(p, self.width(), self.height())
        painter.setPen(QPen(QColor(30, 42, 54), 1.5))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(q, 5.0, 5.0)
        painter.setBrush(QColor(30, 42, 54))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(q, 1.8, 1.8)

    # ---------- interaction ----------
    def _pick_uv(self, pos):
        origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
        hint = self.p_uv if self.dragging_point else None
        return self.surface.nearest_uv_to_ray(
            origin, direction, samples=30, hint=hint, tolerance=0.02,
        )

    def _begin_drag(self, uv):
        uv = self.surface.wrap_uv(*uv)
        self._glide.stop()
        self._finish_pending = False
        self._velocity[:] = 0.0
        if not self.base_locked:
            self.base_uv = uv
        self._target_uv = uv
        self.p_uv = uv
        self.initial_uv = uv
        self.comps, _ = self.surface.cluster(uv[0], uv[1], self.count, 0.24)
        self.initial_comps = self.comps.copy()
        self.path = [uv]
        self.path_len = 0.0
        self.holonomy = None
        self.closure = None
        self.nabla = None
        self.nabla_vec = None
        self.ambient = ambient_directions(self.count)
        self.field_len = self._field_scale(uv)
        self._publish()

    def _advance(self, uv):
        uv = self.surface.wrap_uv(*uv)
        u0, v0 = self.p_uv
        du, dv = self.surface.delta_uv(u0, v0, uv[0], uv[1])
        un, vn = self.surface.wrap_uv(u0 + du, v0 + dv)
        dist = float(np.linalg.norm(
            self.surface.point(un, vn) - self.surface.point(u0, v0)
        ))
        if dist < 1e-7:
            return
        if self.mode == "平行移动":
            steps = max(4, min(32, int(dist / 0.02) + 1))
            self.comps = parallel_transport(
                self.surface, (u0, v0), (un, vn), self.comps, steps=steps
            )
            self.nabla = None
            self.nabla_vec = None
        else:
            self.nabla_vec = self._ambient_derivative(du, dv)
        self.path_len += dist
        self.p_uv = (un, vn)
        if dist > 2e-3 or len(self.path) < 2:
            self.path.append(self.p_uv)
        if len(self.path) > 2400:
            self.path = self.path[-2400:]
        self.field_len = self._field_scale(self.p_uv)
        if self.mode == "平行移动":
            self._update_holonomy()
        self._publish()

    def _update_holonomy(self):
        """Measure the rotation picked up along the path back to the start.

        The last few millimetres are closed with a short transport, so the
        loop only has to *almost* return to the start; the residual neck is
        far too small to bias the angle visibly.
        """
        self.closure = float(np.linalg.norm(
            self.surface.point(*self.p_uv) - self.surface.point(*self.initial_uv)
        ))
        self.holonomy = None
        if self.mode != "平行移动" or self.path_len < 0.20:
            return
        if self.closure > self.CLOSE_TOL or len(self.comps) == 0:
            return
        closed = parallel_transport(
            self.surface, self.p_uv, self.initial_uv, self.comps, steps=3,
        )
        n = self.surface.normal(*self.initial_uv)
        v0 = self.surface.components_to_vector(
            self.initial_uv[0], self.initial_uv[1], self.initial_comps[0]
        )
        v1 = self.surface.components_to_vector(
            self.initial_uv[0], self.initial_uv[1], closed[0]
        )
        self.holonomy = oriented_angle(n, v0, v1)

    def _finish_drag(self):
        if self.mode == "平行移动":
            self._update_holonomy()
        self._publish()
        self.update()

    # ---------- smooth follow ----------
    def _start_glide(self):
        if self._glide.isActive():
            return
        self._clock.start()
        self._last_tick = 0.0
        self._glide.start()

    def _world_speed(self, u, v, velocity):
        vec = self.surface.components_to_vector(u, v, velocity)
        return float(np.linalg.norm(vec))

    def _clamp_world(self, u, v, vector, limit):
        """Scale chart components down until their world length fits the limit."""
        speed = self._world_speed(u, v, vector)
        if speed > limit:
            return vector * (limit / speed)
        return vector

    def _settled(self):
        if self._target_uv is None:
            return True
        gap = self.surface.point(*self._target_uv) - self.surface.point(*self.p_uv)
        if float(np.linalg.norm(gap)) >= self.FOLLOW_EPS:
            return False
        return self._world_speed(self.p_uv[0], self.p_uv[1], self._velocity) < self.FOLLOW_VSTOP

    def _follow(self, dt):
        """Advance the spring for dt seconds; return the new chart point."""
        if self._target_uv is None:
            return self.p_uv, True
        u, v = self.p_uv
        target = self.surface.point(*self._target_uv)
        velocity = np.array(self._velocity, dtype=float, copy=True)
        steps = max(1, min(8, int(math.ceil(dt / 0.010))))
        h = dt / steps
        arrived = False
        for _ in range(steps):
            gap = target - self.surface.point(u, v)
            if (
                float(np.linalg.norm(gap)) < self.FOLLOW_EPS
                and self._world_speed(u, v, velocity) < self.FOLLOW_VSTOP
            ):
                u, v = self._target_uv
                velocity[:] = 0.0
                arrived = True
                break
            delta = self.surface.vector_to_components(u, v, gap)
            accel = self.FOLLOW_OMEGA ** 2 * delta - 2.0 * self.FOLLOW_OMEGA * velocity
            # a bounded acceleration makes the first frames an ease-in
            accel = self._clamp_world(u, v, accel, self.FOLLOW_AMAX)
            velocity = velocity + accel * h
            velocity = self._clamp_world(u, v, velocity, self.FOLLOW_VMAX)
            du, dv = velocity * h
            u, v = self.surface.wrap_uv(u + du, v + dv)
        self._velocity = velocity
        return self.surface.wrap_uv(u, v), arrived

    def _glide_tick(self):
        if self._target_uv is None:
            self._glide.stop()
            return
        elapsed = self._clock.elapsed() * 1e-3
        dt = min(max(elapsed - self._last_tick, 0.002), 0.06)
        self._last_tick = elapsed
        uv, arrived = self._follow(dt)
        self._advance(uv)
        if arrived:
            self._glide.stop()
            if self._finish_pending:
                self._finish_pending = False
                self._finish_drag()
        self.update()

    def _start_stroke(self, uv):
        """A press keeps drawing: the trail never restarts on its own.

        With the base pinned the session (path, transported vectors, base)
        carries on, and the point simply walks to wherever you pressed,
        extending the same trail.  Unlock the base to start over.
        """
        if not self.base_locked:
            self._begin_drag(uv)
            self.dragging_point = True
            return
        gap = self.surface.point(*uv) - self.surface.point(*self.p_uv)
        self._target_uv = uv
        if float(np.linalg.norm(gap)) > self.FOLLOW_EPS:
            self._start_glide()
        self.dragging_point = True

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            uv = self._pick_uv(event.position())
            if uv is not None:
                self._start_stroke(uv)
                self.update()
        elif event.button() == Qt.RightButton:
            self.dragging_camera = True
            self.last_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self.dragging_point and event.buttons() & Qt.LeftButton:
            uv = self._pick_uv(event.position())
            if uv is not None:
                self._target_uv = uv
                self._finish_pending = False
                self._start_glide()
        elif self.dragging_camera and event.buttons() & Qt.RightButton:
            if self.last_mouse is not None:
                dx = event.position().x() - self.last_mouse.x()
                dy = event.position().y() - self.last_mouse.y()
                self.camera.yaw += dx * 0.45
                self.camera.pitch = max(-85.0, min(85.0, self.camera.pitch + dy * 0.45))
                self.camera.update()
                self.last_mouse = event.position()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging_point:
            self.dragging_point = False
            if self._settled():
                self._finish_pending = False
                self._finish_drag()
            else:
                # glide the last millimetres, then close the session there
                self._finish_pending = True
                self._start_glide()
        elif event.button() == Qt.RightButton:
            self.dragging_camera = False
            self.last_mouse = None
            self._surface_cache = None
            self.update()

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self.camera.zoom = max(0.55, min(2.6, self.camera.zoom * factor))
        self.update()


class ConnectionCanvas(ConnectionCanvasBase, SquareCanvas):
    """The canvas the window uses; the surface comes from the GPU if possible."""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.drawPixmap(0, 0, self._surface_pixmap())
        self._paint_overlays(painter)
        painter.end()

    def paint_surface_gl(self, w, h):
        """Push every shaded quad to the GPU in one call."""
        px, py, order, level = self._surface_geometry()
        quad_xy = np.stack([px[order], py[order]], axis=2)
        xy = np.ascontiguousarray(quad_xy.reshape(-1, 2), dtype=np.float32)
        rgb = np.ascontiguousarray(
            self._shade_rgb()[level[order]].repeat(4, axis=0), dtype=np.float32
        )
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, float(w), float(h), 0.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, xy)
        glColorPointer(3, GL_FLOAT, 0, rgb)
        glDrawArrays(GL_QUADS, 0, xy.shape[0])
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("联络 · 协变导数")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.canvas = ConnectionCanvas(self)
        self.init_ui()
        self.canvas.reset_session()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        central = QWidget()
        central.setObjectName("connectionCentral")
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        panel = QWidget()
        panel.setObjectName("connectionPanel")
        panel.setFixedWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 4, 4)
        layout.setSpacing(6)

        surface_group = QGroupBox("曲面")
        sg = QVBoxLayout(surface_group)
        sg.setContentsMargins(6, 6, 6, 6)
        sg.setSpacing(5)
        self.surface_combo = QComboBox()
        self.surface_combo.addItems(SURFACES)
        self.surface_combo.currentTextChanged.connect(self.on_surface)
        sg.addWidget(self.surface_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(("平行移动", "向量场"))
        self.mode_combo.currentTextChanged.connect(self.on_mode)
        sg.addWidget(self.mode_combo)
        self.count_slider = LabeledSlider(
            "向量数", 4, 24, 12, formatter=str, label_width=64, value_width=40
        )
        self.count_slider.valueChanged.connect(self.canvas.set_count)
        sg.addWidget(self.count_slider)
        layout.addWidget(surface_group)

        base_group = QGroupBox("基点")
        bg = QVBoxLayout(base_group)
        bg.setContentsMargins(6, 6, 6, 6)
        bg.setSpacing(4)
        self.base_check = QCheckBox("固定基点")
        self.base_check.setChecked(self.canvas.base_locked)
        self.base_check.setToolTip("勾选后基点固定不动；在基点附近按下会从基点出发，在别处按下不会回到基点。取消勾选后，在曲面上一按即可换基点。")
        self.base_check.toggled.connect(self.on_base_lock)
        bg.addWidget(self.base_check)
        layout.addWidget(base_group)

        field_group = QGroupBox("环境场")
        fg = QVBoxLayout(field_group)
        fg.setContentsMargins(6, 6, 6, 6)
        fg.setSpacing(5)
        self.field_slider = LabeledSlider(
            "场长起伏", 0, 85, 60,
            formatter=lambda v: str(v) + "%", label_width=64, value_width=40,
        )
        self.field_slider.valueChanged.connect(self.canvas.set_field_amp)
        self.field_slider.setEnabled(False)   # only used by 向量场
        fg.addWidget(self.field_slider)
        layout.addWidget(field_group)

        view_group = QGroupBox("显示")
        vg = QVBoxLayout(view_group)
        vg.setContentsMargins(6, 6, 6, 6)
        vg.setSpacing(4)
        self.ghost_check = QCheckBox("初始向量")
        self.ghost_check.setChecked(True)
        self.ghost_check.toggled.connect(self.on_view)
        vg.addWidget(self.ghost_check)
        self.plane_check = QCheckBox("切平面")
        self.plane_check.setChecked(True)
        self.plane_check.toggled.connect(self.on_view)
        vg.addWidget(self.plane_check)
        self.normal_check = QCheckBox("法向量")
        self.normal_check.setChecked(True)
        self.normal_check.toggled.connect(self.on_view)
        vg.addWidget(self.normal_check)
        layout.addWidget(view_group)

        info_group = QGroupBox("读数")
        ig = QVBoxLayout(info_group)
        ig.setContentsMargins(6, 6, 6, 6)
        ig.setSpacing(4)
        self.metric_label = QLabel("路径长度 0.00")
        self.holonomy_label = QLabel("holonomy —")
        self.uv_label = QLabel("u 0.00  v 0.00")
        self.field_label = QLabel("场长 ×1.00")
        for label in (self.metric_label, self.holonomy_label, self.uv_label, self.field_label):
            label.setStyleSheet("color:#56616c;")
            ig.addWidget(label)
        note = QLabel("左键拖动点　右键旋转　滚轮缩放\n平行移动：长度不变，方向转 holonomy\n向量场：红箭头 = ∇u v")
        note.setStyleSheet("color:#7a828c;")
        note.setWordWrap(True)
        ig.addWidget(note)
        layout.addWidget(info_group)

        controls = QHBoxLayout()
        self.reset_btn = QPushButton("重置")
        self.save_btn = QPushButton("保存")
        self.reset_btn.clicked.connect(lambda: self.canvas.reset_session())
        self.save_btn.clicked.connect(self.save)
        controls.addWidget(self.reset_btn)
        controls.addWidget(self.save_btn)
        layout.addLayout(controls)
        layout.addStretch(1)

        root.addWidget(panel)
        root.addWidget(self.canvas, 0, Qt.AlignTop)
        central.setStyleSheet("QWidget#connectionCentral { background-color: #ffffff; }")
        panel.setStyleSheet("QWidget#connectionPanel { background-color: #ffffff; }")
        self.setCentralWidget(central)
        self.resize(960, 640)
        self.setMinimumSize(960, 640)

    @staticmethod
    def _write(label, text):
        if label.text() != text:
            label.setText(text)

    def set_metrics(self, path_len, holonomy, uv, field_len=None, closure=None):
        self._write(self.metric_label, f"路径长度 {path_len:.2f}")
        if holonomy is None:
            self._write(self.holonomy_label, "holonomy —")
        else:
            self._write(self.holonomy_label, f"holonomy {holonomy:.1f}°")
        self._write(self.uv_label, f"u {uv[0]:.2f}  v {uv[1]:.2f}")
        if field_len is not None:
            self._write(self.field_label, f"场长 ×{field_len:.2f}")
        elif closure is None:
            self._write(self.field_label, "离起点 —")
        elif holonomy is not None:
            self._write(self.field_label, f"离起点 {closure:.2f}  已闭合")
        else:
            self._write(self.field_label, f"离起点 {closure:.2f}")

    def on_surface(self, name):
        self.canvas.set_surface(name)

    def on_mode(self, mode):
        self.canvas.set_mode(mode)
        self.field_slider.setEnabled(mode == "向量场")

    def on_base_lock(self, locked):
        self.canvas.set_base_locked(locked)

    def on_view(self, *_):
        self.canvas.show_ghost = self.ghost_check.isChecked()
        self.canvas.show_plane = self.plane_check.isChecked()
        self.canvas.show_normal = self.normal_check.isChecked()
        self.canvas.update()

    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存联络图", "connection.png", "PNG (*.png)"
        )
        if not path:
            return
        if not self.canvas.save_image(path):
            QMessageBox.warning(self, "保存失败", "无法写入所选路径。")


if __name__ == "__main__":
    setup_gl_format()
    raise SystemExit(run_app(MainWindow))
