# -*- coding: utf-8 -*-
"""Shared 3D plumbing: an orbit camera and an offscreen GPU rasteriser.

Thousands of shaded polygons do not belong on the QPainter path: about 8 us
per polygon adds up to tens of milliseconds a frame.  `GpuSurface` renders
them into a QImage with OpenGL instead, and the canvas keeps drawing its
annotations with QPainter on top of that image — anti-aliased lines stay
crisp and the surface stays cheap.

`GpuSurface.create()` returns None when the platform cannot give us a GL
context (headless tests, remote sessions), so callers fall back to a plain
QPainter renderer without special cases.
"""
import math

import numpy as np

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from PySide6.QtWidgets import QApplication

from common.opengl_base import (
    GL_COLOR_ARRAY, GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_MODELVIEW, GL_PROJECTION,
    GL_QUADS, GL_TRIANGLES, GL_VERTEX_ARRAY,
    glClear, glClearColor, glColorPointer, glDisableClientState, glDrawArrays,
    glEnableClientState, glLoadIdentity, glMatrixMode, glOrtho, glVertexPointer,
    glViewport,
)

# Platforms whose Qt plugin cannot create an OpenGL context at all.
NO_GL_PLATFORMS = ("offscreen", "minimal", "vnc")


def unit(v):
    """Normalised copy of a 3-vector; the zero vector maps to itself."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return np.asarray(v, dtype=float) / n


class Camera3D:
    """Orbit camera around `target`, with yaw/pitch in degrees.

    `project` maps a world point to screen pixels and returns the depth along
    the view direction; `ray` undoes it so a screen point can be picked back
    onto a surface.
    """

    def __init__(self, target=(0.0, 0.0, 0.0), distance=4.0, yaw=-58.0, pitch=20.0):
        self.target = np.asarray(target, dtype=float)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.zoom = 1.0
        self.update()

    def update(self):
        y = math.radians(self.yaw)
        p = math.radians(self.pitch)
        eye_dir = np.array(
            [math.cos(p) * math.cos(y), math.sin(p), math.cos(p) * math.sin(y)]
        )
        self.forward = -eye_dir
        self.eye = self.target + eye_dir * self.distance
        self.right = unit(np.cross(self.forward, np.array([0.0, 1.0, 0.0])))
        self.up = unit(np.cross(self.right, self.forward))

    def world_per_pixel(self, width, height):
        """World units covered by one screen pixel at the current zoom."""
        return 1.0 / (min(width, height) * 0.38 * self.zoom)

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


def _begin_ortho(width, height):
    """Pixel coordinates with y pointing down, matching QPainter."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def _draw_arrays(xy, rgb, mode, width, height):
    xy = np.ascontiguousarray(xy, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    _begin_ortho(width, height)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(2, GL_FLOAT, 0, xy)
    glColorPointer(3, GL_FLOAT, 0, rgb)
    glDrawArrays(mode, 0, xy.shape[0])
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)


def quad_vertex_batch(xy, rgb):
    """Flatten (n, 4, 2) corners + (n, 3) colours into vertex arrays.

    Kept separate from the GL call so the batching itself can be tested
    without a GL context.
    """
    xy = np.asarray(xy, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1] != 4 or xy.shape[2] != 2:
        raise ValueError(f"expected (n, 4, 2) corners, got {xy.shape}")
    if rgb.shape != (xy.shape[0], 3):
        raise ValueError(f"expected (n, 3) colours, got {rgb.shape}")
    return xy.reshape(-1, 2), np.repeat(rgb, 4, axis=0)


def triangle_vertex_batch(xy, rgb):
    """Flatten (n, 3, 2) corners + (n, 3) colours into vertex arrays."""
    xy = np.asarray(xy, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1] != 3 or xy.shape[2] != 2:
        raise ValueError(f"expected (n, 3, 2) corners, got {xy.shape}")
    if rgb.shape != (xy.shape[0], 3):
        raise ValueError(f"expected (n, 3) colours, got {rgb.shape}")
    return xy.reshape(-1, 2), np.repeat(rgb, 3, axis=0)


def draw_quads(xy, rgb, width, height):
    """Draw (n, 4, 2) quads with one colour per quad; needs a GL context."""
    vertices, colours = quad_vertex_batch(xy, rgb)
    _draw_arrays(vertices, colours, GL_QUADS, width, height)


def draw_triangles(xy, rgb, width, height):
    """Draw (n, 3, 2) triangles with one colour per triangle; needs GL."""
    vertices, colours = triangle_vertex_batch(xy, rgb)
    _draw_arrays(vertices, colours, GL_TRIANGLES, width, height)


class GpuSurface:
    """Rasterises polygon batches with OpenGL into a QImage (offscreen FBO).

    Use `GpuSurface.create()`; the constructor itself does not touch GL so it
    stays safe to import and to build in tests.
    """

    def __init__(self):
        self.ok = False
        self.context = None
        self.surface = None
        self.fbo = None

    @classmethod
    def create(cls):
        """Ready instance, or None when this platform has no GL."""
        if QApplication.platformName() in NO_GL_PLATFORMS:
            return None
        gpu = cls()
        try:
            fmt = QSurfaceFormat.defaultFormat()
            gpu.surface = QOffscreenSurface()
            gpu.surface.setFormat(fmt)
            gpu.surface.create()
            gpu.context = QOpenGLContext()
            gpu.context.setFormat(fmt)
            ok = bool(gpu.context.create())
            ok = ok and bool(gpu.context.makeCurrent(gpu.surface))
            if ok:
                gpu.context.doneCurrent()
        except Exception:
            ok = False
        gpu.ok = ok
        return gpu if ok else None

    def render(self, width, height, pixel_ratio, draw):
        """Call `draw(width, height)` inside an FBO and return the QImage.

        `draw` receives logical pixels (the same space the canvas uses) and
        must only issue GL calls; it runs with the FBO bound.
        """
        if not self.ok or self.context is None:
            return None
        width = max(1, int(width))
        height = max(1, int(height))
        iw = max(1, int(round(width * pixel_ratio)))
        ih = max(1, int(round(height * pixel_ratio)))
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
            draw(width, height)
            self.fbo.release()
            image = self.fbo.toImage()
            image.setDevicePixelRatio(pixel_ratio)
            return image
        finally:
            self.context.doneCurrent()
