# -*- coding: utf-8 -*-
"""Reusable shaded-mesh canvas (QOpenGLWidget + GLSL + free camera).

Mesh format is a 5-tuple:

    positions (N,3), normals (N,3), uvs (N,2), indices (M,3), extra (N,)

`extra` is a single per-vertex float exposed as attribute location 3; apps can
use it for material id, strand id, temperature, etc.  The default vertex
shader is supplied by the app; the base class owns the GL lifecycle, VBO,
depth buffer, MVP/eye uniforms, mouse orbit and zoom.
"""
import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel

from common.free_camera import FreeCamera


class ShaderMeshCanvas(QOpenGLWidget):
    def __init__(self, vertex_src, fragment_src, mesh, camera=None,
                 overlay=None, parent=None, samples=4):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(int(samples))
        self.setFormat(fmt)

        self.vertex_src = vertex_src
        self.fragment_src = fragment_src
        self.mesh = mesh
        self.camera = camera or FreeCamera()
        self._drag = None
        self._program = None
        self._vbo = None
        self._ibo = None
        self._index_count = 0
        self._uniform_mvp = -1
        self._uniform_eye = -1
        self._gl_ready = False
        self.shader_error = None
        self.setCursor(Qt.OpenHandCursor)

        self.overlay = None
        if overlay:
            self.overlay = QLabel(overlay, self)
            self.overlay.setStyleSheet(
                "background: transparent; color: #5b6472; font-size: 9pt;"
            )
            self.overlay.move(16, 12)
            self.overlay.adjustSize()
            self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_mesh(self, mesh):
        self.mesh = mesh
        if self._gl_ready:
            self.makeCurrent()
            self._upload_mesh()
            self.doneCurrent()
        self.update()

    def set_custom_uniforms(self, program):
        """Hook for app-specific uniforms; called while program is bound."""

    def _upload_mesh(self):
        positions, normals, uvs, indices, extra = self.mesh
        extra = np.asarray(extra, dtype=np.float32).reshape(-1, 1)
        data = np.hstack((positions, normals, uvs, extra)).astype(np.float32)
        if self._vbo is None:
            self._vbo = GL.glGenBuffers(1)
            self._ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        idx = np.asarray(indices, dtype=np.uint32).reshape(-1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes,
                        idx, GL.GL_STATIC_DRAW)
        self._index_count = int(idx.size)

    def initializeGL(self):
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_MULTISAMPLE)
        try:
            vs = compileShader(self.vertex_src, GL.GL_VERTEX_SHADER)
            fs = compileShader(self.fragment_src, GL.GL_FRAGMENT_SHADER)
            self._program = compileProgram(vs, fs)
            self._uniform_mvp = GL.glGetUniformLocation(self._program, "u_mvp")
            self._uniform_eye = GL.glGetUniformLocation(self._program, "u_eye")
        except Exception as exc:
            self.shader_error = str(exc)
            self._program = None
        self._upload_mesh()
        self._gl_ready = True

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, int(w * self.devicePixelRatioF()),
                      int(h * self.devicePixelRatioF()))

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._program is None or self._index_count == 0:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        mvp = self.camera.mvp_matrix(w / h)
        eye = self.camera.eye_for_zoom()

        GL.glUseProgram(self._program)
        GL.glUniformMatrix4fv(
            self._uniform_mvp, 1, GL.GL_FALSE,
            np.asarray(mvp.data(), dtype=np.float32).reshape(4, 4),
        )
        GL.glUniform3f(self._uniform_eye,
                       float(eye[0]), float(eye[1]), float(eye[2]))
        self.set_custom_uniforms(self._program)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        stride = 9 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(6 * 4))
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 1, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(8 * 4))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
        GL.glDrawElements(GL.GL_TRIANGLES, self._index_count,
                          GL.GL_UNSIGNED_INT, None)
        for i in range(4):
            GL.glDisableVertexAttribArray(i)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        delta = event.position() - self._drag
        self._drag = event.position()
        self.camera.rotate(delta.x(), delta.y())
        self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        self.camera.zoom = max(
            0.5, min(2.8, self.camera.zoom *
                     (1.12 ** (event.angleDelta().y() / 120.0)))
        )
        self.camera.update()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.camera.reset()
        self.update()
