# -*- coding: utf-8 -*-
"""Free-orbit quaternion camera shared by GL/3D apps.

The orientation is a unit quaternion; target, distance and zoom remain
ordinary camera state.  There is deliberately no pitch clamp and no world-up
singularity, so dragging can tumble through any orientation.
"""
import math

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QMatrix4x4, QQuaternion, QVector3D


class FreeCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), distance=6.0,
                 yaw=-50.0, pitch=18.0):
        self.target = np.asarray(target, dtype=float)
        self.distance = float(distance)
        self.zoom = 1.0
        self._yaw = float(yaw)
        self._pitch = float(pitch)
        self.q = self._initial_q()
        self.update()

    def update(self):
        # One matrix multiplication per basis vector keeps the quaternion
        # conversion explicit and readable.
        f = self.q.rotatedVector(QVector3D(0.0, 0.0, -1.0))
        u = self.q.rotatedVector(QVector3D(0.0, 1.0, 0.0))
        r = self.q.rotatedVector(QVector3D(1.0, 0.0, 0.0))
        self.forward = np.array([f.x(), f.y(), f.z()])
        self.up = np.array([u.x(), u.y(), u.z()])
        self.right = np.array([r.x(), r.y(), r.z()])
        self.eye = self.target - self.forward * self.distance

    def rotate(self, dx, dy, sensitivity=0.55):
        """Rotate around the current screen axes (pixels -> degrees)."""
        qy = QQuaternion.fromAxisAndAngle(
            QVector3D(*self.up), float(dx) * sensitivity
        )
        qx = QQuaternion.fromAxisAndAngle(
            QVector3D(*self.right), float(dy) * sensitivity
        )
        self.q = qy * qx * self.q
        self.q.normalize()
        self.update()

    def reset(self):
        self.zoom = 1.0
        self.q = self._initial_q()
        self.update()

    def set_angles(self, yaw, pitch):
        """Jump to a yaw/pitch view (used when the scene switches to a plane)."""
        self._yaw = float(yaw)
        self._pitch = float(pitch)
        self.q = self._initial_q()
        self.update()

    def _initial_q(self):
        """Match the old Camera3D yaw/pitch convention exactly."""
        y = math.radians(self._yaw)
        p = math.radians(self._pitch)
        eye_dir = np.array([
            math.cos(p) * math.cos(y),
            math.sin(p),
            math.cos(p) * math.sin(y),
        ], dtype=float)
        forward = -eye_dir
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(right)
        if n < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= n
        up = np.cross(right, forward)
        back = -forward
        m = np.column_stack((right, up, back))
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            yq = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            yq = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            yq = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            yq = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = QQuaternion(w, x, yq, z)
        q.normalize()
        return q

    def orientation_key(self, digits=3):
        """Stable cache key for the current view orientation."""
        return tuple(round(v, int(digits)) for v in (
            self.q.scalar(), self.q.x(), self.q.y(), self.q.z()
        ))

    def world_per_pixel(self, width, height):
        return 1.0 / (min(width, height) * 0.38 * max(1e-6, self.zoom))

    def project(self, point, width, height):
        """Camera3D-compatible orthographic projection: (QPointF, depth)."""
        rel = np.asarray(point, dtype=float) - self.eye
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

    def eye_for_zoom(self):
        return self.target - self.forward * (
            self.distance / max(0.2, float(self.zoom))
        )

    def view_matrix(self):
        m = QMatrix4x4()
        m.lookAt(
            QVector3D(*self.eye_for_zoom()),
            QVector3D(*self.target),
            QVector3D(*self.up),
        )
        return m

    def projection_matrix(self, aspect, fov=42.0, near=0.05, far=100.0):
        m = QMatrix4x4()
        m.perspective(fov, max(1e-6, float(aspect)), near, far)
        return m

    def mvp_matrix(self, aspect):
        return self.projection_matrix(aspect) * self.view_matrix()
