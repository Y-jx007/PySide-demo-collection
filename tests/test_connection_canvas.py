# -*- coding: utf-8 -*-
"""Interaction checks for the Connection canvas.

The dragged trail must extend immediately under the pointer, while the
critically damped spring remains available for click-to-travel and settling.
The ambient field must change length continuously while the point travels.
"""
import importlib.util
import math
import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import apps_manifest  # noqa: E402
from support.math.connection import make_surface  # noqa: E402

_APP = None


def app_path():
    """Follow the manifest, so moving the app does not break the test."""
    for app in apps_manifest.APPS:
        if app["title"] == "Christoffel":
            return os.path.join(ROOT, "apps", app["path"])
    raise AssertionError("Christoffel 未在 apps_manifest.py 注册")


def load_app():
    global _APP
    if _APP is None:
        spec = importlib.util.spec_from_file_location("connection_app", app_path())
        _APP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_APP)
    return _APP


class TestConnectionCanvas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.module = load_app()

    def setUp(self):
        self.window = self.module.MainWindow()
        self.window.show()
        self.app.processEvents()
        self.canvas = self.window.canvas

    def tearDown(self):
        self.canvas._glide.stop()
        self.window.close()

    def distance(self, a, b):
        delta = self.canvas.surface.point(*a) - self.canvas.surface.point(*b)
        return float(np.linalg.norm(delta))

    def test_follow_eases_in_and_lands_exactly(self):
        canvas = self.canvas
        start = canvas.surface.initial()
        canvas._begin_drag(start)
        target = (start[0] + 0.45, start[1] + 0.25)
        canvas._target_uv = target
        span = self.distance(start, target)
        steps = []
        previous = canvas.p_uv
        arrived = False
        for _ in range(240):
            uv, arrived = canvas._follow(1.0 / 60.0)
            canvas._advance(uv)
            steps.append(self.distance(previous, canvas.p_uv))
            previous = canvas.p_uv
            if arrived:
                break
        self.assertTrue(arrived, "the point never reached its target")
        self.assertLess(self.distance(canvas.p_uv, target), 1e-3)
        # the first frame is an ease-in, not a teleport
        self.assertLess(steps[0], 0.5 * span)
        # never faster than the configured world-speed cap
        cap = canvas.FOLLOW_VMAX / 60.0
        self.assertLessEqual(max(steps), cap * 1.02)
        # speed grows and decays instead of starting at full speed
        self.assertLess(steps[0], 0.6 * max(steps))

    def test_drag_to_extends_the_path_immediately(self):
        canvas = self.canvas
        start = canvas.surface.initial()
        target = (start[0] + 0.35, start[1] + 0.18)
        canvas._begin_drag(start)
        canvas._drag_to(target)
        self.assertFalse(canvas._glide.isActive())
        self.assertLess(self.distance(canvas.p_uv, target), 1e-6)
        self.assertLess(self.distance(canvas.path[-1], target), 1e-6)
        self.assertGreater(len(canvas.path), 1)

    def test_long_path_keeps_the_newest_sample_visible(self):
        canvas = self.canvas
        start = canvas.surface.initial()
        # A small closed arc keeps every uv inside the chart. PATH_MAX_POINTS+1
        # points force the projected-path cache into stride-2 sampling; the
        # extra points must still reach the screen instead of freezing.
        limit = canvas.PATH_MAX_POINTS
        arc = [
            (start[0] + 0.2 * math.cos(0.004 * i),
             start[1] + 0.2 * math.sin(0.004 * i))
            for i in range(limit + 1)
        ]
        canvas.path = arc
        canvas._path_cache = None
        canvas._path_screen()
        canvas.path = arc + [
            (start[0] + 0.2 * math.cos(0.004 * i),
             start[1] + 0.2 * math.sin(0.004 * i))
            for i in range(limit + 1, limit + 10)
        ]
        pts = canvas._path_screen()
        last = canvas.camera.project(
            canvas.surface.point(*canvas.path[-1]), canvas.width(), canvas.height()
        )[0]
        self.assertLess(
            math.hypot(pts[-1].x() - last.x(), pts[-1].y() - last.y()), 0.5
        )

    def test_path_tail_is_not_dropped_at_2400_points(self):
        canvas = self.canvas
        start = canvas.surface.initial()
        first = (start[0] + 0.2, start[1] + 0.1)
        canvas.path = [first] + [start] * 2500
        canvas._advance((start[0] + 0.1, start[1] + 0.05))
        self.assertEqual(canvas.path[0], first)
        self.assertGreater(len(canvas.path), 2500)

    def test_follow_is_framerate_independent(self):
        canvas = self.canvas
        start = canvas.surface.initial()
        target = (start[0] - 0.40, start[1] - 0.20)
        ends = []
        for dt in (1.0 / 120.0, 1.0 / 60.0, 1.0 / 30.0):
            canvas._begin_drag(start)
            canvas._target_uv = target
            elapsed = 0.0
            while elapsed < 0.5:
                uv, _ = canvas._follow(dt)
                canvas._advance(uv)
                elapsed += dt
            ends.append(self.distance(canvas.p_uv, target))
        self.assertLess(max(ends), 0.01, ends)

    def test_field_length_varies_continuously(self):
        canvas = self.canvas
        canvas.set_field_amp(55)
        start = canvas.surface.initial()
        canvas._begin_drag(start)
        lengths = [canvas.field_len]
        for k in range(1, 90):
            canvas._advance((start[0] + 0.02 * k, start[1] + 0.012 * k))
            lengths.append(canvas.field_len)
        self.assertGreater(max(lengths) - min(lengths), 0.15)
        self.assertTrue(all(v > 0.0 for v in lengths))
        jumps = [abs(b - a) for a, b in zip(lengths, lengths[1:])]
        self.assertLess(max(jumps), 0.06)

    def test_zero_amplitude_keeps_a_uniform_field(self):
        canvas = self.canvas
        canvas.set_field_amp(0)
        start = canvas.surface.initial()
        canvas._begin_drag(start)
        for k in range(1, 40):
            canvas._advance((start[0] + 0.03 * k, start[1] + 0.02 * k))
            self.assertAlmostEqual(canvas.field_len, 1.0, places=9)

    def test_arrow_length_scales_with_the_envelope(self):
        canvas = self.canvas
        uv = canvas.p_uv
        canvas.set_field_amp(0)
        base = [float(np.linalg.norm(v)) for v in canvas._ambient_vectors(uv)]
        canvas.set_field_amp(55)
        scaled = [float(np.linalg.norm(v)) for v in canvas._ambient_vectors(uv)]
        envelope = canvas._field_scale(uv)
        self.assertNotAlmostEqual(envelope, 1.0, places=3)
        for before, after in zip(base, scaled):
            if before > 1e-6:
                self.assertAlmostEqual(after / before, envelope, places=6)

    def test_unlocked_base_follows_the_press(self):
        canvas = self.canvas
        canvas.set_base_locked(False)
        canvas._start_stroke((0.25, 0.35))
        self.assertLess(self.distance(canvas.base_uv, (0.25, 0.35)), 1e-6)
        self.assertLess(self.distance(canvas.initial_uv, (0.25, 0.35)), 1e-6)

    def test_fast_move_interpolates_along_the_surface(self):
        canvas = self.canvas
        start = canvas.surface.initial()
        canvas._begin_drag(start)
        before = len(canvas.path)
        canvas._advance((start[0] + 0.4, start[1] + 0.25))
        self.assertGreater(len(canvas.path), before + 2)

    def test_locked_base_stays_put(self):
        canvas = self.canvas
        canvas.set_base_locked(True)
        locked = tuple(canvas.base_uv)
        # a press elsewhere keeps the base: the stroke starts there and the
        # point travels to the cursor, drawing the trail as it goes
        canvas._start_stroke((0.25, 0.35))
        self.assertLess(self.distance(canvas.base_uv, locked), 1e-9)
        self.assertLess(self.distance(canvas.initial_uv, locked), 1e-9)
        self.assertLess(self.distance(canvas.p_uv, locked), 1e-6)
        for _ in range(120):
            uv, arrived = canvas._follow(1.0 / 60.0)
            canvas._advance(uv)
            if arrived:
                break
        self.assertLess(self.distance(canvas.p_uv, (0.25, 0.35)), 1e-3)
        self.assertGreater(canvas.path_len, 0.1)
        self.assertLess(self.distance(canvas.base_uv, locked), 1e-9)

    def test_a_press_continues_the_same_trail(self):
        canvas = self.canvas
        canvas.set_base_locked(True)
        canvas._start_stroke(canvas.base_uv)
        for k in range(1, 40):
            canvas._advance((canvas.base_uv[0] + 0.01 * k, canvas.base_uv[1]))
        first = canvas.path[0]
        length = canvas.path_len
        points = len(canvas.path)
        canvas._start_stroke((canvas.base_uv[0] + 0.3, canvas.base_uv[1] + 0.2))
        for _ in range(120):
            uv, arrived = canvas._follow(1.0 / 60.0)
            canvas._advance(uv)
            if arrived:
                break
        self.assertEqual(canvas.path[0], first)
        self.assertGreater(len(canvas.path), points)
        self.assertGreater(canvas.path_len, length)

    def test_back_side_samples_are_marked(self):
        canvas = self.canvas
        base = canvas.base_uv
        eye = canvas.camera.eye / np.linalg.norm(canvas.camera.eye)
        back = -eye
        v_back = math.acos(max(-1.0, min(1.0, float(back[2]))))
        u_back = math.atan2(float(back[1]), float(back[0])) % (2.0 * math.pi)
        canvas._begin_drag(base)
        for k in range(1, 41):
            t = k / 40.0
            canvas._advance(canvas.surface.wrap_uv(
                base[0] + (u_back - base[0]) * t,
                base[1] + (v_back - base[1]) * t,
            ))
        canvas._path_screen()
        visible = canvas._path_vis
        self.assertEqual(len(visible), len(canvas.path))
        self.assertTrue(any(visible))
        self.assertFalse(all(visible))
    def test_pressing_on_the_base_starts_exactly_there(self):
        canvas = self.canvas
        canvas.set_base_locked(True)
        locked = tuple(canvas.base_uv)
        canvas._start_stroke((locked[0] + 0.03, locked[1] - 0.02))
        self.assertLess(self.distance(canvas.initial_uv, locked), 1e-9)
        self.assertLess(self.distance(canvas.base_uv, locked), 1e-9)
        self.assertLess(canvas.path_len, 0.05)

    def test_base_is_locked_by_default(self):
        self.assertTrue(self.canvas.base_locked)
        self.assertTrue(self.window.base_check.isChecked())

    def test_base_checkbox_toggles_the_lock(self):
        check = self.window.base_check
        self.assertTrue(check.isChecked())
        check.setChecked(False)
        self.assertFalse(self.canvas.base_locked)
        check.setChecked(True)
        self.assertTrue(self.canvas.base_locked)

    def test_locked_base_survives_a_mode_switch(self):
        canvas = self.canvas
        canvas.set_base_locked(True)
        locked = tuple(canvas.base_uv)
        self.window.mode_combo.setCurrentText("向量场")
        self.app.processEvents()
        self.assertEqual(canvas.mode, "向量场")
        self.assertLess(self.distance(canvas.base_uv, locked), 1e-9)
        self.assertLess(self.distance(canvas.initial_uv, locked), 1e-9)
        self.window.mode_combo.setCurrentText("平行移动")
        self.app.processEvents()
        self.assertLess(self.distance(canvas.base_uv, locked), 1e-9)
    def test_loop_holonomy_is_live_and_stable(self):
        canvas = self.canvas
        canvas.set_mode("平行移动")
        centre = (0.9, 1.9)
        radius = 0.35
        start = (centre[0] + radius, centre[1])
        canvas._begin_drag(start)
        n = 120
        for k in range(1, n + 1):
            a = 2.0 * math.pi * k / n
            canvas._advance((centre[0] + radius * math.cos(a),
                             centre[1] + radius * math.sin(a)))
        self.assertIsNotNone(canvas.holonomy, "a closed loop must report holonomy")
        angle = canvas.holonomy
        self.assertGreater(abs(angle), 3.0)
        self.assertLess(canvas.closure, canvas.CLOSE_TOL)
        # the residual neck is closed with a short transport: the angle stays put
        canvas._advance((start[0] + 0.045, start[1] - 0.035))
        self.assertIsNotNone(canvas.holonomy)
        self.assertAlmostEqual(canvas.holonomy, angle, delta=0.6)
        # and it survives the release
        canvas.dragging_point = False
        canvas._finish_drag()
        self.assertAlmostEqual(canvas.holonomy, angle, delta=0.6)

    def test_open_loop_has_no_holonomy(self):
        canvas = self.canvas
        canvas.set_mode("平行移动")
        canvas._begin_drag((0.9, 1.9))
        canvas._advance((1.45, 2.45))
        self.assertIsNone(canvas.holonomy)
        self.assertGreater(canvas.closure, canvas.CLOSE_TOL)

    def test_pick_finds_the_point_under_the_cursor(self):
        canvas = self.canvas
        for px, py in ((300, 300), (250, 260), (380, 330), (300, 420), (360, 240)):
            picked = canvas._pick_uv(QPointF(px, py))
            self.assertIsNotNone(picked, (px, py))
            point, _ = canvas.camera.project(
                canvas.surface.point(*picked), canvas.width(), canvas.height()
            )
            self.assertLess(math.hypot(point.x() - px, point.y() - py), 0.75)

    def test_pick_ignores_the_empty_canvas(self):
        canvas = self.canvas
        self.assertIsNone(canvas._pick_uv(QPointF(5, 5)))
        self.assertIsNone(canvas._pick_uv(QPointF(595, 595)))

    def test_hinted_pick_matches_the_global_pick(self):
        surface = self.canvas.surface
        camera = self.canvas.camera
        width = self.canvas.width()
        height = self.canvas.height()
        hint = None
        for k in range(60):
            x = 300 + 120 * math.cos(0.05 * k)
            y = 300 + 120 * math.sin(0.05 * k)
            origin, direction = camera.ray(x, y, width, height)
            global_uv = surface.nearest_uv_to_ray(origin, direction, samples=30)
            hinted = surface.nearest_uv_to_ray(
                origin, direction, samples=30, hint=hint, tolerance=0.02
            )
            self.assertIsNotNone(hinted)
            self.assertLess(self.distance(global_uv, hinted), 0.02, (x, y))
            hint = hinted


if __name__ == "__main__":
    unittest.main()
