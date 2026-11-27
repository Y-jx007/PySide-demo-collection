# -*- coding: utf-8 -*-
"""Checks for the GPU Bremsstrahlung fields and colored LIC."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''
import numpy as np
from support.physics.lw_taichi import LIC_N, evaluate_np, lic_np, vector_field_np

raw = evaluate_np(0.3, 0.65, 0.20, 0.70, 0.72, 3.6, False)
raw_c = evaluate_np(0.3, 0.65, 0.20, 0.70, 0.72, 3.6, True)
assert all(np.isfinite(part).all() for part in raw)
assert all(np.isfinite(part).all() for part in raw_c)
assert raw[0].shape == (512, 512)
assert np.max(np.abs(raw_c[2] - raw[2])) > 0.0

vec = vector_field_np(0.3, 0.65, 0.20, 0.70, 0.72, 3.6, False)
vec_c = vector_field_np(0.3, 0.65, 0.20, 0.70, 0.72, 3.6, True)
assert all(np.isfinite(part).all() for part in vec.values())
assert all(np.isfinite(part).all() for part in vec_c.values())

for flags in (
    dict(show_e=False, show_coulomb=False, show_brems=True, show_sync=False, show_total=False),
    dict(show_e=False, show_coulomb=False, show_brems=False, show_sync=True, show_total=False),
    dict(show_e=False, show_coulomb=False, show_brems=False, show_sync=False, show_total=True),
    dict(show_e=False, show_coulomb=True, show_brems=False, show_sync=False, show_total=True),
):
    img = lic_np(0.3, 0.65, 0.20, 0.70, 0.72, 3.6, **flags)
    assert img.shape == (LIC_N, LIC_N, 3)
    assert img.dtype == np.uint8
    assert np.isfinite(img).all()
print("brems fields ok")
'''


def test_fields_and_colored_lic():
    result = subprocess.run(
        [sys.executable, "-c", CODE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "brems fields ok" in result.stdout
