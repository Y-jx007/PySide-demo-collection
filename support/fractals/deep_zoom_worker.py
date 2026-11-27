"""Subprocess worker for :mod:`support.fractals.deep_zoom`.

Run as ``python -m support.fractals.deep_zoom_worker``.  Keeping this in a
separate process avoids importing the Qt/Taichi application in every
worker (the Windows ``spawn`` start method would otherwise re-import the
app module).
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    from support.fractals.deep_zoom import _render_rows

    args = (
        payload["cx"], payload["cy"], payload["scale"],
        int(payload["size"]), int(payload["max_iter"]),
        int(payload["precision"]), int(payload["y0"]), int(payload["y1"]),
    )
    y0, y1, data = _render_rows(args)
    assert y0 == args[-2] and y1 == args[-1]
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
