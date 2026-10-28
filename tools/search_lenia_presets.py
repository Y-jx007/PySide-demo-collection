"""Search stable Lenia creatures and append them to the preset gallery.

Default mode samples **random parameters** and draws a new initial state with a
simulated round brush (dots, strokes, arcs, random walks, crosses), then lets it
evolve and scores the result.  It does not start from an existing preset.

Usage (from the project root):

    python tools/search_lenia_presets.py --trials 300 --save
    python tools/search_lenia_presets.py --trials 300 --dry-run
    python tools/search_lenia_presets.py --mode mutate --trials 200 --dry-run

The script uses the same ``LeniaSimulation`` as the app, so saved presets load
without any format changes.  A backup of the existing ``.npz`` is created
before writing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
APPS = ROOT / "apps"
sys.path.insert(0, str(APPS))
sys.path.insert(0, str(APPS / "Cellular Automata"))

import Lenia  # noqa: E402  (needs sys.path setup above)

SIZE = 256
DEFAULT_PRESET = ROOT / "assets" / "data" / "lenia_presets.npz"


# ---------------------------------------------------------------- presets

def load_presets(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    names = sorted({k.rsplit("_", 1)[0] for k in data.keys() if k.endswith("_state")})
    out = {}
    for name in names:
        state_key = f"{name}_state"
        param_key = f"{name}_params"
        if state_key in data and param_key in data:
            out[name] = {
                "state": np.asarray(data[state_key], dtype=np.float32),
                "params": np.asarray(data[param_key], dtype=np.float32),
            }
    return out


def save_presets(path: Path, presets: dict) -> Path | None:
    backup = None
    if path.exists():
        backup_dir = path.parent / "preset_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / (
            f"{path.stem}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}{path.suffix}"
        )
        shutil.copy2(path, backup)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    payload = {}
    for name, item in presets.items():
        payload[f"{name}_state"] = np.asarray(item["state"], dtype=np.float32)
        payload[f"{name}_params"] = np.asarray(item["params"], dtype=np.float32)
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return backup


# ---------------------------------------------------------------- metrics

def centroid(arr: np.ndarray, threshold: float = 0.05) -> tuple[float, float]:
    ys, xs = np.nonzero(arr > threshold)
    if len(ys) == 0:
        return SIZE / 2.0, SIZE / 2.0
    ang_y = 2 * np.pi * ys / SIZE
    ang_x = 2 * np.pi * xs / SIZE
    cy = (np.arctan2(np.sin(ang_y).sum(), np.cos(ang_y).sum()) / (2 * np.pi) % 1.0) * SIZE
    cx = (np.arctan2(np.sin(ang_x).sum(), np.cos(ang_x).sum()) / (2 * np.pi) % 1.0) * SIZE
    return float(cy), float(cx)


def center_state(arr: np.ndarray) -> np.ndarray:
    cy, cx = centroid(arr)
    return np.roll(arr, (int(round(SIZE / 2 - cy)), int(round(SIZE / 2 - cx))), axis=(0, 1))


def torus_delta(a: float, b: float) -> float:
    return (a - b + SIZE / 2) % SIZE - SIZE / 2


def state_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = center_state(a)
    b = center_state(b)
    den = float(np.mean(a * a) + np.mean(b * b)) + 1e-8
    return float(np.mean((a - b) ** 2) / den)


def evaluate(sim, state: np.ndarray, params: np.ndarray, burn: int, observe: int,
             every: int) -> dict | None:
    R, rings, rho, omega = params[:4]
    betas = params[4:10]
    mu, sigma, dt = params[10:13]
    sim.update_kernel(int(R), int(rings), float(rho), float(omega), [float(b) for b in betas])
    sim.set_state_numpy(np.asarray(state, dtype=np.float32))
    for _ in range(burn):
        sim.step(float(mu), float(sigma), float(dt))

    frames, masses, areas, cents = [], [], [], []
    for i in range(observe):
        sim.step(float(mu), float(sigma), float(dt))
        if i % every:
            continue
        raw = sim.get_state_numpy().copy()
        frames.append(center_state(raw))
        masses.append(float(raw.sum()))
        areas.append(float((raw > 0.05).mean()))
        cents.append(centroid(raw))

    if len(frames) < 4:
        return None
    final = frames[-1]
    active = final > 0.05
    ys, xs = np.nonzero(active)
    if len(ys) == 0:
        return None

    bbox = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    masses_arr = np.asarray(masses)
    area = float(np.mean(areas))
    mass_cv = float(masses_arr.std() / (masses_arr.mean() + 1e-8))

    shape_err = 1.0
    best_lag = 0
    for lag in range(1, min(30, len(frames) - 1) + 1):
        a, b = frames[-1], frames[-1 - lag]
        den = float(np.mean(a * a) + np.mean(b * b)) + 1e-8
        err = float(np.mean((a - b) ** 2) / den)
        if err < shape_err:
            shape_err = err
            best_lag = lag

    speed = float(np.hypot(torus_delta(cents[-1][0], cents[0][0]),
                           torus_delta(cents[-1][1], cents[0][1])) / max(1, observe))
    return {
        "state": final,
        "mass": float(masses_arr.mean()),
        "mass_cv": mass_cv,
        "area": area,
        "bbox": bbox,
        "err": float(shape_err),
        "period": best_lag * every,
        "speed": speed,
    }


def is_stable(m: dict) -> bool:
    if m["mass"] < 80 or m["mass"] > 12000:
        return False
    if m["area"] < 0.002 or m["area"] > 0.18:
        return False
    if max(m["bbox"]) > 190:
        return False
    if m["mass_cv"] > 0.12:
        return False
    if m["err"] > 0.12:
        return False
    return True


def score(m: dict) -> float:
    locality = 1.0 - min(1.0, max(m["bbox"]) / 190.0)
    activity = 1.0 - min(1.0, abs(m["area"] - 0.04) / 0.08)
    return (
        0.45 * (1.0 - min(1.0, m["err"]))
        + 0.25 * (1.0 - min(1.0, m["mass_cv"] / 0.12))
        + 0.20 * locality
        + 0.10 * max(0.0, activity)
    )


def classify(m: dict) -> str:
    if m["speed"] * 100 >= 1.5:
        return "滑翔"
    if m["mass_cv"] > 0.02 or m["err"] > 0.01:
        return "振子"
    return "驻波"


# ---------------------------------------------------------------- candidates

def clamp_params(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float32).copy()
    p[0] = np.clip(round(float(p[0])), 5, 50)
    p[1] = np.clip(round(float(p[1])), 1, 6)
    p[2] = np.clip(float(p[2]), 0.05, 0.95)
    p[3] = np.clip(float(p[3]), 0.02, 0.50)
    p[4:10] = np.clip(p[4:10], 0.0, 1.50)
    p[10] = np.clip(float(p[10]), 0.05, 0.40)
    p[11] = np.clip(float(p[11]), 0.005, 0.080)
    p[12] = np.clip(float(p[12]), 0.03, 0.50)
    p[13] = SIZE
    return p


def mutate_params(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = np.asarray(base, dtype=np.float32).copy()
    p[0] += rng.integers(-3, 4)
    if rng.random() < 0.25:
        p[1] += rng.integers(-1, 2)
    p[2] += rng.normal(0, 0.035)
    p[3] += rng.normal(0, 0.012)
    p[4:10] += rng.normal(0, 0.08, 6)
    p[10] += rng.normal(0, 0.012)
    p[11] *= float(np.exp(rng.normal(0, 0.12)))
    p[12] += rng.normal(0, 0.012)
    return clamp_params(p)


def random_params(rng: np.random.Generator) -> np.ndarray:
    """Sample the whole plausible parameter box, not a neighbour of a preset."""
    p = np.array([
        rng.integers(6, 51),
        rng.integers(1, 7),
        rng.uniform(0.10, 0.90),
        rng.uniform(0.03, 0.40),
        *rng.uniform(0.0, 1.30, 6),
        rng.uniform(0.08, 0.32),
        rng.uniform(0.008, 0.060),
        rng.uniform(0.05, 0.40),
        SIZE,
    ], dtype=np.float32)
    return clamp_params(p)


def _stamp_disc(arr: np.ndarray, cy: int, cx: int, radius: int, value: float) -> None:
    y0, y1 = max(0, cy - radius), min(SIZE, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(SIZE, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    sub = arr[y0:y1, x0:x1]
    np.maximum(sub, value, out=sub, where=mask)


def _stroke(arr: np.ndarray, p0: tuple[int, int], p1: tuple[int, int],
            radius: int, value: float) -> None:
    y0, x0 = p0
    y1, x1 = p1
    steps = max(2, int(np.hypot(y1 - y0, x1 - x0)))
    for t in np.linspace(0.0, 1.0, steps):
        _stamp_disc(arr, int(round(y0 + (y1 - y0) * t)),
                    int(round(x0 + (x1 - x0) * t)), radius, value)


def brush_state(rng: np.random.Generator) -> np.ndarray:
    """Simulate a user drawing with the app's round brush.

    Randomly combines dots, straight strokes, arcs, random walks and crosses
    on an empty 256x256 canvas.  No preset state is used.
    """
    arr = np.zeros((SIZE, SIZE), dtype=np.float32)
    margin = 36
    lo, hi = margin, SIZE - margin

    for _ in range(int(rng.integers(1, 5))):
        kind = int(rng.integers(0, 6))
        radius = int(rng.integers(2, 11))
        value = float(rng.uniform(0.65, 1.0))

        if kind == 0:  # dots
            for _ in range(int(rng.integers(1, 4))):
                cy, cx = rng.integers(lo, hi, 2)
                _stamp_disc(arr, int(cy), int(cx), radius, value)

        elif kind == 1:  # straight stroke
            y0, x0 = rng.integers(lo, hi, 2)
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(10, 80)
            _stroke(arr, (int(y0), int(x0)),
                    (int(y0 + length * np.sin(angle)), int(x0 + length * np.cos(angle))),
                    radius, value)

        elif kind == 2:  # arc / circle
            cy, cx = rng.integers(lo + 12, hi - 12, 2)
            rr = float(rng.uniform(10, min(42, hi - max(cy, cx))))
            start = rng.uniform(0, 2 * np.pi)
            span = rng.uniform(np.pi / 2, 2 * np.pi)
            steps = max(10, int(rr * span))
            for t in np.linspace(0.0, 1.0, steps):
                a = start + span * t
                _stamp_disc(arr, int(round(cy + rr * np.sin(a))),
                            int(round(cx + rr * np.cos(a))), radius, value)

        elif kind == 3:  # random walk
            y, x = rng.integers(lo, hi, 2).astype(float)
            angle = rng.uniform(0, 2 * np.pi)
            for _ in range(int(rng.integers(12, 40))):
                angle += rng.normal(0, 0.6)
                length = float(rng.uniform(4, 12))
                y = float(np.clip(y + length * np.sin(angle), lo, hi - 1))
                x = float(np.clip(x + length * np.cos(angle), lo, hi - 1))
                _stamp_disc(arr, int(round(y)), int(round(x)), radius, value)

        elif kind == 4:  # cross / star
            cy, cx = rng.integers(lo + 12, hi - 12, 2)
            base = rng.uniform(0, np.pi)
            for k in range(int(rng.integers(2, 4))):
                a = base + k * np.pi / 2
                half = rng.uniform(12, 50)
                _stroke(arr, (int(cy - half * np.sin(a)), int(cx - half * np.cos(a))),
                        (int(cy + half * np.sin(a)), int(cx + half * np.cos(a))),
                        radius, value)

        else:  # several short strokes
            for _ in range(int(rng.integers(2, 5))):
                y0, x0 = rng.integers(lo, hi, 2)
                a = rng.uniform(0, 2 * np.pi)
                length = rng.uniform(8, 30)
                _stroke(arr, (int(y0), int(x0)),
                        (int(y0 + length * np.sin(a)), int(x0 + length * np.cos(a))),
                        radius, value)

    if float(arr.sum()) < 30:
        _stamp_disc(arr, SIZE // 2, SIZE // 2, 8, 1.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def blob_state(rng: np.random.Generator) -> np.ndarray:
    arr = np.zeros((SIZE, SIZE), dtype=np.float32)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for _ in range(int(rng.integers(1, 3))):
        cy = int(rng.integers(56, SIZE - 56))
        cx = int(rng.integers(56, SIZE - 56))
        radius = float(rng.uniform(5.0, 14.0))
        amp = float(rng.uniform(0.55, 1.0))
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        arr = np.maximum(arr, amp * np.exp(-d2 / (2.0 * radius * radius)).astype(np.float32))
    return arr


def perturb_state(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    state = np.roll(base, (int(rng.integers(-8, 9)), int(rng.integers(-8, 9))), axis=(0, 1))
    if rng.random() < 0.5:
        state = state + rng.normal(0, 0.008, state.shape).astype(np.float32)
    return np.clip(state, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--mode", choices=("random", "mutate", "mixed"), default="random",
                        help="random: random params + brush drawing (default); "
                             "mutate: small changes around existing presets; "
                             "mixed: 50/50")
    parser.add_argument("--burn", type=int, default=150)
    parser.add_argument("--observe", type=int, default=200)
    parser.add_argument("--validate-steps", type=int, default=1200,
                        help="long-run recheck before saving; 0 disables")
    parser.add_argument("--min-distance", type=float, default=0.02,
                        help="reject candidates too similar to existing/found states")
    parser.add_argument("--preset-file", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--preview-dir", type=Path, default=None,
                        help="optionally write one PNG per selected candidate")
    parser.add_argument("--save", action="store_true", help="write accepted presets to the npz")
    parser.add_argument("--dry-run", action="store_true", help="do not write anything")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    presets = load_presets(args.preset_file)
    base_items = list(presets.values())
    print(f"loaded {len(presets)} presets from {args.preset_file}", flush=True)

    sim = Lenia.LeniaSimulation(SIZE)
    found = []
    seen_states = [item["state"] for item in base_items]
    t0 = time.perf_counter()

    for trial in range(1, args.trials + 1):
        use_base = bool(base_items) and (
            args.mode == "mutate" or (args.mode == "mixed" and rng.random() < 0.5)
        )
        if use_base:
            base = base_items[int(rng.integers(0, len(base_items)))]
            params = mutate_params(base["params"], rng)
            state = perturb_state(base["state"], rng)
        else:
            # explore new parameter regions and new drawn shapes, not preset neighbours
            params = random_params(rng)
            state = brush_state(rng)

        m = evaluate(sim, state, params, args.burn, args.observe, 2)
        if m is None or not is_stable(m):
            if trial % 25 == 0:
                print(f"  {trial}/{args.trials} ...", flush=True)
            continue

        # reject duplicates of existing and already-found states
        if any(state_distance(m["state"], s) < args.min_distance for s in seen_states):
            if trial % 25 == 0:
                print(f"  {trial}/{args.trials} ...", flush=True)
            continue

        m["params"] = params
        m["score"] = score(m)
        m["kind"] = classify(m)
        found.append(m)
        seen_states.append(m["state"])
        print(
            f"  found #{len(found):02d} {m['kind']} score={m['score']:.3f} "
            f"mass={m['mass']:.0f} area={m['area']:.3f} bbox={m['bbox']} "
            f"cv={m['mass_cv']:.3f} err={m['err']:.3f} speed={m['speed']:.4f}",
            flush=True,
        )
        if trial % 25 == 0:
            print(f"  {trial}/{args.trials} ...", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"done: {len(found)} candidates in {elapsed:.1f}s", flush=True)

    # pick a diverse top set: at most 3 per kind, then fill by score
    found.sort(key=lambda x: x["score"], reverse=True)
    selected, counts = [], {}
    for item in found:
        if len(selected) >= args.top:
            break
        kind = item["kind"]
        if counts.get(kind, 0) >= 3:
            continue
        selected.append(item)
        counts[kind] = counts.get(kind, 0) + 1
    if len(selected) < args.top:
        chosen = {id(item) for item in selected}
        for item in found:
            if len(selected) >= args.top:
                break
            if id(item) in chosen:
                continue
            selected.append(item)
            chosen.add(id(item))

    if not selected:
        print("no stable candidates; nothing saved", flush=True)
        return

    print("\nselected:")
    for item in selected:
        p = item["params"]
        print(
            f"  {item['kind']} score={item['score']:.3f} R={int(p[0])} rings={int(p[1])} "
            f"rho={p[2]:.2f} omega={p[3]:.2f} mu={p[10]:.3f} sigma={p[11]:.4f} dt={p[12]:.2f}"
        )

    if args.validate_steps > 0:
        validated = []
        print(f"\nlong-run validation ({args.validate_steps} steps):", flush=True)
        for item in selected:
            vm = evaluate(sim, item["state"], item["params"], burn=100,
                          observe=args.validate_steps, every=5)
            if vm is not None and is_stable(vm):
                item["validation"] = vm
                validated.append(item)
                print(
                    f"  keep {item['kind']}: mass={vm['mass']:.0f} area={vm['area']:.3f} "
                    f"bbox={vm['bbox']} cv={vm['mass_cv']:.3f} err={vm['err']:.3f} "
                    f"speed={vm['speed']:.4f}",
                    flush=True,
                )
            else:
                print(f"  drop {item['kind']}: failed long-run stability", flush=True)
        selected = validated
        if not selected:
            print("no candidates survived long-run validation; nothing saved", flush=True)
            return

    if args.preview_dir and selected:
        import matplotlib.pyplot as plt

        args.preview_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(selected, 1):
            path = args.preview_dir / f"{index:02d}_{item['kind']}.png"
            plt.imsave(path, center_state(item["state"]), cmap="cubehelix", vmin=0.0, vmax=1.0)
        print(f"previews: {args.preview_dir}", flush=True)

    if args.dry_run or not args.save:
        print("\n(dry run: preset file not modified; add --save to write)")
        return

    used = set(presets.keys())
    prefix = "自动" if args.mode == "mutate" else "随机"
    for index, item in enumerate(selected, 1):
        kind = item["kind"]
        name = f"{prefix}{item['params'][1]:.0f}环{kind}{index:02d}"
        while name in used:
            name += "x"
        used.add(name)
        presets[name] = {"state": center_state(item["state"]).astype(np.float32),
                         "params": np.asarray(item["params"], dtype=np.float32)}
        print(f"  + {name}")

    backup = save_presets(args.preset_file, presets)
    print(f"\nsaved {len(selected)} presets to {args.preset_file}", flush=True)
    if backup:
        print(f"backup: {backup}", flush=True)


if __name__ == "__main__":
    main()
