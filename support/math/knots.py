# -*- coding: utf-8 -*-
"""Knot catalogue and programmatic construction.

Naming convention
-----------------
* canonical key: Rolfsen label ``n_k`` (``5_2``), displayed with subscripts.
* aliases: English / Chinese common names and the torus notation ``T(p, q)``.
* nothing is stored per knot except the braid word and closed-form invariants:
  the 3D closed braid and the Jones polynomial are *generated*.

Pipeline
--------
name -> (braid word, strand count)
     -> closed-braid curve (rendered as a hemp tube)
     -> Temperley-Lieb state sum -> Jones polynomial V(t)
     -> determinant |V(-1)|, Alexander from the catalogue or the torus formula
"""
from fractions import Fraction
import math
import re
import unicodedata

# --------------------------------------------------------------------------
# Catalogue: rolefsen key -> braid word (signed generators), strand count,
# crossing number, Conway notation, Alexander coefficients {exponent: coef}.
# Braid words come from the KnotInfo database (Gittings' minimal braids).
# --------------------------------------------------------------------------
KNOT_TABLE = {
    "0_1": dict(pretty="0₁", braid=(), strands=1, crossings=0,
                 conway="", alex={0: 1},
                 aliases=('unknot', 'trivial', '平凡结', '无结')),
    "3_1": dict(pretty="3₁", braid=(1, 1, 1), strands=2, crossings=3,
                 conway="[3]", alex={0: 1, 1: -1, 2: 1},
                 aliases=('trefoil', '三叶结', '三叶', '三叶线')),
    "4_1": dict(pretty="4₁", braid=(1, -2, 1, -2), strands=3, crossings=4,
                 conway="[22]", alex={0: 1, 1: -3, 2: 1},
                 aliases=('figure-eight', 'figure eight', 'figure8', '八字结', '8字结')),
    "5_1": dict(pretty="5₁", braid=(1, 1, 1, 1, 1), strands=2, crossings=5,
                 conway="[5]", alex={0: 1, 1: -1, 2: 1, 3: -1, 4: 1},
                 aliases=('cinquefoil', '五叶结', '五叶')),
    "5_2": dict(pretty="5₂", braid=(1, 1, 1, 2, -1, 2), strands=3, crossings=5,
                 conway="[32]", alex={0: 2, 1: -3, 2: 2},
                 aliases=('three-twist', '3-twist', '三扭结')),
    "6_1": dict(pretty="6₁", braid=(1, 1, 2, -1, -3, 2, -3), strands=4, crossings=6,
                 conway="[42]", alex={0: 2, 1: -5, 2: 2},
                 aliases=('six-one',)),
    "6_2": dict(pretty="6₂", braid=(1, 1, 1, -2, 1, -2), strands=3, crossings=6,
                 conway="[312]", alex={0: 1, 1: -3, 2: 3, 3: -3, 4: 1},
                 aliases=()),
    "6_3": dict(pretty="6₃", braid=(1, 1, -2, 1, -2, -2), strands=3, crossings=6,
                 conway="[2112]", alex={0: 1, 1: -3, 2: 5, 3: -3, 4: 1},
                 aliases=()),
    "7_1": dict(pretty="7₁", braid=(1, 1, 1, 1, 1, 1, 1), strands=2, crossings=7,
                 conway="[7]", alex={0: 1, 1: -1, 2: 1, 3: -1, 4: 1, 5: -1, 6: 1},
                 aliases=('seven-one',)),
    "7_2": dict(pretty="7₂", braid=(1, 1, 1, 2, -1, 2, 3, -2, 3), strands=4, crossings=7,
                 conway="[52]", alex={0: 3, 1: -5, 2: 3},
                 aliases=()),
    "7_3": dict(pretty="7₃", braid=(1, 1, 1, 1, 1, 2, -1, 2), strands=3, crossings=7,
                 conway="[43]", alex={0: 2, 1: -3, 2: 3, 3: -3, 4: 2},
                 aliases=()),
    "7_4": dict(pretty="7₄", braid=(1, 1, 2, -1, 2, 2, 3, -2, 3), strands=4, crossings=7,
                 conway="[313]", alex={0: 4, 1: -7, 2: 4},
                 aliases=()),
    "7_5": dict(pretty="7₅", braid=(1, 1, 1, 1, 2, -1, 2, 2), strands=3, crossings=7,
                 conway="[322]", alex={0: 2, 1: -4, 2: 5, 3: -4, 4: 2},
                 aliases=()),
    "7_6": dict(pretty="7₆", braid=(1, 1, -2, 1, 3, -2, 3), strands=4, crossings=7,
                 conway="[2212]", alex={0: 1, 1: -5, 2: 7, 3: -5, 4: 1},
                 aliases=()),
    "7_7": dict(pretty="7₇", braid=(-1, 2, -1, 2, -3, 2, -3), strands=4, crossings=7,
                 conway="[21112]", alex={0: 1, 1: -5, 2: 9, 3: -5, 4: 1},
                 aliases=()),
    "8_1": dict(pretty="8₁", braid=(1, 1, 2, -1, 2, 3, -2, -4, 3, -4), strands=5, crossings=8,
                 conway="[62]", alex={0: 3, 1: -7, 2: 3},
                 aliases=()),
    "8_2": dict(pretty="8₂", braid=(1, 1, 1, 1, 1, -2, 1, -2), strands=3, crossings=8,
                 conway="[512]", alex={0: 1, 1: -3, 2: 3, 3: -3, 4: 3, 5: -3, 6: 1},
                 aliases=()),
    "8_3": dict(pretty="8₃", braid=(1, 1, 2, -1, -3, 2, -3, -4, 3, -4), strands=5, crossings=8,
                 conway="[44]", alex={0: 4, 1: -9, 2: 4},
                 aliases=()),
    "8_4": dict(pretty="8₄", braid=(-1, -1, -1, 2, -1, 2, 3, -2, 3), strands=4, crossings=8,
                 conway="[413]", alex={0: 2, 1: -5, 2: 5, 3: -5, 4: 2},
                 aliases=()),
    "8_5": dict(pretty="8₅", braid=(1, 1, 1, -2, 1, 1, 1, -2), strands=3, crossings=8,
                 conway="[3;3;2]", alex={0: 1, 1: -3, 2: 4, 3: -5, 4: 4, 5: -3, 6: 1},
                 aliases=()),
    "8_6": dict(pretty="8₆", braid=(1, 1, 1, 1, 2, -1, -3, 2, -3), strands=4, crossings=8,
                 conway="[332]", alex={0: 2, 1: -6, 2: 7, 3: -6, 4: 2},
                 aliases=()),
    "8_7": dict(pretty="8₇", braid=(-1, -1, -1, -1, 2, -1, 2, 2), strands=3, crossings=8,
                 conway="[4112]", alex={0: 1, 1: -3, 2: 5, 3: -5, 4: 5, 5: -3, 6: 1},
                 aliases=()),
    "8_8": dict(pretty="8₈", braid=(-1, -1, -1, -2, 1, 3, -2, 3, 3), strands=4, crossings=8,
                 conway="[2312]", alex={0: 2, 1: -6, 2: 9, 3: -6, 4: 2},
                 aliases=()),
    "8_9": dict(pretty="8₉", braid=(1, 1, 1, -2, 1, -2, -2, -2), strands=3, crossings=8,
                 conway="[3113]", alex={0: 1, 1: -3, 2: 5, 3: -7, 4: 5, 5: -3, 6: 1},
                 aliases=()),
    "8_10": dict(pretty="8₁₀", braid=(-1, -1, -1, 2, -1, -1, 2, 2), strands=3, crossings=8,
                 conway="[3;21;2]", alex={0: 1, 1: -3, 2: 6, 3: -7, 4: 6, 5: -3, 6: 1},
                 aliases=()),
    "8_11": dict(pretty="8₁₁", braid=(1, 1, 2, -1, 2, 2, -3, 2, -3), strands=4, crossings=8,
                 conway="[3212]", alex={0: 2, 1: -7, 2: 9, 3: -7, 4: 2},
                 aliases=()),
    "8_12": dict(pretty="8₁₂", braid=(1, -2, 1, 3, -2, -4, 3, -4), strands=5, crossings=8,
                 conway="[2222]", alex={0: 1, 1: -7, 2: 13, 3: -7, 4: 1},
                 aliases=()),
    "8_13": dict(pretty="8₁₃", braid=(1, 1, -2, 1, -2, -2, -3, 2, -3), strands=4, crossings=8,
                 conway="[31112]", alex={0: 2, 1: -7, 2: 11, 3: -7, 4: 2},
                 aliases=()),
    "8_14": dict(pretty="8₁₄", braid=(1, 1, 1, 2, -1, 2, -3, 2, -3), strands=4, crossings=8,
                 conway="[22112]", alex={0: 2, 1: -8, 2: 11, 3: -8, 4: 2},
                 aliases=()),
    "8_15": dict(pretty="8₁₅", braid=(1, 1, -2, 1, 3, 2, 2, 2, 3), strands=4, crossings=8,
                 conway="[21;21;2]", alex={0: 3, 1: -8, 2: 11, 3: -8, 4: 3},
                 aliases=()),
    "8_16": dict(pretty="8₁₆", braid=(-1, -1, 2, -1, -1, 2, -1, 2), strands=3, crossings=8,
                 conway="[.2.20]", alex={0: 1, 1: -4, 2: 8, 3: -9, 4: 8, 5: -4, 6: 1},
                 aliases=()),
    "8_17": dict(pretty="8₁₇", braid=(1, 1, -2, 1, -2, 1, -2, -2), strands=3, crossings=8,
                 conway="[.2.2]", alex={0: 1, 1: -4, 2: 8, 3: -11, 4: 8, 5: -4, 6: 1},
                 aliases=()),
    "8_18": dict(pretty="8₁₈", braid=(1, -2, 1, -2, 1, -2, 1, -2), strands=3, crossings=8,
                 conway="[8*]", alex={0: 1, 1: -5, 2: 10, 3: -13, 4: 10, 5: -5, 6: 1},
                 aliases=()),
    "8_19": dict(pretty="8₁₉", braid=(1, 1, 1, 2, 1, 1, 1, 2), strands=3, crossings=8,
                 conway="[3;3;2-]", alex={0: 1, 1: -1, 3: 1, 5: -1, 6: 1},
                 aliases=('torus34',)),
    "8_20": dict(pretty="8₂₀", braid=(1, 1, 1, -2, -1, -1, -1, -2), strands=3, crossings=8,
                 conway="[3;21;2-]", alex={0: 1, 1: -2, 2: 3, 3: -2, 4: 1},
                 aliases=()),
    "8_21": dict(pretty="8₂₁", braid=(1, 1, 1, 2, -1, -1, 2, 2), strands=3, crossings=8,
                 conway="[21;21;2-]", alex={0: 1, 1: -4, 2: 5, 3: -4, 4: 1},
                 aliases=()),
}

TORUS_NAMES = {(2, 3): "3_1", (2, 5): "5_1", (2, 7): "7_1", (3, 4): "8_19"}
# Torus knots outside the 8-crossing catalogue: label -> (p, q).
EXTRA_TORUS = {
    "9_1": (2, 9),
    "11a367": (2, 11),
    "10_124": (3, 5),
    "13a5190": (2, 13),
}
DEFAULT_KNOT = "3_1"

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _pretty_label(key):
    if "_" not in key:
        return key
    head, tail = key.split("_", 1)
    return head + tail.translate(_SUBSCRIPT)


class KnotError(ValueError):
    """Raised when a typed name cannot be resolved."""


# --------------------------------------------------------------- name lookup
def _clean(text):
    """NFKC (subscripts -> digits), drop the 结/knot suffix, lower-case."""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    for suffix in ("扭结", "纽结", "结", "knot"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s.lower()


def _catalogue_spec(key):
    entry = KNOT_TABLE[key]
    return dict(key=key, pretty=entry["pretty"], braid=tuple(entry["braid"]),
                strands=int(entry["strands"]), crossings=int(entry["crossings"]),
                alex=dict(entry["alex"]), conway=entry["conway"],
                source="catalogue")


def _torus(p, q, label=None):
    if p < 2 or q < 2:
        raise KnotError("环面结要求 p, q ≥ 2")
    if p > q:
        p, q = q, p
    g = math.gcd(p, q)
    if g != 1:
        raise KnotError(f"T({p},{q}) 是 {g} 分支链环，不是纽结")
    word = tuple(i for _ in range(q) for i in range(1, p))
    key = TORUS_NAMES.get((p, q))
    if key:
        spec = _catalogue_spec(key)
        spec["source"] = "torus"
        return spec
    name = label
    if name is None:
        for extra_label, pair in EXTRA_TORUS.items():
            if pair == (p, q):
                name = extra_label
                break
    name = name or f"T({p},{q})"
    return dict(key=name, pretty=_pretty_label(name), braid=word, strands=p,
                crossings=min(p * (q - 1), q * (p - 1)),
                alex=_torus_alexander(p, q), conway="", source="torus")


def resolve(text):
    """Resolve a typed knot name to a spec dict (braid, invariants, labels)."""
    raw = str(text).strip()
    if not raw:
        raise KnotError("请输入结名，例如 5_2 / Trefoil / 三叶结 / T(2,5)")
    norm = _clean(raw)
    if not norm:
        raise KnotError(f"无法识别的结名：{raw}")

    m = re.fullmatch(r"t(?:orus)?\s*[\(\[]?\s*(\d+)\s*[,;:/ ]\s*(\d+)\s*[\)\]]?",
                     norm)
    if m:
        return _torus(int(m.group(1)), int(m.group(2)))

    for key, entry in KNOT_TABLE.items():
        names = [key, key.replace("_", " "), key.replace("_", "-"),
                 entry["pretty"], *entry["aliases"]]
        for name in names:
            cleaned = _clean(name)
            if norm in (cleaned, cleaned + "结"):
                return _catalogue_spec(key)

    # torus knots beyond the catalogue
    flat = norm.replace(" ", "_").replace("-", "_")
    if flat in EXTRA_TORUS:
        return _torus(*EXTRA_TORUS[flat], label=flat)
    for label, (p, q) in EXTRA_TORUS.items():
        if norm.isdigit() and label.replace("_", "") == norm:
            return _torus(p, q, label=label)

    if norm.isdigit():
        hits = [f"{norm[:i]}_{norm[i:]}" for i in range(1, len(norm))
                if f"{norm[:i]}_{norm[i:]}" in KNOT_TABLE]
        if len(hits) == 1:
            return _catalogue_spec(hits[0])
        if len(hits) > 1:
            raise KnotError(f"名称有歧义：{raw}（可能是 {' / '.join(hits)}）")
    if re.fullmatch(r"\d+[_\s.-]?\d+", norm):
        raise KnotError(
            f"{raw} 暂未收录：目录覆盖 0₁–8₂₁；"
            "环面结可写 T(p,q)（如 T(2,9) 即 9₁）")
    raise KnotError(f"未知结名：{raw}；可用 n_k（3_1…8_21）、别名或 T(p,q)")


def catalogue_names():
    return ([KNOT_TABLE[k]["pretty"] for k in KNOT_TABLE]
            + [_pretty_label(k) for k in EXTRA_TORUS])


# ------------------------------------------------------- Alexander (torus)
def _poly_mul(a, b):
    out = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            out[e1 + e2] = out.get(e1 + e2, 0) + c1 * c2
    return {e: c for e, c in out.items() if c}


def _poly_div(num, den):
    num = {e: Fraction(c) for e, c in num.items()}
    den = {e: Fraction(c) for e, c in den.items()}
    dn = max(den)
    quot = {}
    while num and max(num) >= dn:
        top = max(num)
        c = num[top] / den[dn]
        shift = top - dn
        quot[shift] = quot.get(shift, Fraction(0)) + c
        for e, dc in den.items():
            num[e + shift] = num.get(e + shift, Fraction(0)) - c * dc
            if num[e + shift] == 0:
                del num[e + shift]
    return quot, num


def _torus_alexander(p, q):
    """Δ(t) ≐ (t^pq − 1)(t − 1) / ((t^p − 1)(t^q − 1))."""
    num = _poly_mul({p * q: 1, 0: -1}, {1: 1, 0: -1})
    den = _poly_mul({p: 1, 0: -1}, {q: 1, 0: -1})
    quot, rem = _poly_div(num, den)
    if rem:
        return {}
    lead = quot[max(quot)]
    if lead < 0:
        quot = {e: -c for e, c in quot.items()}
    return {e: int(c) if c.denominator == 1 else c for e, c in quot.items()}


# ------------------------------------------- Temperley-Lieb -> Jones poly
def _identity(n):
    m = [0] * (2 * n)
    for i in range(n):
        m[i] = n + i
        m[n + i] = i
    return tuple(m)


def _generator(n, i):
    m = list(_identity(n))
    m[i], m[i + 1] = i + 1, i
    m[n + i], m[n + i + 1] = n + i + 1, n + i
    return tuple(m)


def _compose(a, b, n):
    """Compose two TL diagrams: returns (matching, number of closed loops)."""
    parent = list(range(3 * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(2 * n):
        union(i, a[i])
    for i in range(2 * n):
        src = n + i if i < n else 2 * n + (i - n)
        dst = b[i]
        union(src, n + dst if dst < n else 2 * n + (dst - n))

    groups = {}
    for v in range(3 * n):
        groups.setdefault(find(v), []).append(v)
    loops = 0
    out = {}
    for nodes in groups.values():
        outer = [v for v in nodes if v < n or v >= 2 * n]
        if not outer:
            loops += 1
            continue
        if len(outer) != 2:
            raise RuntimeError("Temperley-Lieb composition produced a bad diagram")
        x, y = outer
        ix = x if x < n else n + (x - 2 * n)
        iy = y if y < n else n + (y - 2 * n)
        out[ix] = iy
        out[iy] = ix
    return tuple(out[i] for i in range(2 * n)), loops


def _closure_loops(m, n):
    parent = list(range(2 * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(2 * n):
        rx, ry = find(i), find(m[i])
        if rx != ry:
            parent[ry] = rx
    for i in range(n):
        rx, ry = find(i), find(n + i)
        if rx != ry:
            parent[ry] = rx
    return len({find(i) for i in range(2 * n)})


_DELTA = {2: Fraction(-1), -2: Fraction(-1)}


def _poly_add(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, Fraction(0)) + c
    return {e: c for e, c in out.items() if c}


def _poly_shift(p, k):
    return {e + k: c for e, c in p.items() if c}


def _delta_pow(k):
    p = {0: Fraction(1)}
    for _ in range(k):
        p = _poly_mul(p, _DELTA)
    return p


def kauffman_bracket_braid(word, strands):
    """Kauffman bracket of the closed braid, as {A exponent: Fraction}."""
    n = max(1, int(strands))
    state = {_identity(n): {0: Fraction(1)}}
    for g in word:
        i = abs(int(g)) - 1
        if i < 0 or i >= n - 1:
            raise KnotError(f"辫子生成元 σ{i + 1} 超出 {n} 股范围")
        gen = _generator(n, i)
        sign = 1 if g > 0 else -1
        new = {}
        for matching, poly in state.items():
            new[matching] = _poly_add(new.get(matching, {}), _poly_shift(poly, sign))
            comp, loops = _compose(matching, gen, n)
            term = _poly_shift(poly, -sign)
            if loops:
                term = _poly_mul(term, _delta_pow(loops))
            new[comp] = _poly_add(new.get(comp, {}), term)
        state = {m: p for m, p in new.items() if p}

    total = {}
    for matching, poly in state.items():
        loops = _closure_loops(matching, n)
        term = _poly_mul(poly, _delta_pow(max(0, loops - 1)))
        total = _poly_add(total, term)
    return total


def jones_from_braid(word, strands):
    """Jones polynomial V(t) of the closed braid, as {exponent: Fraction}."""
    word = tuple(int(g) for g in word)
    writhe = sum(1 if g > 0 else -1 for g in word)
    bracket = kauffman_bracket_braid(word, strands)
    sign = Fraction(-1) ** writhe
    out = {}
    for a, c in bracket.items():
        exponent = Fraction(-(a - 3 * writhe), 4)
        out[exponent] = out.get(exponent, Fraction(0)) + c * sign
    return {e: c for e, c in out.items() if c}


def determinant_from_jones(poly):
    """det(K) = |V(-1)|."""
    total = Fraction(0)
    for e, c in poly.items():
        if e.denominator != 1:
            continue
        total += c * (1 if int(e) % 2 == 0 else -1)
    return abs(int(total)) if total.denominator == 1 else abs(total)


def writhe_of(word):
    return sum(1 if g > 0 else -1 for g in word)


def mirror_jones(poly):
    return {(-e): c for e, c in poly.items()}


# --------------------------------------------------------- braid geometry
def braid_permutation(word, strands):
    """strand index (start slot) -> end slot."""
    slots = list(range(strands))
    for g in word:
        i = abs(int(g)) - 1
        a, b = slots.index(i), slots.index(i + 1)
        slots[a], slots[b] = i + 1, i
    return slots


def _smooth(u):
    return u * u * (3.0 - 2.0 * u)


def _torus_params(word, strands):
    """If the braid is the standard torus braid (σ1…σ_{p-1})^q, return (p, q, sign)."""
    n = int(strands)
    w = tuple(int(g) for g in word)
    if n < 2 or not w:
        return None
    base = tuple(range(1, n))
    for sign in (1, -1):
        unit = base if sign > 0 else tuple(-g for g in base)
        if len(w) % len(unit):
            continue
        q = len(w) // len(unit)
        if q and w == unit * q:
            return (n, q, sign)
    return None


def torus_curve(p, q, sign=1, samples=900, major=2.0, minor=0.72):
    """Classic (p, q) torus knot: winds p times around the axis, q times the tube."""
    pts = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        rr = major + minor * math.cos(q * t)
        pts.append((rr * math.cos(p * t), rr * math.sin(p * t),
                    sign * minor * math.sin(q * t)))
    return pts


TORUS_KEYS = {"3_1": (2, 3), "5_1": (2, 5), "7_1": (2, 7), "8_19": (3, 4)}


def braid_curves(word, strands, per_step=30, major=2.10, minor=0.66,
                 over=0.09, key=None):
    """Closed braid laid on a torus surface.

    The braid length becomes the major angle (one full turn) and the strand
    slot becomes the minor angle: slot k sits at theta = 2*pi*k/n.  Every
    generator swaps two neighbouring slots (with the over-strand bulging out
    and the under-strand dipping in), so after one full turn strand s ends
    exactly where strand sigma(s) begins.  The curve is therefore closed with
    no closure frame, and torus braids come out as classic torus knots.
    """
    n = max(1, int(strands))
    word = tuple(int(g) for g in word)

    if n == 1:
        pts = [(math.cos(t), math.sin(t), 0.0)
               for t in [2 * math.pi * i / 96 for i in range(96)]]
        return _normalise([pts])

    torus = None
    if key in TORUS_KEYS:
        p, q = TORUS_KEYS[key]
        torus = (p, q, 1)
    else:
        torus = _torus_params(word, n)
    if torus is not None:
        p, q, sign = torus
        return _normalise([torus_curve(p, q, sign)])

    total = max(1, len(word))

    def point(slot, u, bump):
        theta = 2.0 * math.pi * slot / n
        phi = 2.0 * math.pi * u
        rr = minor + bump
        return ((major + rr * math.cos(theta)) * math.cos(phi),
                (major + rr * math.cos(theta)) * math.sin(phi),
                rr * math.sin(theta))

    slots = list(range(n))
    paths = [[point(s, 0.0, 0.0)] for s in range(n)]

    for k, g in enumerate(word):
        i = abs(g) - 1
        sign = 1 if g > 0 else -1
        a, b = i, i + 1
        sa, sb = slots.index(a), slots.index(b)
        over_strand = sa if sign > 0 else sb
        for s in range(n):
            start = slots[s]
            if s == sa or s == sb:
                target = b if start == a else a
                bump_sign = 1.0 if s == over_strand else -1.0
            else:
                target = start
                bump_sign = 0.0
            for j in range(1, per_step + 1):
                t = j / per_step
                e = _smooth(t)
                slot = start + (target - start) * e
                u = (k + t) / total
                # every strand is sampled through every step: skipping the
                # idle strands would join two crossings with a straight chord
                bump = (over * bump_sign * math.sin(math.pi * t) ** 2)
                paths[s].append(point(slot, u, bump))
        slots[sa], slots[sb] = b, a

    final_slot = braid_permutation(word, n)
    seen = set()
    components = []
    for start in range(n):
        if start in seen:
            continue
        comp = []
        s = start
        while s not in seen:
            seen.add(s)
            comp.extend(paths[s])
            s = final_slot[s]
        components.append(_smooth_closed(_chaikin(_dedupe(comp), 2), 8, 0.38))
    return _normalise(components)


def _smooth_closed(pts, iterations=8, factor=0.38):
    """Laplacian smoothing on a closed polyline (rounds hard corners)."""
    for _ in range(iterations):
        count = len(pts)
        out = []
        for i in range(count):
            ax, ay, az = pts[i - 1]
            bx, by, bz = pts[i]
            cx, cy, cz = pts[(i + 1) % count]
            out.append((bx + factor * ((ax + cx) * 0.5 - bx),
                        by + factor * ((ay + cy) * 0.5 - by),
                        bz + factor * ((az + cz) * 0.5 - bz)))
        pts = out
    return pts


def _chaikin(pts, iterations=2):
    """Corner-cutting smoothing (closed polyline), keeps the curve close."""
    for _ in range(iterations):
        out = []
        count = len(pts)
        for i in range(count):
            ax, ay, az = pts[i]
            bx, by, bz = pts[(i + 1) % count]
            out.append((0.75 * ax + 0.25 * bx,
                        0.75 * ay + 0.25 * by,
                        0.75 * az + 0.25 * bz))
            out.append((0.25 * ax + 0.75 * bx,
                        0.25 * ay + 0.75 * by,
                        0.25 * az + 0.75 * bz))
        pts = out
    return pts


def _dedupe(pts, eps=1e-9):
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) + abs(p[2] - out[-1][2]) > eps:
            out.append(p)
    if len(out) > 1 and (abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1])
                         + abs(out[0][2] - out[-1][2])) < eps:
        out.pop()
    return out


def _normalise(components, target=1.9):
    xs = [p[0] for c in components for p in c]
    ys = [p[1] for c in components for p in c]
    zs = [p[2] for c in components for p in c]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    scale = target / span
    out = []
    for comp in components:
        out.append([((x - cx) * scale, (y - cy) * scale, (z - cz) * scale)
                    for x, y, z in comp])
    return out


# ------------------------------------------------------------ formatting
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _power(exp, var="t"):
    exp = int(exp)
    if exp == 0:
        return ""
    if exp == 1:
        return var
    return var + str(exp).translate(_SUP)


def format_polynomial(poly, var="t"):
    """{Fraction exponent: Fraction coef} -> readable Laurent polynomial."""
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
        body = number + _power(exponent, var)
        if not body:
            body = "1"
        pieces.append((sign, body))
    first_sign, first_body = pieces[0]
    text = ("-" if first_sign == "-" else "") + first_body
    for sign, body in pieces[1:]:
        text += f" {sign} {body}"
    return text
