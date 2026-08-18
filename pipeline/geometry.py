"""Stage 5's geometry, factored out so annotations can follow the pixels.

Stage 5 moves every pixel of a sheet three times - a small tilt, an optional
keystone, and the margin paste - and the annotation JSON that ecg-image-kit
writes describes only the *clean* render, a canvas that no longer exists by the
time the JPEG is written. `visualize_annotations.py` refuses to draw on that
mismatch rather than putting boxes in the wrong place.

This module reconstructs the mapping so a point on the clean render can be
carried onto the augmented image, which is what makes the boxes transferable.

Two properties let this be done for a corpus that is already built, without
re-rendering anything:

  * the augmentation is deterministic - `stage5_augment._rng` keys a PCG64
    stream off (record, render_k) and the build seed, nothing else;
  * the geometry draws are consumed *first*, before any photometric draw.

So replaying the head of the stream reproduces the exact tilt and keystone a
given image was built with. `geometry_draws` is that replay, and stage 5 calls
the same function when it augments, so the two cannot disagree about what was
drawn or in what order.

**The keystone is bilinear, not projective.** PIL's `Image.QUAD` is not a
homography: its coefficients carry a `u*v` cross term (see `Image.transform`,
"quadrilateral warp"), so it maps a quadrilateral to a rectangle by bilinear
interpolation of the corners. Treating it as a homography looks nearly right
and is wrong by up to ~3.5 px at the corners, which is most of the way across
an ST segment at this DPI. The forward direction therefore needs an inverse
bilinear solve, done in closed form and refined by Newton below.

Coordinates here are (x, y) in pixels, origin top-left, y down - OpenCV's
convention. The kit's JSON stores points as [y, x]; converting is the caller's
job (see `annotate_augmented.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """The geometric half of one image's augmentation."""

    degrees: float                     # tilt, CCW positive (PIL's convention)
    quad: tuple[float, ...] | None     # 8-tuple source quad, or None if skipped
    src_size: tuple[int, int]          # (w, h) of the clean render
    margin: int                        # px added on every side, last

    @property
    def dst_size(self) -> tuple[int, int]:
        w, h = self.src_size
        return w + 2 * self.margin, h + 2 * self.margin


def geometry_draws(rng: np.random.Generator, size: tuple[int, int],
                   margin: int) -> Geometry:
    """Consume the geometry head of an augmentation's RNG stream.

    MUST stay exactly in step with the draws stage 5 performs, in the same
    order: the photometric draws that follow depend on the stream being
    advanced by precisely this much, so a corpus built before this refactor
    still reproduces byte for byte. `stage5_augment.augment` calls this rather
    than drawing inline, so there is one definition of the order, not two.
    """
    w, h = size
    degrees = float(rng.uniform(-3.0, 3.0))

    quad = None
    if rng.random() < 0.7:                       # keystone, as if photographed
        m = rng.uniform(0.004, 0.018)
        dx, dy = m * w, m * h
        # Tuple elements evaluate left to right; this order is load-bearing.
        quad = (
            rng.uniform(0, dx), rng.uniform(0, dy),
            rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), rng.uniform(0, dy),
        )
    return Geometry(degrees=degrees, quad=quad, src_size=(w, h), margin=margin)


# ------------------------------------------------------------------- rotation
def rotation_matrix(deg: float, w: int, h: int) -> np.ndarray:
    """Forward (source -> destination) matrix for `_rotate_on_paper`.

    This mirrors PIL's own `Image.rotate(expand=True)` arithmetic, then the
    centre-crop back to the original size that `_rotate_on_paper` does.

    PIL builds a *destination -> source* matrix (the direction a resampling
    loop needs) and rotates about the image centre:

        src = M @ (dst - centre) + centre

    Note PIL negates the angle first (`angle = -math.radians(angle)`), so with
    cos/sin taken of the *positive* angle that matrix is [[cos, -sin],
    [sin, cos]] and the forward map is its inverse, [[cos, sin], [-sin, cos]].
    Getting that transpose backwards is a silent error that grows with the
    tilt - about 83 px at 3 degrees - so it is pinned by a landmark test in
    `tests/test_geometry.py` rather than trusted to reasoning.

    Expanding the canvas shifts the destination by half the growth, and the
    crop shifts it back by the integer floor of that same half. The two do not
    cancel exactly - the crop floors where the expand does not - so the
    sub-pixel residual is carried rather than assumed away.
    """
    rad = math.radians(deg)
    cos, sin = math.cos(rad), math.sin(rad)

    # PIL rounds the matrix it uses to 15 decimals; match it so the landmark
    # test agrees to well under a pixel rather than merely closely.
    cos, sin = round(cos, 15), round(sin, 15)

    cx, cy = w / 2.0, h / 2.0

    # Expanded canvas size, by PIL's formula: transform the four corners
    # through the dst->src matrix and take the ceil/floor extent.
    xs, ys = [], []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        xs.append(cos * (x - cx) - sin * (y - cy) + cx)
        ys.append(sin * (x - cx) + cos * (y - cy) + cy)
    nw = math.ceil(max(xs)) - math.floor(min(xs))
    nh = math.ceil(max(ys)) - math.floor(min(ys))

    # expand shifts by the float half-growth; the crop takes the floored int.
    ox = (nw - w) / 2.0 - ((nw - w) // 2)
    oy = (nh - h) / 2.0 - ((nh - h) // 2)

    return np.array([
        [cos,  sin, cx - cos * cx - sin * cy + ox],
        [-sin, cos, cy + sin * cx - cos * cy + oy],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)


# ------------------------------------------------------------------- keystone
def quad_coefficients(quad: tuple[float, ...], w: int, h: int) -> np.ndarray:
    """PIL's eight bilinear coefficients for a QUAD transform (dst -> src).

    Reproduces the block in `PIL.Image.Image.transform` labelled "quadrilateral
    warp", where the quad is given as NW, SW, SE, NE:

        src_x = c0 + c1*u + c2*v + c3*u*v
        src_y = c4 + c5*u + c6*v + c7*u*v
    """
    nw, sw, se, ne = quad[0:2], quad[2:4], quad[4:6], quad[6:8]
    x0, y0 = nw
    as_, at = 1.0 / w, 1.0 / h
    return np.array([
        x0,
        (ne[0] - x0) * as_,
        (sw[0] - x0) * at,
        (se[0] - sw[0] - ne[0] + x0) * as_ * at,
        y0,
        (ne[1] - y0) * as_,
        (sw[1] - y0) * at,
        (se[1] - sw[1] - ne[1] + y0) * as_ * at,
    ], dtype=np.float64)


def _bilinear_forward(coef: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Evaluate the dst -> src bilinear map on (N, 2) destination points."""
    u, v = uv[:, 0], uv[:, 1]
    c = coef
    return np.stack([
        c[0] + c[1] * u + c[2] * v + c[3] * u * v,
        c[4] + c[5] * u + c[6] * v + c[7] * u * v,
    ], axis=1)


def invert_bilinear(coef: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Solve the bilinear map for destination coords given source points.

    Eliminating u between the two equations leaves a quadratic in v:

        A v^2 + B v + C = 0

    The near-rectangular quads this pipeline draws make A tiny but not always
    zero, so both the linear and quadratic branches are handled, and the root
    that lands nearer the canvas is taken. Two Newton steps then polish the
    result, which matters because A is small enough for the closed form to lose
    precision to cancellation.
    """
    c = coef
    X, Y = pts[:, 0], pts[:, 1]

    A = c[2] * c[7] - c[6] * c[3]
    B = ((Y - c[4]) * c[3] - c[6] * c[1] - (X - c[0]) * c[7] + c[2] * c[5])
    C = (Y - c[4]) * c[1] - (X - c[0]) * c[5]

    with np.errstate(invalid="ignore", divide="ignore"):
        disc = np.maximum(B * B - 4 * A * C, 0.0)
        sqrt_disc = np.sqrt(disc)
        linear = np.where(np.abs(B) > 1e-15, -C / np.where(np.abs(B) > 1e-15, B, 1.0), 0.0)
        r1 = np.where(np.abs(A) > 1e-15, (-B + sqrt_disc) / (2 * np.where(np.abs(A) > 1e-15, A, 1.0)), linear)
        r2 = np.where(np.abs(A) > 1e-15, (-B - sqrt_disc) / (2 * np.where(np.abs(A) > 1e-15, A, 1.0)), linear)

    # Pick the root that stays near the sheet rather than the far branch of
    # the hyperbola, which for a near-rectangular quad sits enormously far off.
    pick_r1 = np.abs(r1) <= np.abs(r2)
    v = np.where(np.abs(A) > 1e-15, np.where(pick_r1, r1, r2), linear)

    denom = c[1] + c[3] * v
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    u = (X - c[0] - c[2] * v) / denom

    # Newton refinement on the 2x2 system; the Jacobian is well conditioned
    # for these quads, and two iterations reach float precision.
    for _ in range(2):
        uv = np.stack([u, v], axis=1)
        res = _bilinear_forward(c, uv) - pts
        j11 = c[1] + c[3] * v
        j12 = c[2] + c[3] * u
        j21 = c[5] + c[7] * v
        j22 = c[6] + c[7] * u
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)
        u = u - (res[:, 0] * j22 - res[:, 1] * j12) / det
        v = v - (res[:, 1] * j11 - res[:, 0] * j21) / det

    return np.stack([u, v], axis=1)


# -------------------------------------------------------------------- forward
def forward_points(geom: Geometry, pts: np.ndarray) -> np.ndarray:
    """Carry (N, 2) (x, y) points from the clean render to the augmented image.

    Applies the same three steps stage 5 applies to the pixels, in order:
    tilt, keystone, margin.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    w, h = geom.src_size

    matrix = rotation_matrix(geom.degrees, w, h)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    out = (homogeneous @ matrix.T)[:, :2]

    if geom.quad is not None:
        out = invert_bilinear(quad_coefficients(geom.quad, w, h), out)

    return out + geom.margin
