"""Pins the stage-5 geometry replay that `annotate_augmented.py` depends on.

Two things here are load-bearing and both fail silently rather than loudly:

  * the forward transform must agree with what PIL actually did to the pixels.
    An inverted rotation transpose still produces plausible-looking boxes; it
    is just wrong by ~83 px at 3 degrees. So the transform is checked against
    landmarks pushed through the real `_rotate_on_paper` / `Image.QUAD`, not
    against a second implementation of the same reasoning.

  * the RNG stream must stay exactly as it was. Every image in the corpus, and
    every annotation mapped onto it, is reproducible only while the geometry
    draws are consumed in the same order and quantity. `test_stream_unchanged`
    pins the drawn values so a future edit to the draw order fails here rather
    than quietly invalidating an already-built corpus.

Run with:  ~/.cache/ecgkit-venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import config as C          # noqa: E402
import geometry             # noqa: E402
from stage5_augment import _rotate_on_paper   # noqa: E402

W, H = 1650, 1275
LANDMARKS = [(200, 150), (1400, 150), (200, 1100), (1400, 1100),
             (825, 637), (1000, 400), (60, 60), (1590, 1215)]


def _apply_real_geometry(img: Image.Image, geom: geometry.Geometry) -> Image.Image:
    """Exactly the geometric steps stage 5 performs, and nothing else."""
    img = _rotate_on_paper(img, geom.degrees, (0, 0, 0))
    if geom.quad is not None:
        img = img.transform(img.size, Image.QUAD, geom.quad,
                            resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    m = geom.margin
    canvas = Image.new("RGB", (img.width + 2 * m, img.height + 2 * m), (0, 0, 0))
    canvas.paste(img, (m, m))
    return canvas


def _centroid_after(point: tuple[int, int], geom: geometry.Geometry) -> tuple[float, float]:
    """Push a single marked block through the real transform, return its centroid."""
    x, y = point
    arr = np.zeros((H, W), np.uint8)
    arr[y - 2:y + 3, x - 2:x + 3] = 255
    out = _apply_real_geometry(Image.fromarray(arr).convert("RGB"), geom)
    o = np.asarray(out.convert("L")).astype(np.float64)
    total = o.sum()
    assert total > 0, "landmark fell off the canvas; pick one further from the edge"
    yy, xx = np.mgrid[0:o.shape[0], 0:o.shape[1]]
    return (o * xx).sum() / total, (o * yy).sum() / total


@pytest.mark.parametrize("seed", range(12))
def test_forward_matches_pillow(seed: int) -> None:
    """Predicted positions must match where PIL actually put the pixels.

    The tolerance is sub-pixel; what is left is bicubic centroid bias, not
    model error. A transposed rotation blows through this by two orders of
    magnitude.
    """
    geom = geometry.geometry_draws(np.random.default_rng(seed), (W, H), C.MARGIN_PX)
    predicted = geometry.forward_points(geom, np.array(LANDMARKS, dtype=float))

    for (point, pred) in zip(LANDMARKS, predicted):
        cx, cy = _centroid_after(point, geom)
        assert np.hypot(cx - pred[0], cy - pred[1]) < 0.5


def test_keystone_is_bilinear_not_projective() -> None:
    """Guards the reason `forward_points` is not just a 3x3 multiply.

    PIL's QUAD carries a u*v cross term, so the mapping is bilinear. If a
    future refactor swaps in a homography because it "looks projective", the
    error is small enough to miss by eye - a few pixels at the corners - but
    that is most of an ST segment at 150 DPI. This asserts the two genuinely
    disagree, so the distinction cannot be optimised away by accident.
    """
    import cv2

    geom = geometry.Geometry(
        degrees=0.0,
        quad=(20.0, 15.0, 5.0, H - 25.0, W - 18.0, H - 6.0, W - 9.0, 22.0),
        src_size=(W, H), margin=0)

    src = np.array([[geom.quad[0], geom.quad[1]], [geom.quad[2], geom.quad[3]],
                    [geom.quad[4], geom.quad[5]], [geom.quad[6], geom.quad[7]]],
                   dtype=np.float32)
    dst = np.array([[0, 0], [0, H], [W, H], [W, 0]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)

    pts = np.array(LANDMARKS, dtype=float)
    bilinear = geometry.forward_points(geom, pts)

    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ homography.T
    projective = hom[:, :2] / hom[:, 2:3]

    # They agree at the corners by construction and diverge in between.
    assert np.abs(bilinear - projective).max() > 1.0


@pytest.mark.parametrize("seed", range(8))
def test_inverse_bilinear_roundtrips(seed: int) -> None:
    """invert_bilinear must actually invert PIL's coefficients."""
    geom = geometry.geometry_draws(np.random.default_rng(seed), (W, H), C.MARGIN_PX)
    if geom.quad is None:
        pytest.skip("no keystone drawn for this seed")

    coef = geometry.quad_coefficients(geom.quad, W, H)
    dest = np.array([[3.0, 7.0], [W - 4.0, 11.0], [W / 2, H / 2],
                     [17.0, H - 9.0], [W - 21.0, H - 13.0]])
    src = geometry._bilinear_forward(coef, dest)
    back = geometry.invert_bilinear(coef, src)
    assert np.abs(back - dest).max() < 1e-6


def test_stream_unchanged() -> None:
    """The corpus is only reproducible while these exact values are drawn.

    Pinned from the build that produced the current corpus. If this fails, the
    draw order or count in `geometry_draws` changed and every existing image's
    annotation mapping is invalid - regenerate, do not just update the numbers.
    """
    geom = geometry.geometry_draws(np.random.default_rng(0), (W, H), C.MARGIN_PX)
    assert geom.degrees == pytest.approx(0.8217701239287258, abs=1e-12)
    assert geom.quad is not None
    assert len(geom.quad) == 8
    assert geom.quad[0] == pytest.approx(0.12472560984383414, abs=1e-12)

    # A seed that draws no keystone must consume exactly two values, so the
    # photometric draws that follow line up either way.
    plain = geometry.geometry_draws(np.random.default_rng(1), (W, H), C.MARGIN_PX)
    assert plain.quad is None
    assert plain.degrees == pytest.approx(0.0709297482015403, abs=1e-12)


def test_dst_size_matches_corpus() -> None:
    """The margin is what turns a 1650x1275 render into a 1686x1311 image."""
    geom = geometry.geometry_draws(np.random.default_rng(3), (W, H), C.MARGIN_PX)
    assert geom.dst_size == (W + 2 * C.MARGIN_PX, H + 2 * C.MARGIN_PX)
    assert geom.dst_size == (1686, 1311)
