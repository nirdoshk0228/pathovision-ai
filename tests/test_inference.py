import numpy as np
import pytest

from pathovision.inference.quantify import max_feret_diameter, quantify, size_category
from pathovision.inference.stitching import StitchAccumulator
from pathovision.preprocessing.stain import MacenkoNormalizer, estimate_stain_matrix, hed_jitter
from pathovision.data.augment import dihedral, dihedral_inverse


def test_stitch_constant_tiles_reconstruct_constant():
    acc = StitchAccumulator((4000, 3000), downsample=10, window="hann")
    fp, stride = 500, 250
    for y in range(0, 3000 - fp + 1, stride):
        for x in range(0, 4000 - fp + 1, stride):
            acc.add(np.full((64, 64), 0.7, np.float32), x, y, fp)
    out = acc.result()
    assert np.allclose(out[acc.coverage()], 0.7, atol=1e-5)


def test_stitch_no_gaps_with_awkward_footprint():
    """Non-integer footprint / downsample must not leave uncovered seams between adjacent tiles."""
    W = H = 10_000
    fp, ds = 384.0, 48.83
    acc = StitchAccumulator((W, H), ds, window="flat")
    n = int(np.ceil(W / fp))
    for r in range(n):
        for c in range(n):
            acc.add(np.ones((2, 2), np.float32), c * fp, r * fp, fp)
    assert acc.coverage().all()


def test_stitch_blends_overlap_smoothly():
    acc = StitchAccumulator((2000, 1000), downsample=4, window="hann")
    acc.add(np.zeros((128, 128)), 0, 0, 1000)
    acc.add(np.ones((128, 128)), 500, 0, 1000)
    acc.add(np.ones((128, 128)), 1000, 0, 1000)
    row = acc.result()[125]
    # monotone transition across the overlap instead of a hard step
    seg = row[125:250]
    assert np.all(np.diff(seg) >= -1e-6) and 0.05 < seg[len(seg) // 2] < 0.95


def test_feret_diameter_of_disk():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    pts = np.stack([50 + 30 * np.cos(t), 50 + 30 * np.sin(t)], 1)
    assert max_feret_diameter(pts) == pytest.approx(60, rel=0.01)


def test_quantify_disk_area_and_category():
    prob = np.zeros((400, 400), np.float32)
    yy, xx = np.mgrid[0:400, 0:400]
    prob[(yy - 200) ** 2 + (xx - 200) ** 2 <= 100 ** 2] = 0.9
    tissue = np.ones_like(prob, bool)
    ds, mpp = 16.0, 0.25  # 4 um per map pixel -> disk radius 400 um
    q = quantify(prob, tissue, ds, mpp)
    assert q["n_lesions"] == 1
    assert q["tumor_area_mm2"] == pytest.approx(np.pi * 0.4 ** 2, rel=0.02)
    assert q["largest_lesion_diameter_mm"] == pytest.approx(0.8, rel=0.03)
    assert q["slide_category"] == "micrometastasis"
    assert size_category(2.5) == "macrometastasis" and size_category(0.1) == "ITC"


def _he_like(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((128, 128, 3), (235, 170, 205), np.float32)
    for _ in range(80):
        y, x = rng.integers(0, 128, 2)
        img[max(0, y - 3): y + 3, max(0, x - 3): x + 3] = (90, 50, 140)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def test_macenko_shapes_and_background_passthrough():
    img = _he_like()
    he = estimate_stain_matrix(img)
    assert he.shape == (3, 2) and np.allclose(np.linalg.norm(he, axis=0), 1)
    assert he[0, 0] > he[0, 1]  # haematoxylin first (stronger red absorption)
    n = MacenkoNormalizer().fit(_he_like(1))
    out = n.transform(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    blank = np.full((64, 64, 3), 245, np.uint8)
    assert np.array_equal(n.transform(blank), blank)


def test_hed_jitter_identity_when_zero():
    img = _he_like()
    out = hed_jitter(img, sigma=0.0, bias=0.0)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 2


@pytest.mark.parametrize("k", range(8))
def test_dihedral_inverse(k):
    a = np.arange(5 * 7 * 3).reshape(5, 7, 3)
    assert np.array_equal(dihedral_inverse(dihedral(a, k), k), a)
