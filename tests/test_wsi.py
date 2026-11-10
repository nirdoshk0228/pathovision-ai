import numpy as np
import pytest

from pathovision.wsi.annotations import Polygon, load_asap_xml, mask_to_polygons, rasterize, save_asap_xml
from pathovision.wsi.tiler import IntegralImage, TileGrid
from pathovision.wsi.tissue import tissue_mask


def test_integral_image_matches_bruteforce():
    rng = np.random.default_rng(0)
    m = rng.random((57, 83)) > 0.6
    ii = IntegralImage(m)
    for _ in range(200):
        x0, x1 = sorted(rng.integers(0, 84, 2))
        y0, y1 = sorted(rng.integers(0, 58, 2))
        if x1 == x0 or y1 == y0:
            continue
        assert ii.rect_mean(x0, y0, x1, y1) == pytest.approx(m[y0:y1, x0:x1].mean())


def test_tilegrid_covers_slide_and_filters_background():
    W, H, fp = 10_000, 6_000, 384.0
    g = TileGrid((W, H), fp)
    xy = g.coords()
    assert xy[:, 0].min() == 0 and xy[:, 1].min() == 0
    assert xy[:, 0].max() + fp >= W and xy[:, 1].max() + fp >= H  # last tile reaches the border
    # tissue only in the left half -> roughly half the tiles survive
    tissue = np.zeros((60, 100), bool)
    tissue[:, :50] = True
    g2 = TileGrid((W, H), fp, tissue=tissue, mask_downsample=100.0, min_tissue=0.5)
    assert 0.4 < len(g2) / g.total_cells < 0.6
    assert (g2.coords()[:, 0] < W / 2).all()


def test_tilegrid_overlap_halves_stride():
    g = TileGrid((4096, 4096), 512, overlap=0.5)
    assert g.stride == 256
    assert g.n_cols == 15


def test_annotation_roundtrip_and_exclusion(tmp_path):
    outer = Polygon(np.array([[100, 100], [900, 100], [900, 900], [100, 900]], float), "_0")
    hole = Polygon(np.array([[400, 400], [600, 400], [600, 600], [400, 600]], float), "_2")
    p = tmp_path / "a.xml"
    save_asap_xml([outer, hole], str(p))
    polys = load_asap_xml(str(p))
    assert len(polys) == 2 and sum(q.is_exclusion for q in polys) == 1
    m = rasterize(polys, (100, 100), downsample=10)
    assert m[20, 20] and not m[50, 50] and not m[5, 5]
    expected = (80 * 80 - 20 * 20) / 100 ** 2
    # cv2.fillPoly also fills pixels touched by the edge (<= 1 px per boundary)
    assert abs(m.mean() - expected) < 0.015


def test_rasterize_patch_offset_matches_global():
    poly = Polygon(np.array([[1000, 1000], [3000, 1200], [2500, 3000], [1200, 2600]], float))
    full = rasterize([poly], (400, 400), downsample=10)
    patch = rasterize([poly], (50, 50), downsample=10, offset=(1500, 1800))
    assert (patch == full[180:230, 150:200]).mean() > 0.98


def test_mask_to_polygons_roundtrip():
    m = np.zeros((200, 200), bool)
    m[50:150, 40:160] = True
    m[90:110, 90:110] = False  # hole
    polys = mask_to_polygons(m, downsample=4.0, simplify_px=0)
    back = rasterize(polys, (200, 200), 4.0)
    assert (back == m).mean() > 0.99


def test_tissue_mask_separates_stain_from_glass():
    img = np.full((300, 400, 3), 242, np.uint8)
    img[60:240, 80:300] = (200, 120, 180)  # pink/purple tissue
    img += np.random.default_rng(0).integers(0, 6, img.shape, dtype=np.uint8)
    m = tissue_mask(img)
    assert m[150, 200] and not m[10, 10]
    assert abs(m.mean() - (180 * 220) / (300 * 400)) < 0.03
