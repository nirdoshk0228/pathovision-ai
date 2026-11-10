import json
import os

import numpy as np
import pytest
import torch

from pathovision.models.classifier import PatchClassifier
from pathovision.models.unet import UNet, build_segmenter
from pathovision.training.engine import tta_logits, tta_seg_logits


def test_classifier_output_shape():
    m = PatchClassifier("resnet18", pretrained=False).eval()
    assert m(torch.randn(3, 3, 96, 96)).shape == (3,)


@pytest.mark.parametrize("arch", ["unet", "resnet18_unet"])
def test_segmenter_preserves_spatial_size(arch):
    m = build_segmenter(arch, pretrained=False).eval()
    for s in (128, 256, 200):
        assert m(torch.randn(1, 3, s, s)).shape == (1, 1, s, s)


def test_seg_tta_is_exact_for_equivariant_model():
    """Identity 'model' is perfectly equivariant: TTA must return the input unchanged."""
    x = torch.randn(2, 1, 16, 16)
    assert torch.allclose(tta_seg_logits(lambda t: t, x), x, atol=1e-6)
    assert tta_logits(lambda t: t.mean((1, 2, 3)), x).shape == (2,)


@pytest.fixture(scope="module")
def small_slide(tmp_path_factory):
    from pathovision.utils.synthetic import make_synthetic_slide

    d = tmp_path_factory.mktemp("slide")
    info = make_synthetic_slide(str(d / "s.tif"), str(d / "s.xml"), width=8192, height=6144, n_tumors=3,
                                seed=3, verbose=False)
    return info


def test_reader_pyramid_and_mpp(small_slide):
    from pathovision.wsi.reader import WSIReader

    with WSIReader(small_slide["path"]) as r:
        assert r.dimensions == (8192, 6144)
        assert r.info.level_count >= 3
        assert r.mpp == pytest.approx(0.243, rel=1e-3)
        assert r.best_level_for_downsample(4.0) == 2
        assert r.read_at_mpp(1000, 1000, 96, 0.972).shape == (96, 96, 3)
        thumb, ds = r.thumbnail(1024)
        assert max(thumb.shape[:2]) == 1024 and ds == pytest.approx(8.0)
        # out-of-bounds reads are padded with white background
        assert r.read_region(8100, 6100, 0, 200, 200)[-1, -1].tolist() == [255, 255, 255]


def test_end_to_end_slide_analysis(small_slide, tmp_path):
    from pathovision.inference.slide_inference import AnalysisConfig, LoadedModel, SlideAnalyzer

    cls = LoadedModel(PatchClassifier("resnet18", pretrained=False).eval(), {"patch_size": 96, "mpp": 0.972})
    seg = LoadedModel(UNet(base=8).eval(), {"patch_size": 128, "mpp": 1.944})
    cfg = AnalysisConfig(num_workers=0, batch_size=64, seg_gate=0.0, map_max_side=1024, device="cpu")
    rep = SlideAnalyzer(cfg, classifier=cls, segmenter=seg).analyze(
        small_slide["path"], str(tmp_path), small_slide["xml"])
    for f in ("report.json", "heatmap_overlay.png", "prediction_overlay.png", "tiles.csv",
              "predictions.geojson", "predictions.xml", "tumor_mask.png", "prob_map.npy"):
        assert os.path.exists(tmp_path / f), f
    saved = json.load(open(tmp_path / "report.json"))
    assert saved["benchmark"]["tiles_classified"] > 0
    assert 0 < saved["benchmark"]["tiles_skipped_background_pct"] < 100
    assert "dice" in saved["ground_truth"]
    assert rep["_artefacts"]["map"].shape == rep["_artefacts"]["tissue_map"].shape
