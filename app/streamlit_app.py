"""PathoVision AI — interactive whole-slide analysis dashboard.

    streamlit run app/streamlit_app.py

Model and data locations come from the sidebar (defaults: models/ and data/slides/).
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pathovision.inference.slide_inference import AnalysisConfig, SlideAnalyzer, load_classifier, load_segmenter  # noqa: E402
from pathovision.inference.viz import add_colorbar, draw_mask_contours, overlay_heatmap, overlay_mask, resize_to  # noqa: E402
from pathovision.utils.common import get_device  # noqa: E402
from pathovision.wsi.annotations import mask_to_polygons, polygons_to_geojson  # noqa: E402
from pathovision.wsi.reader import PYRAMID_EXTS, WSIReader  # noqa: E402

st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

SLIDE_EXTS = sorted(PYRAMID_EXTS | {".png", ".jpg", ".jpeg"})


@st.cache_resource(show_spinner="Loading models...")
def get_models(cls_path: str, seg_path: str):
    device = get_device()
    cls = load_classifier(cls_path, device)
    seg = load_segmenter(seg_path, device) if seg_path and os.path.exists(seg_path) else None
    return cls, seg


def list_slides(folder: str):
    out = []
    for ext in SLIDE_EXTS:
        out += glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True)
    return sorted(out)


def find_xml(slide_path: str):
    sid = os.path.splitext(os.path.basename(slide_path))[0]
    root = os.path.dirname(os.path.dirname(slide_path))
    for cand in (os.path.join(os.path.dirname(slide_path), sid + ".xml"), os.path.join(root, "annotations", sid + ".xml")):
        if os.path.exists(cand):
            return cand
    return None


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔬 PathoVision AI")
st.sidebar.caption("Whole-slide metastasis detection in H&E lymph-node sections")
with st.sidebar.expander("Models", expanded=False):
    cls_path = st.text_input("Classifier checkpoint", os.path.join(ROOT, "models", "classifier.pt"))
    seg_path = st.text_input("Segmenter checkpoint (optional)", os.path.join(ROOT, "models", "segmenter.pt"))
slide_dir = st.sidebar.text_input("Slide folder", os.path.join(ROOT, "data", "slides"))
source = st.sidebar.radio("Slide source", ["From folder", "Upload"], horizontal=True)
slide_path = None
if source == "From folder":
    slides = list_slides(slide_dir)
    if slides:
        slide_path = st.sidebar.selectbox("Slide", slides, format_func=lambda p: os.path.relpath(p, slide_dir))
    else:
        st.sidebar.info(f"No slides found in {slide_dir}")
else:
    up = st.sidebar.file_uploader("Slide (.tif/.svs/.ndpi or small .png/.jpg)", type=[e[1:] for e in SLIDE_EXTS])
    if up is not None:
        tmp = os.path.join(tempfile.gettempdir(), "pathovision_upload_" + up.name)
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        slide_path = tmp

st.sidebar.subheader("Analysis settings")
use_seg = st.sidebar.toggle("U-Net segmentation", value=True, help="Pixel-level masks around suspicious regions")
tta = st.sidebar.select_slider("Test-time augmentation views", [1, 2, 4, 8], value=1)
seg_gate = st.sidebar.slider("Segmentation gate (classifier prob)", 0.05, 0.9, 0.3, 0.05)
stain = st.sidebar.toggle("Macenko stain normalisation", value=False)
gt_toggle = st.sidebar.toggle("Compare with annotation (if available)", value=True)
run = st.sidebar.button("Analyse slide", type="primary", width="stretch", disabled=slide_path is None)

# ------------------------------------------------------------------ header
st.title("PathoVision AI")
st.caption("Gigapixel H&E analysis: tissue detection → patch classification → U-Net segmentation → lesion quantification")

if not os.path.exists(cls_path):
    st.warning("Classifier checkpoint not found. Train one (see README) or run `python scripts/demo_synthetic.py` "
               "to create demo models, then set the path in the sidebar.")
    st.stop()

if slide_path is not None:
    rd = WSIReader(slide_path)
    info = rd.info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slide", os.path.basename(slide_path))
    c2.metric("Size (level 0)", f"{info.width:,} × {info.height:,}")
    c3.metric("Gigapixels", f"{info.width * info.height / 1e9:.2f}")
    c4.metric("Resolution", f"{info.mpp:.3f} µm/px", help=f"source: {info.mpp_source}; {info.level_count} pyramid levels")
    rd.close()

if run and slide_path:
    cls_m, seg_m = get_models(cls_path, seg_path)
    cfg = AnalysisConfig(tta=tta, seg_gate=seg_gate, stain_norm=stain, num_workers=0)
    analyzer = SlideAnalyzer(cfg, classifier=cls_m, segmenter=seg_m if use_seg else None)
    bar = st.progress(0.0, text="Starting...")
    labels = {"tissue": "Detecting tissue", "classification": "Classifying patches", "segmentation": "Segmenting"}

    def prog(stage, frac):
        bar.progress(min(1.0, frac), text=f"{labels.get(stage, stage)}: {frac * 100:.0f}%")

    gt = find_xml(slide_path) if gt_toggle else None
    rep = analyzer.analyze(slide_path, None, gt, prog)
    bar.empty()
    st.session_state["result"] = {"report": rep, "slide": slide_path}

res = st.session_state.get("result")
if not res:
    st.info("Pick a slide and press **Analyse slide**.")
    st.stop()

rep, a = res["report"], res["report"]["_artefacts"]
q, ss, bm = rep["quantification"], rep["slide_score"], rep["benchmark"]

# ------------------------------------------------------------------ summary
label = ss["predicted_label"].upper()
st.subheader(f"Result: {'🔴' if label == 'TUMOR' else '🟢'} {label} — {q['slide_category']}")
m = st.columns(6)
m[0].metric("Max patch probability", f"{ss['max_tile_prob']:.3f}")
m[1].metric("Tumour burden", f"{q['tumor_burden_pct']:.2f} %")
m[2].metric("Tumour area", f"{q['tumor_area_mm2']:.3f} mm²")
m[3].metric("Lesions", q["n_lesions"])
m[4].metric("Largest lesion", f"{q['largest_lesion_diameter_mm']:.2f} mm")
m[5].metric("Analysis time", f"{bm['total_seconds']:.1f} s")
if "ground_truth" in rep:
    g = rep["ground_truth"]
    st.caption(f"Against annotation **{g['xml']}**: Dice = **{g['dice']:.3f}**, IoU = **{g['iou']:.3f}** "
               f"(annotated tumour area {g['gt_tumor_area_mm2']:.3f} mm²)")

thumb = a["thumb"]
th, tw = thumb.shape[:2]
tissue_t = resize_to(a["tissue_map"].astype(np.uint8), (th, tw), nearest=True) > 0
pred = (a["map"] >= a["threshold"]) & a["tissue_map"]

tabs = st.tabs(["Heatmap", "Segmentation", "Tissue", "Lesions", "Patch inspector", "Performance", "Export"])

with tabs[0]:
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)
    floor = st.slider("Hide probabilities below", 0.0, 0.9, 0.1, 0.05)
    c1, c2 = st.columns(2)
    c1.image(add_colorbar(overlay_heatmap(thumb, a["cls_map"], tissue_t, alpha, floor)),
             caption="Patch-level tumour probability", width="stretch")
    c2.image(thumb, caption="Original (thumbnail)", width="stretch")

with tabs[1]:
    img = overlay_mask(thumb, pred, (230, 30, 30), 0.45)
    img = draw_mask_contours(img, pred, (200, 0, 0), 1)
    if a.get("gt_map") is not None:
        img = draw_mask_contours(img, a["gt_map"], (0, 200, 0), 2)
    st.image(img, caption=f"Predicted tumour ({rep['map_source']}) in red"
             + (", pathologist annotation outlined in green" if a.get("gt_map") is not None else ""),
             width="stretch")

with tabs[2]:
    st.image(overlay_mask(thumb, a["tissue"], (0, 160, 255), 0.4), width="stretch",
             caption=f"Tissue mask: {bm['tiles_classified']:,} of {bm['grid_cells_total']:,} grid patches contain tissue "
                     f"({bm['tiles_skipped_background_pct']}% skipped as background)")

with tabs[3]:
    if q["lesions"]:
        df = pd.DataFrame(q["lesions"])
        st.dataframe(df[["id", "category", "area_mm2", "max_diameter_mm", "mean_prob", "max_prob", "centroid_x", "centroid_y"]]
                     .style.format({"area_mm2": "{:.4f}", "max_diameter_mm": "{:.3f}", "mean_prob": "{:.3f}",
                                    "max_prob": "{:.3f}", "centroid_x": "{:.0f}", "centroid_y": "{:.0f}"}),
                     width="stretch", hide_index=True)
        lid = st.selectbox("Zoom into lesion", df["id"].tolist())
        L = df[df["id"] == lid].iloc[0]
        rd = WSIReader(res["slide"])
        pad = 0.25
        x0 = int(L.bbox_x - pad * L.bbox_w)
        y0 = int(L.bbox_y - pad * L.bbox_h)
        side = int(max(L.bbox_w, L.bbox_h) * (1 + 2 * pad))
        x0 -= (side - int(L.bbox_w * (1 + 2 * pad))) // 2
        y0 -= (side - int(L.bbox_h * (1 + 2 * pad))) // 2
        out_px = 900
        ds = max(1.0, side / out_px)
        crop = rd.read_at_downsample(x0, y0, out_px, out_px, ds)
        um_px = ds * rd.mpp
        rd.close()
        mds = a["map_ds"]
        r0, c0 = int(y0 / mds), int(x0 / mds)
        n = max(1, int(round(side / mds)))
        sub = np.zeros((n, n), bool)
        rr0, cc0 = max(0, r0), max(0, c0)
        rr1, cc1 = min(pred.shape[0], r0 + n), min(pred.shape[1], c0 + n)
        if rr1 > rr0 and cc1 > cc0:
            sub[rr0 - r0: rr1 - r0, cc0 - c0: cc1 - c0] = pred[rr0:rr1, cc0:cc1]
        crop = draw_mask_contours(crop, sub, (255, 0, 0), 2)
        st.image(crop, caption=f"Lesion {lid}: {L.category}, {L.max_diameter_mm:.2f} mm "
                               f"(view at {um_px:.2f} µm/px, predicted boundary in red)",
                 width="stretch")
    else:
        st.success("No lesions above threshold.")

with tabs[4]:
    k = st.slider("Show top-k most suspicious patches", 4, 32, 12, 4)
    ctx = st.slider("Context (× patch size)", 1.0, 4.0, 2.0, 0.5)
    order = np.argsort(-a["tile_probs"])[:k]
    rd = WSIReader(res["slide"])
    fp = a["cls_footprint"]
    cols = st.columns(4)
    for i, j in enumerate(order):
        x, y = a["coords"][j, 0], a["coords"][j, 1]
        side = fp * ctx
        cx, cy = x + fp / 2 - side / 2, y + fp / 2 - side / 2
        patch = rd.read_at_downsample(int(cx), int(cy), 256, 256, side / 256)
        s0 = int(256 * (1 - 1 / ctx) / 2)
        cv2.rectangle(patch, (s0, s0), (256 - s0, 256 - s0), (0, 0, 0), 1)
        cols[i % 4].image(patch, caption=f"p = {a['tile_probs'][j]:.3f} @ ({x}, {y})", width="stretch")
    rd.close()
    st.caption("Black box = the region the classifier actually saw; the rest is surrounding context.")

with tabs[5]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patches classified", f"{bm['tiles_classified']:,}")
    c2.metric("Classifier throughput", f"{bm['classification_tiles_per_sec']:,.0f} patch/s")
    c3.metric("Peak RAM", f"{bm['peak_rss_mb']:,.0f} MB")
    c4.metric("Full-res RGB would need", f"{bm['full_res_rgb_would_need_gb']:,.1f} GB")
    st.bar_chart(pd.Series(bm["stage_seconds"], name="seconds"))
    st.json(bm)

with tabs[6]:
    rep_clean = {k: v for k, v in rep.items() if k != "_artefacts"}
    st.download_button("report.json", json.dumps(rep_clean, indent=2, default=str), "report.json", "application/json")
    if q["lesions"]:
        st.download_button("lesions.csv", pd.DataFrame(q["lesions"]).to_csv(index=False), "lesions.csv", "text/csv")
    gj = polygons_to_geojson(mask_to_polygons(pred, a["map_ds"]))
    st.download_button("predictions.geojson (QuPath)", json.dumps(gj), "predictions.geojson", "application/geo+json")
    buf = io.BytesIO(cv2.imencode(".png", (pred * 255).astype(np.uint8))[1].tobytes())
    st.download_button("tumor_mask.png", buf, "tumor_mask.png", "image/png")
    st.caption("GeoJSON coordinates are level-0 pixels; import into QuPath to review predictions on the full slide.")
