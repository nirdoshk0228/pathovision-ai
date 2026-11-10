"""REST API for slide analysis (FastAPI).

Whole-slide analysis takes seconds to minutes, so the API is asynchronous:
``POST /analyze`` returns a job id immediately, a single worker thread runs jobs
in order (one GPU, one model copy), and clients poll ``GET /jobs/{id}``.

    PV_CLS_CKPT=models/classifier.pt PV_SEG_CKPT=models/segmenter.pt \
    PV_SLIDE_DIR=data/slides uvicorn pathovision.api.main:app --port 8000

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from pathovision import __version__
from pathovision.inference.slide_inference import AnalysisConfig, SlideAnalyzer, load_classifier, load_segmenter
from pathovision.utils.common import get_device
from pathovision.wsi.reader import PYRAMID_EXTS

CLS_CKPT = os.environ.get("PV_CLS_CKPT", "models/classifier.pt")
SEG_CKPT = os.environ.get("PV_SEG_CKPT", "models/segmenter.pt")
SLIDE_DIR = os.environ.get("PV_SLIDE_DIR", "data/slides")
OUT_DIR = os.environ.get("PV_OUTPUT_DIR", "outputs/api")
UPLOAD_DIR = os.path.join(OUT_DIR, "_uploads")
ALLOWED_EXTS = PYRAMID_EXTS | {".png", ".jpg", ".jpeg"}
DOWNLOADABLE = {"report.json", "lesions.csv", "tiles.csv", "predictions.geojson", "predictions.xml",
                "heatmap_overlay.png", "prediction_overlay.png", "tissue_mask.png", "thumbnail.png", "tumor_mask.png"}

app = FastAPI(title="PathoVision AI", version=__version__,
              description="Whole-slide image analysis: tissue detection, tumour heatmaps, segmentation, quantification.")
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, dict] = {}
_lock = threading.Lock()
_models: Dict[str, object] = {}  # loaded once, shared by every job


def _get_analyzer(tta: int, use_seg: bool) -> SlideAnalyzer:
    if not _models:
        if not os.path.exists(CLS_CKPT):
            raise RuntimeError(f"classifier checkpoint not found: {CLS_CKPT}")
        device = get_device()
        _models["cls"] = load_classifier(CLS_CKPT, device)
        _models["seg"] = load_segmenter(SEG_CKPT, device) if os.path.exists(SEG_CKPT) else None
    cfg = AnalysisConfig(tta=tta, num_workers=0)
    return SlideAnalyzer(cfg, classifier=_models["cls"], segmenter=_models["seg"] if use_seg else None)


def _update(job_id: str, **kw) -> None:
    with _lock:
        _jobs[job_id].update(kw)


def _run_job(job_id: str, slide_path: str, tta: int, use_seg: bool, gt_xml: Optional[str]) -> None:
    _update(job_id, status="running", started_at=time.time())
    try:
        analyzer = _get_analyzer(tta, use_seg)
        out = os.path.join(OUT_DIR, job_id)
        rep = analyzer.analyze(slide_path, out, gt_xml,
                               progress=lambda stage, frac: _update(job_id, stage=stage, progress=round(frac, 3)))
        rep.pop("_artefacts", None)
        q = rep["quantification"]
        summary = {"predicted_label": rep["slide_score"]["predicted_label"],
                   "max_tile_prob": rep["slide_score"]["max_tile_prob"],
                   "tumor_burden_pct": q["tumor_burden_pct"], "tumor_area_mm2": q["tumor_area_mm2"],
                   "n_lesions": q["n_lesions"], "largest_lesion_diameter_mm": q["largest_lesion_diameter_mm"],
                   "slide_category": q["slide_category"], "total_seconds": rep["benchmark"]["total_seconds"]}
        _update(job_id, status="done", progress=1.0, summary=summary, finished_at=time.time(),
                files=sorted(f for f in os.listdir(out) if f in DOWNLOADABLE))
    except Exception as e:  # report failures to the client instead of losing them
        _update(job_id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "classifier": os.path.exists(CLS_CKPT),
            "segmenter": os.path.exists(SEG_CKPT)}


@app.get("/slides")
def list_slides():
    if not os.path.isdir(SLIDE_DIR):
        return {"slides": []}
    return {"slides": sorted(f for f in os.listdir(SLIDE_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS)}


@app.post("/analyze")
async def analyze(file: Optional[UploadFile] = File(None), slide_name: Optional[str] = Form(None),
                  tta: int = Form(1), segmentation: bool = Form(True)):
    """Analyse an uploaded slide, or one already on the server (``slide_name`` in PV_SLIDE_DIR).

    Real WSIs are 1-3 GB, so in practice slides live on shared storage and are
    referenced by name; upload is meant for small images and demos.
    """
    if tta not in (1, 2, 4, 8):
        raise HTTPException(400, "tta must be 1, 2, 4 or 8")
    gt = None
    if file is not None:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(400, f"unsupported file type {ext}")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        slide_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        with open(slide_path, "wb") as f:
            shutil.copyfileobj(file.file, f, length=8 << 20)  # stream to disk in 8 MB chunks
        name = file.filename
    elif slide_name:
        safe = os.path.basename(slide_name)  # no path traversal
        slide_path = os.path.join(SLIDE_DIR, safe)
        if not os.path.exists(slide_path):
            raise HTTPException(404, f"slide not found: {safe}")
        xml = os.path.join(SLIDE_DIR, os.path.splitext(safe)[0] + ".xml")
        gt = xml if os.path.exists(xml) else None
        name = safe
    else:
        raise HTTPException(400, "provide a file or slide_name")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "slide": name, "status": "queued", "stage": None, "progress": 0.0,
                         "created_at": time.time()}
    _executor.submit(_run_job, job_id, slide_path, tta, segmentation, gt)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.get("/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    if job_id not in _jobs or name not in DOWNLOADABLE:
        raise HTTPException(404, "not found")
    path = os.path.join(OUT_DIR, job_id, name)
    if not os.path.exists(path):
        raise HTTPException(404, "not ready")
    return FileResponse(path, filename=name)
