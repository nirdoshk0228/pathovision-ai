"""Read/write pathologist annotations and convert them to masks.

CAMELYON16 ground truth is ASAP XML: polygons in level-0 coordinates grouped as
``_0`` / ``_1`` (metastasis) and ``_2`` (normal tissue cut out *inside* a tumour
outline). Exclusion polygons must be subtracted after the tumour polygons are
filled, otherwise those holes are wrongly counted as tumour.

Predictions can be exported the same way (ASAP XML) and as GeoJSON for QuPath,
so a pathologist can review model output in the tools they already use.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

EXCLUSION_GROUPS = {"_2", "exclusion", "non-tumor", "non_tumor", "normal", "none"}


@dataclass
class Polygon:
    coords: np.ndarray  # (N, 2) float level-0 x, y
    group: str = "_0"
    name: str = ""
    properties: dict = field(default_factory=dict)

    @property
    def is_exclusion(self) -> bool:
        return self.group.strip().lower() in EXCLUSION_GROUPS

    def bounds(self) -> Tuple[float, float, float, float]:
        x0, y0 = self.coords.min(axis=0)
        x1, y1 = self.coords.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


def load_asap_xml(path: str) -> List[Polygon]:
    root = ET.parse(path).getroot()
    polys: List[Polygon] = []
    for ann in root.iter("Annotation"):
        pts = []
        for c in ann.iter("Coordinate"):
            order = int(float(c.get("Order", len(pts))))
            pts.append((order, float(c.get("X").replace(",", ".")), float(c.get("Y").replace(",", "."))))
        if len(pts) < 3:
            continue
        pts.sort(key=lambda t: t[0])
        coords = np.array([[p[1], p[2]] for p in pts], np.float64)
        polys.append(Polygon(coords, group=ann.get("PartOfGroup", "_0"), name=ann.get("Name", "")))
    return polys


def save_asap_xml(polys: Sequence[Polygon], path: str) -> None:
    root = ET.Element("ASAP_Annotations")
    anns = ET.SubElement(root, "Annotations")
    groups = sorted({p.group for p in polys}) or ["_0"]
    for i, p in enumerate(polys):
        a = ET.SubElement(anns, "Annotation", Name=p.name or f"Annotation {i}", Type="Polygon",
                          PartOfGroup=p.group, Color="#F4FA58" if not p.is_exclusion else "#64FE2E")
        cs = ET.SubElement(a, "Coordinates")
        for j, (x, y) in enumerate(p.coords):
            ET.SubElement(cs, "Coordinate", Order=str(j), X=f"{x:.2f}", Y=f"{y:.2f}")
    g = ET.SubElement(root, "AnnotationGroups")
    for name in groups:
        ET.SubElement(g, "Group", Name=name, PartOfGroup="None", Color="#FF0000")
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")


def rasterize(
    polys: Iterable[Polygon],
    out_hw: Tuple[int, int],
    downsample: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Binary tumour mask of shape ``out_hw``.

    Pixel (r, c) of the output covers level-0 region
    ``offset + (c, r) * downsample``. Use ``offset`` = patch top-left and
    ``downsample`` = level-0 px per output px to get a patch-aligned mask.
    """
    h, w = out_hw
    mask = np.zeros((h, w), np.uint8)
    polys = list(polys)
    ox, oy = offset
    win = (ox, oy, ox + w * downsample, oy + h * downsample)

    def _draw(sel: List[Polygon], value: int) -> None:
        pts = []
        for p in sel:
            x0, y0, x1, y1 = p.bounds()
            if x1 < win[0] or y1 < win[1] or x0 > win[2] or y0 > win[3]:
                continue  # cheap bounding-box rejection
            # output pixel i spans level-0 [i*ds, (i+1)*ds) and OpenCV puts its centre at i, hence -0.5
            q = (p.coords - np.array([ox, oy])) / downsample - 0.5
            pts.append(np.round(q * 16).astype(np.int32))  # 4 bits sub-pixel precision
        if pts:
            cv2.fillPoly(mask, pts, value, lineType=cv2.LINE_8, shift=4)

    _draw([p for p in polys if not p.is_exclusion], 1)
    _draw([p for p in polys if p.is_exclusion], 0)
    return mask.astype(bool)


def mask_to_polygons(
    mask: np.ndarray,
    downsample: float,
    min_area_px: int = 4,
    simplify_px: float = 1.0,
    group: str = "_0",
) -> List[Polygon]:
    """Contours of a low-resolution binary mask, scaled back to level-0 coords.

    Holes inside lesions become exclusion polygons so the round trip
    mask -> polygons -> rasterize is faithful.
    """
    from pathovision.wsi.tissue import fill_holes

    m = mask.astype(bool)
    holes = fill_holes(m) & ~m
    out: List[Polygon] = []
    # Outer boundaries of the mask, then boundaries of the hole regions themselves
    # (OpenCV's own hole contours trace the surrounding foreground ring instead).
    for src, grp in ((m, group), (holes, "_2")):
        contours, _ = cv2.findContours(src.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area_px:
                continue
            if simplify_px > 0:
                cnt = cv2.approxPolyDP(cnt, simplify_px, True)
            if len(cnt) < 3:
                continue
            # contour vertices are pixel centres -> level-0 centre of pixel i is (i + 0.5) * ds
            coords = (cnt.reshape(-1, 2).astype(np.float64) + 0.5) * downsample
            out.append(Polygon(coords, group=grp, name=f"pred_{len(out)}"))
    return out


def polygons_to_geojson(polys: Sequence[Polygon], properties: Optional[List[dict]] = None) -> dict:
    """GeoJSON FeatureCollection (QuPath-compatible) in level-0 pixel coordinates."""
    feats = []
    for i, p in enumerate(polys):
        ring = p.coords.tolist()
        ring.append(ring[0])
        props = {"objectType": "annotation",
                 "classification": {"name": "Exclusion" if p.is_exclusion else "Tumor",
                                    "color": [100, 254, 46] if p.is_exclusion else [255, 0, 0]}}
        props.update(p.properties)
        if properties and i < len(properties):
            props.update(properties[i])
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def save_geojson(polys: Sequence[Polygon], path: str) -> None:
    with open(path, "w") as f:
        json.dump(polygons_to_geojson(polys), f)
