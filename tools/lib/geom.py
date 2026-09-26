"""Geometry helpers: mask <-> polygon conversion, GeoJSON builders."""
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString, mapping, shape
from shapely.ops import unary_union
from .common import rnd

UPS = 4  # supersampling factor for mask vectorisation


def mask_to_polygons(mask, blur=0.9, simplify=0.35, min_area=4.0):
    """Vectorise a boolean mask (source pixel grid) into smooth shapely polygons.

    Pixel (i, j) covers [i, i+1) x [j, j+1) in continuous coordinates.
    """
    m = mask.astype(np.float32)
    big = cv2.resize(m, (m.shape[1] * UPS, m.shape[0] * UPS), interpolation=cv2.INTER_NEAREST)
    if blur > 0:
        big = cv2.GaussianBlur(big, (0, 0), blur * UPS)
    b = (big >= 0.5).astype(np.uint8)
    cnts, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    if hier is None:
        return []
    hier = hier[0]
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:
            continue  # hole; handled by parent
        shell = (c[:, 0, :].astype(float) + 0.5) / UPS
        holes = []
        k = hier[i][2]
        while k != -1:
            h = (cnts[k][:, 0, :].astype(float) + 0.5) / UPS
            if len(h) >= 4:
                holes.append(h)
            k = hier[k][0]
        if len(shell) < 4:
            continue
        p = Polygon(shell, holes).buffer(0)
        if p.area < min_area:
            continue
        p = p.simplify(simplify, preserve_topology=True)
        if p.is_empty:
            continue
        if isinstance(p, MultiPolygon):
            polys += [q for q in p.geoms if q.area >= min_area]
        else:
            polys.append(p)
    return polys


def poly_coords(p, n=2):
    ext = [[rnd(x, n), rnd(y, n)] for x, y in p.exterior.coords]
    rings = [ext]
    for r in p.interiors:
        rings.append([[rnd(x, n), rnd(y, n)] for x, y in r.coords])
    return rings


def geojson_polygon(p, props, n=2):
    if isinstance(p, MultiPolygon):
        return {'type': 'Feature', 'properties': props,
                'geometry': {'type': 'MultiPolygon', 'coordinates': [poly_coords(q, n) for q in p.geoms]}}
    return {'type': 'Feature', 'properties': props, 'geometry': {'type': 'Polygon', 'coordinates': poly_coords(p, n)}}


def geojson_line(pts, props, n=2):
    return {'type': 'Feature', 'properties': props,
            'geometry': {'type': 'LineString', 'coordinates': [[rnd(x, n), rnd(y, n)] for x, y in pts]}}


def geojson_point(p, props, n=2):
    return {'type': 'Feature', 'properties': props, 'geometry': {'type': 'Point', 'coordinates': [rnd(p[0], n), rnd(p[1], n)]}}


def fc(features, name, extra=None):
    d = {'type': 'FeatureCollection', 'name': name,
         'crs_note': 'coordinates are source-image pixels of assets/maps/source/base_map.webp (x right, y down); see docs/COORDINATES.md',
         'features': features}
    if extra:
        d.update(extra)
    return d


def rasterize_polys(polys, shape_hw, value=1, ups=1):
    """Rasterise shapely polygons into a uint8 mask of the source grid."""
    h, w = shape_hw
    out = np.zeros((h * ups, w * ups), np.uint8)
    for p in polys:
        geoms = p.geoms if isinstance(p, MultiPolygon) else [p]
        for g in geoms:
            ext = (np.asarray(g.exterior.coords) * ups - 0.5).round().astype(np.int32)
            cv2.fillPoly(out, [ext], value)
            for r in g.interiors:
                cv2.fillPoly(out, [(np.asarray(r.coords) * ups - 0.5).round().astype(np.int32)], 0)
    if ups > 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    return out


def polyline_length(pts):
    pts = np.asarray(pts, float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
