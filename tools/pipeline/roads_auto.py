"""Automatic road-centreline graph from the base map.

thin roads : hysteresis on the road-likeness field (colour x bright-ridge)
wide roads : opened pavement mask (>= ~5 px wide), fills highway lane markings
The union is skeletonised, converted to a graph, pruned and smoothed.
Output: tools/.cache/roads_auto.json  {nodes:[[x,y]], edges:[{a,b,pts}]}
This is an intermediate; tools/pipeline/roads.py curates it into the final network.
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
import networkx as nx
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize, remove_small_objects
from skan import Skeleton

from tools.lib.common import path, W, H, save_json
from tools.lib.features import road_prob, hsv, water_mask
from tools.lib.trace import pave_dt, smooth_polyline, rdp


def build_mask():
    p = road_prob()
    thin = apply_hysteresis_threshold(p, 0.16, 0.36)
    dt = pave_dt()
    wide = dt >= 2.2
    # grow wide cores back to full road width
    wide = cv2.dilate(wide.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool) & (dt > 0)
    m = thin | wide
    wm = np.load(path('tools/.cache/water_mask.npy'))
    m &= ~wm
    m = remove_small_objects(m, max_size=20)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    # blobs wider than ~14 px are lots/plazas/roofs: exclude their cores
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    blob = cv2.dilate((d > 7).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    return m, blob


def skeleton_graph(sk):
    S = Skeleton(sk)
    G = nx.MultiGraph()
    coords = S.coordinates
    for i in range(S.n_paths):
        idx = S.path(i)
        pts = coords[idx][:, ::-1] + 0.5  # (x, y)
        a, b = int(idx[0]), int(idx[-1])
        G.add_node(a, xy=tuple(pts[0]))
        G.add_node(b, xy=tuple(pts[-1]))
        G.add_edge(a, b, pts=pts)
    return G


def length(pts):
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def prune(G, spur=8.0, rounds=4):
    for _ in range(rounds):
        rm = []
        for u, v, k, d in G.edges(keys=True, data=True):
            du, dv = G.degree(u), G.degree(v)
            if u == v:
                if length(d['pts']) < 12:
                    rm.append((u, v, k))
                continue
            if (du == 1) != (dv == 1) and length(d['pts']) < spur:
                rm.append((u, v, k))
        if not rm:
            break
        G.remove_edges_from(rm)
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
        merge_deg2(G)
    return G


def merge_deg2(G):
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if n not in G or G.degree(n) != 2:
                continue
            es = list(G.edges(n, keys=True, data=True))
            if len(es) != 2:
                continue
            (u1, v1, k1, d1), (u2, v2, k2, d2) = es
            o1 = v1 if u1 == n else u1
            o2 = v2 if u2 == n else u2
            if o1 == n or o2 == n:
                continue
            p1 = orient(d1['pts'], G.nodes[o1]['xy'], G.nodes[n]['xy'])
            p2 = orient(d2['pts'], G.nodes[n]['xy'], G.nodes[o2]['xy'])
            G.remove_edge(u1, v1, k1)
            G.remove_edge(u2, v2, k2)
            G.remove_node(n)
            G.add_edge(o1, o2, pts=np.vstack([p1, p2[1:]]))
            changed = True


def orient(pts, a, b):
    pts = np.asarray(pts)
    if np.hypot(*(pts[0] - a)) <= np.hypot(*(pts[-1] - a)):
        return pts
    return pts[::-1]


def main():
    m, blob = build_mask()
    m2 = m & ~blob
    sk = skeletonize(m2)
    sk = remove_small_objects(sk, max_size=6, connectivity=2)
    G = skeleton_graph(sk)
    merge_deg2(G)
    prune(G, spur=7.0)
    # drop small isolated components
    for comp in list(nx.connected_components(G)):
        L = sum(length(d['pts']) for _, _, d in G.subgraph(comp).edges(data=True))
        if L < 30:
            G.remove_nodes_from(comp)
    nodes = {}
    out_nodes, out_edges = [], []
    for n in G.nodes:
        nodes[n] = len(out_nodes)
        out_nodes.append([round(G.nodes[n]['xy'][0], 2), round(G.nodes[n]['xy'][1], 2)])
    for u, v, d in G.edges(data=True):
        pts = orient(d['pts'], G.nodes[u]['xy'], G.nodes[v]['xy'])
        out_edges.append({'a': nodes[u], 'b': nodes[v], 'pts': [[round(x, 2), round(y, 2)] for x, y in pts]})
    save_json(path('tools/.cache/roads_auto.json'), {'nodes': out_nodes, 'edges': out_edges})
    np.save(path('tools/.cache/road_mask.npy'), m)
    tot = sum(length(np.array(e['pts'])) for e in out_edges)
    print(f'auto graph: {len(out_nodes)} nodes, {len(out_edges)} edges, {tot:.0f} px total length')


if __name__ == '__main__':
    main()
