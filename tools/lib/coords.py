"""Canonical coordinate transforms (Python mirror of src/core/coords.ts).

Source of truth: data/world/world.json. Every stage (pipeline, Blender,
web 2D, web 3D) converts through these functions only.
"""
from .common import load_json, path

_W = load_json(path('data/world/world.json'))['coordinate_system']
MPP = _W['meters_per_pixel']
OX, OY = _W['origin_px']
IMG_W, IMG_H = 2000, 667


def px_to_world(px, py, elev=0.0):
    """-> (X east, Y up, Z south) metres, Y-up (three.js / glTF)."""
    return ((px - OX) * MPP, elev, (py - OY) * MPP)


def world_to_px(X, Z):
    return (X / MPP + OX, Z / MPP + OY)


def px_to_blender(px, py, elev=0.0):
    """-> Blender Z-up (bx east, by north, bz up)."""
    return ((px - OX) * MPP, -(py - OY) * MPP, elev)


def px_to_norm(px, py):
    return (px / IMG_W, py / IMG_H)


def px_to_leaflet(px, py):
    return (-py, px)
