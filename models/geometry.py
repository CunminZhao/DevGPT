import math
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import marching_cubes


EPS = 1e-12

GEOMETRY_FEATURE_NAMES = [
    "log_volume_cell",
    "log_area_cell",
    "area_norm_cell",
    "sphericity_cell",
    "elongation_ratio_cell",
    "pivotability_index_cell",
    "flatness_cell",
    "corey_shape_factor_cell",
    "rel_pos_x",
    "rel_pos_y",
    "rel_pos_z",
    "rel_pos_norm_x",
    "rel_pos_norm_y",
    "rel_pos_norm_z",
    "contact_area_fraction",
    "contact_area_norm",
    "contact_anisotropy",
    "contact_spread",
    "huang_shape_factor_cell",
    "convex_spreading_index_cell",
]


def compute_geometry_features(data, voxel_size, surface_level, boundary_step_size):
    """Compute the NPME shallow geometry vector from a 3-class voxel map.

    The draft defines 16 semantic geometry groups. The two relative-position
    groups are 3D vectors, so the flattened vector used by the MLP has 20
    scalar values. The returned values are bounded into [-pi, pi].
    """
    if data.ndim != 3:
        raise ValueError(f"geometry input must be a 3D array, got ndim={data.ndim}")

    target_mask = data == 2
    embryo_mask = data != 0
    environment_mask = data == 1

    voxel_volume = float(voxel_size) ** 3
    voxel_area = float(voxel_size) ** 2
    volume_cell = float(np.count_nonzero(target_mask) * voxel_volume)

    surface = _surface_stats(_surface_point_cloud_from_mask(target_mask, surface_level))
    area_cell = float(surface["area"])
    convex_volume = float(surface["convex_volume"])
    convex_surface = float(surface["convex_surface"])
    a, b, c = surface["axis_lengths"]

    c_cell = _centroid_zyx(target_mask)
    c_embryo = _centroid_zyx(embryo_mask)
    if np.isnan(c_cell).any() or np.isnan(c_embryo).any():
        rel_pos_zyx = np.full(3, np.nan, dtype=float)
        rel_pos_norm_zyx = np.full(3, np.nan, dtype=float)
    else:
        rel_pos_zyx = c_cell - c_embryo
        boundary_len = _boundary_length_along_vector(embryo_mask, c_embryo, rel_pos_zyx, boundary_step_size)
        if np.isnan(boundary_len) or abs(boundary_len) < EPS:
            rel_pos_norm_zyx = np.full(3, np.nan, dtype=float)
        else:
            rel_pos_norm_zyx = rel_pos_zyx / (boundary_len + EPS)

    contact = _contact_stats(
        target_mask=target_mask,
        environment_mask=environment_mask,
        volume_cell=volume_cell,
        area_cell=area_cell,
        voxel_area=voxel_area,
    )

    raw = np.asarray(
        [
            math.log(volume_cell + EPS),
            math.log(area_cell + EPS),
            _safe_ratio(area_cell, _safe_pow(volume_cell, 2.0 / 3.0) + EPS),
            _safe_ratio(_safe_pow(36.0 * math.pi * volume_cell * volume_cell, 1.0 / 3.0), area_cell + EPS),
            _safe_ratio(a, b + EPS),
            _safe_ratio(c, b + EPS),
            _safe_ratio(c, a + EPS),
            _safe_ratio(c, math.sqrt(max(a * b, 0.0)) + EPS),
            *_zyx_to_xyz(rel_pos_zyx),
            *_zyx_to_xyz(rel_pos_norm_zyx),
            contact["area_fraction"],
            contact["area_norm"],
            contact["anisotropy"],
            contact["spread"],
            _safe_ratio(b + c, 2.0 * a + EPS),
            _safe_ratio(
                _safe_pow(36.0 * math.pi * convex_volume * convex_volume, 1.0 / 3.0),
                convex_surface + EPS,
            ),
        ],
        dtype=np.float32,
    )

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return (math.pi * np.tanh(raw)).astype(np.float32)


def _centroid_zyx(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(3, np.nan, dtype=float)
    return coords.mean(axis=0).astype(float)


def _surface_point_cloud_from_mask(mask: np.ndarray, level: float) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return np.empty((0, 3), dtype=float)
    try:
        verts, _, _, _ = marching_cubes(mask.astype(np.float32), level=level)
    except ValueError:
        return np.empty((0, 3), dtype=float)
    return verts.astype(float)


def _surface_stats(points_zyx: np.ndarray) -> dict[str, float | np.ndarray]:
    area = float("nan")
    convex_volume = float("nan")
    convex_surface = float("nan")
    if points_zyx.shape[0] >= 4:
        try:
            hull = ConvexHull(points_zyx)
            area = float(hull.area)
            convex_surface = float(hull.area)
            convex_volume = float(hull.volume)
        except QhullError:
            pass

    axis_lengths = np.full(3, np.nan, dtype=float)
    if points_zyx.shape[0] >= 3:
        centered = points_zyx - points_zyx.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        axis_lengths = np.sqrt(np.clip(eigvals, a_min=0.0, a_max=None)).astype(float)

    return {
        "area": area,
        "convex_volume": convex_volume,
        "convex_surface": convex_surface,
        "axis_lengths": axis_lengths,
    }


def _contact_stats(
    target_mask: np.ndarray,
    environment_mask: np.ndarray,
    volume_cell: float,
    area_cell: float,
    voxel_area: float,
) -> dict[str, float]:
    contact_faces = 0
    contact_voxels = np.zeros_like(target_mask, dtype=bool)
    for axis in range(3):
        target_before = np.take(target_mask, range(target_mask.shape[axis] - 1), axis=axis)
        env_after = np.take(environment_mask, range(1, environment_mask.shape[axis]), axis=axis)
        contact = target_before & env_after
        contact_faces += int(np.count_nonzero(contact))
        _mark_contact_voxels(contact_voxels, contact, axis, 0)

        target_after = np.take(target_mask, range(1, target_mask.shape[axis]), axis=axis)
        env_before = np.take(environment_mask, range(environment_mask.shape[axis] - 1), axis=axis)
        contact = target_after & env_before
        contact_faces += int(np.count_nonzero(contact))
        _mark_contact_voxels(contact_voxels, contact, axis, 1)

    contact_area = float(contact_faces * voxel_area)
    contact_points = np.argwhere(contact_voxels).astype(float)
    anisotropy = float("nan")
    spread = float("nan")
    if contact_points.shape[0] >= 3:
        centered = contact_points - contact_points.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eta1, eta2, eta3 = np.clip(np.sort(np.linalg.eigvalsh(cov))[::-1], a_min=0.0, a_max=None)
        anisotropy = 1.0 - _safe_ratio(eta2 + eta3, 2.0 * eta1 + EPS)
        radius_cell = _safe_pow((3.0 * volume_cell) / (4.0 * math.pi + EPS), 1.0 / 3.0)
        spread = _safe_ratio(math.sqrt(max(float(np.trace(cov)), 0.0)), radius_cell + EPS)

    return {
        "area_fraction": _safe_ratio(contact_area, area_cell + EPS),
        "area_norm": _safe_ratio(contact_area, _safe_pow(volume_cell, 2.0 / 3.0) + EPS),
        "anisotropy": anisotropy,
        "spread": spread,
    }


def _mark_contact_voxels(output: np.ndarray, contact: np.ndarray, axis: int, offset: int) -> None:
    index = [slice(None)] * output.ndim
    index[axis] = slice(offset, offset + contact.shape[axis])
    output[tuple(index)] |= contact


def _boundary_length_along_vector(
    support_mask: np.ndarray,
    origin_zyx: np.ndarray,
    direction_zyx: np.ndarray,
    step_size: float,
) -> float:
    if np.count_nonzero(support_mask) == 0:
        return float("nan")

    origin = np.asarray(origin_zyx, dtype=float)
    direction = np.asarray(direction_zyx, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < EPS:
        return float("nan")
    unit = direction / (norm + EPS)

    if not _is_inside_nearest(origin, support_mask):
        nearest = _nearest_nonzero_voxel(origin, support_mask)
        if nearest is None:
            return float("nan")
        origin = nearest

    max_distance = max(1.0, 2.0 * float(np.linalg.norm(np.asarray(support_mask.shape, dtype=float))))
    last_inside_t = 0.0
    t = float(step_size)
    while t <= max_distance:
        probe = origin + unit * t
        if not _is_inside_nearest(probe, support_mask):
            low = last_inside_t
            high = t
            for _ in range(16):
                mid = 0.5 * (low + high)
                if _is_inside_nearest(origin + unit * mid, support_mask):
                    low = mid
                else:
                    high = mid
            return float(low)
        last_inside_t = t
        t += float(step_size)

    return float("nan")


def _is_inside_nearest(point_zyx: np.ndarray, mask: np.ndarray) -> bool:
    idx = np.round(point_zyx).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(mask.shape)):
        return False
    return bool(mask[tuple(idx)])


def _nearest_nonzero_voxel(point_zyx: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    nz = np.argwhere(mask)
    if nz.size == 0:
        return None
    distances = np.linalg.norm(nz.astype(float) - point_zyx[None, :], axis=1)
    return nz[np.argmin(distances)].astype(float)


def _zyx_to_xyz(vector_zyx: np.ndarray) -> np.ndarray:
    return np.asarray([vector_zyx[2], vector_zyx[1], vector_zyx[0]], dtype=float)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or abs(denominator) < EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_pow(value: float, exponent: float) -> float:
    if np.isnan(value) or value < 0:
        return float("nan")
    return float(value**exponent)
