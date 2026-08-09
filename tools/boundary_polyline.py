from collections import deque

import cv2
import numpy as np


def select_boundary_component(prob, threshold):
    mask = (prob >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    if num_labels <= 1:
        return mask, None, components

    width = mask.shape[1]
    best_label = None
    best_score = -1.0
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        area = float(stats[label_id, cv2.CC_STAT_AREA])
        right_score = float(xs.mean()) / max(width - 1, 1)
        score = area * (0.7 + 0.3 * right_score)
        component = {
            "label": int(label_id),
            "area": area,
            "mean_x": float(xs.mean()),
            "mean_y": float(ys.mean()),
            "min_y": int(ys.min()),
            "max_y": int(ys.max()),
            "score": float(score),
        }
        components.append(component)
        if score > best_score:
            best_score = score
            best_label = label_id

    if best_label is None:
        return mask, None, components
    return mask, (labels == best_label).astype(np.uint8), components


def zhang_suen_thinning(binary):
    img = (binary > 0).astype(np.uint8)
    if img.sum() == 0:
        return img

    changed = True
    max_iter = 100
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        p2 = img[:-2, 1:-1]
        p3 = img[:-2, 2:]
        p4 = img[1:-1, 2:]
        p5 = img[2:, 2:]
        p6 = img[2:, 1:-1]
        p7 = img[2:, :-2]
        p8 = img[1:-1, :-2]
        p9 = img[:-2, :-2]
        p1 = img[1:-1, 1:-1]

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )

        marker = (
            (p1 == 1)
            & (b >= 2)
            & (b <= 6)
            & (a == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if marker.any():
            img[1:-1, 1:-1][marker] = 0
            changed = True

        p2 = img[:-2, 1:-1]
        p3 = img[:-2, 2:]
        p4 = img[1:-1, 2:]
        p5 = img[2:, 2:]
        p6 = img[2:, 1:-1]
        p7 = img[2:, :-2]
        p8 = img[1:-1, :-2]
        p9 = img[:-2, :-2]
        p1 = img[1:-1, 1:-1]

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )
        marker = (
            (p1 == 1)
            & (b >= 2)
            & (b <= 6)
            & (a == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if marker.any():
            img[1:-1, 1:-1][marker] = 0
            changed = True

    return img


def _neighbors(pixel, pixels):
    y, x = pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = (y + dy, x + dx)
            if nb in pixels:
                yield nb


def _connected_components_pixels(pixels):
    unseen = set(pixels)
    comps = []
    while unseen:
        start = unseen.pop()
        comp = [start]
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in _neighbors(cur, unseen):
                unseen.remove(nb)
                q.append(nb)
                comp.append(nb)
        comps.append(comp)
    return comps


def _bfs_farthest(start, pixels):
    q = deque([start])
    parent = {start: None}
    dist = {start: 0}
    farthest = start
    while q:
        cur = q.popleft()
        if dist[cur] > dist[farthest]:
            farthest = cur
        for nb in _neighbors(cur, pixels):
            if nb in parent:
                continue
            parent[nb] = cur
            dist[nb] = dist[cur] + 1
            q.append(nb)
    return farthest, parent, dist


def longest_skeleton_path(skeleton):
    coords = list(map(tuple, np.argwhere(skeleton > 0)))
    if len(coords) < 2:
        return []

    best_path = []
    for comp in _connected_components_pixels(coords):
        pixels = set(comp)
        if len(pixels) < 2:
            continue
        degrees = {p: sum(1 for _ in _neighbors(p, pixels)) for p in pixels}
        starts = [p for p, deg in degrees.items() if deg <= 1]
        start = starts[0] if starts else comp[0]
        a, _, _ = _bfs_farthest(start, pixels)
        b, parent, _ = _bfs_farthest(a, pixels)
        path = []
        cur = b
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path = path[::-1]
        if len(path) > len(best_path):
            best_path = path

    return [[float(x), float(y)] for y, x in best_path]


def pca_centerline(component, max_points):
    ys, xs = np.where(component > 0)
    if len(xs) < 2:
        return []
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = coords.mean(axis=0)
    shifted = coords - center
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    direction = vh[0]
    t = shifted @ direction
    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
    bins = max(max_points * 3, 24)
    edges = np.linspace(float(t.min()), float(t.max()), bins + 1)
    points = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((t >= lo) & (t <= hi))[0]
        if len(idx) == 0:
            continue
        # Pick the most central pixel in this slice, not the arithmetic mean.
        best = idx[np.argmax(dist[ys[idx], xs[idx]])]
        points.append([float(xs[best]), float(ys[best])])
    return sample_polyline_by_arclength(points, max_points)


def sample_polyline_by_arclength(points, max_points):
    if len(points) <= max_points:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in points]
    pts = np.asarray(points, dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(length[-1])
    if total <= 1e-6:
        return [[round(float(pts[0, 0]), 2), round(float(pts[0, 1]), 2)]]
    targets = np.linspace(0, total, max_points)
    sampled = []
    for target in targets:
        idx = int(np.searchsorted(length, target, side="right") - 1)
        idx = max(0, min(idx, len(seg) - 1))
        denom = seg[idx] if seg[idx] > 1e-6 else 1.0
        ratio = (target - length[idx]) / denom
        point = pts[idx] * (1 - ratio) + pts[idx + 1] * ratio
        sampled.append([round(float(point[0]), 2), round(float(point[1]), 2)])
    deduped = []
    for x, y in sampled:
        if deduped and abs(deduped[-1][0] - x) < 1e-3 and abs(deduped[-1][1] - y) < 1e-3:
            continue
        deduped.append([x, y])
    return deduped


def component_to_skeleton_polyline(component, max_points):
    if component is None or component.sum() == 0:
        return []
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    skeleton = zhang_suen_thinning(cleaned)
    path = longest_skeleton_path(skeleton)
    if len(path) < 2:
        return pca_centerline(cleaned, max_points)
    return sample_polyline_by_arclength(path, max_points)


def mask_to_polyline(prob, threshold, max_points):
    mask, component, components = select_boundary_component(prob, threshold)
    line = component_to_skeleton_polyline(component, max_points)
    return mask, component, line, components
