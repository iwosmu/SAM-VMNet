"""
Stenosis Detection Pipeline - Python port of MATLAB code.
Detects coronary artery stenosis from vessel segmentation masks.

Pipeline: Skeleton extraction → Radius measurement → Bifurcation detection →
          Path finding → V-shape narrowing detection → Severity classification
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from collections import deque
import matplotlib.pyplot as plt


def mom_for_seg(x0, y0, rr, image):
    """Measure vessel radius at (x0,y0) by ray-casting outward."""
    alpha = np.arange(0, 2 * np.pi, np.pi / 1800)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    h, w = image.shape[:2]
    for i in np.arange(1, rr + 0.5, 0.5):
        x = x0 + i * cos_a
        y = y0 + i * sin_a
        for j in range(len(alpha)):
            rx, ry = int(round(x[j])), int(round(y[j]))
            if 0 <= rx < h and 0 <= ry < w:
                if image[rx, ry] != 255:
                    return abs(i)
        # If any ray hit boundary in this radius, we already returned
    return 100.0


def check_neighbors(BW, y, x):
    """Find 8-connected skeleton neighbors of point (y,x)."""
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    h, w = BW.shape
    neighbors = []
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and BW[ny, nx]:
            neighbors.append((ny, nx))
    return neighbors


def get_neighbors(BW, current):
    """Get 8-connected skeleton neighbors for pathfinding."""
    return check_neighbors(BW, current[0], current[1])


def find_path(BW, start, goal):
    """BFS shortest path through skeleton. start/goal are (col, row) format."""
    # Flip to (row, col) for internal use
    start_rc = (start[1], start[0])
    goal_rc = (goal[1], goal[0])
    h, w = BW.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w, 2), -1, dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)

    dist[start_rc[0], start_rc[1]] = 0
    visited[start_rc[0], start_rc[1]] = True
    queue = deque([start_rc])

    while queue:
        cr, cc = queue.popleft()
        if (cr, cc) == goal_rc:
            break
        for nr, nc in get_neighbors(BW, (cr, cc)):
            if not visited[nr, nc]:
                visited[nr, nc] = True
                new_dist = dist[cr, cc] + 1
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    prev[nr, nc] = [cr, cc]
                    queue.append((nr, nc))

    if np.isinf(dist[goal_rc[0], goal_rc[1]]):
        raise ValueError(f"No path from {start} to {goal}")

    # Backtrack
    path = []
    cur = goal_rc
    while cur != start_rc:
        path.append(cur)
        r, c = cur
        cur = (int(prev[r, c, 0]), int(prev[r, c, 1]))
    path.append(start_rc)
    path.reverse()
    return np.array(path), len(path)


def build_radius_lookup(point_data):
    """Build a dict for O(1) radius lookup."""
    return {(p['x'], p['y']): p['radius'] for p in point_data}


def get_radius(lookup, point):
    """Get radius for a point (row, col)."""
    return lookup.get((point[0], point[1]), np.nan)


def detect_v_shapes(shortest_path, radius_lookup):
    """Find V-shaped narrowing patterns along a path (recursive→iterative)."""
    queue = []
    i = 0
    n = len(shortest_path)
    while i < n - 1:
        # Phase 1: find first decreasing point
        found = False
        while i < n - 1:
            cr = get_radius(radius_lookup, tuple(shortest_path[i]))
            nr = get_radius(radius_lookup, tuple(shortest_path[i + 1]))
            if cr > nr:
                queue.append(tuple(shortest_path[i]))
                found = True
                break
            i += 1
        if not found:
            break
        # Phase 2: find V bottom
        found = False
        while i < n - 1:
            cr = get_radius(radius_lookup, tuple(shortest_path[i]))
            nr = get_radius(radius_lookup, tuple(shortest_path[i + 1]))
            if cr >= nr:
                i += 1
            else:
                queue.append(tuple(shortest_path[i]))
                found = True
                break
        if not found:
            break
        # Phase 3: find last increasing point
        found = False
        while i < n - 1:
            cr = get_radius(radius_lookup, tuple(shortest_path[i]))
            nr = get_radius(radius_lookup, tuple(shortest_path[i + 1]))
            if cr <= nr:
                i += 1
            else:
                queue.append(tuple(shortest_path[i]))
                found = True
                break
        if not found:
            break
    return queue


def run_stenosis_detection(original_img_path, mask_img_path, target_size=(800, 600)):
    """
    Main stenosis detection pipeline.

    Args:
        original_img_path: Path to original angiography image
        mask_img_path: Path to segmented vessel mask
        target_size: (height, width) to resize to

    Returns:
        dict with 'points', 'degrees', 'skeleton', 'bifurcations', 'mask'
    """
    # --- 1. Preprocessing ---
    Im = cv2.imread(original_img_path)
    Im = cv2.resize(Im, (target_size[1], target_size[0]))

    im = cv2.imread(mask_img_path)
    im = cv2.resize(im, (target_size[1], target_size[0]))
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # --- 2. Skeleton extraction ---
    binary = (im > 127).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    m_coords, n_coords = np.where(skeleton == 1)  # row, col
    print(f"  Skeleton points: {len(m_coords)}")

    # --- 3. Radius calculation ---
    point_data = []
    for j in range(len(m_coords)):
        x0, y0 = int(m_coords[j]), int(n_coords[j])
        r = mom_for_seg(x0, y0, 110, im)
        point_data.append({'x': x0, 'y': y0, 'radius': r})
    radius_lookup = build_radius_lookup(point_data)
    print(f"  Radii computed: {len(point_data)}")

    # --- 4. Bifurcation detection ---
    seg_points = []
    for i in range(len(m_coords)):
        nbrs = check_neighbors(skeleton, int(m_coords[i]), int(n_coords[i]))
        if len(nbrs) == 3:
            seg_points.append((int(n_coords[i]), int(m_coords[i])))  # (col, row)

    seg_points = np.array(seg_points) if seg_points else np.empty((0, 2), dtype=int)
    print(f"  Bifurcation points: {len(seg_points)}")

    # --- 5. Filter close bifurcation points ---
    if len(seg_points) > 1:
        dist_matrix = cdist(seg_points, seg_points)
        keep = np.ones(len(seg_points), dtype=bool)
        for i in range(len(seg_points)):
            for j in range(i + 1, len(seg_points)):
                if dist_matrix[i, j] < 8:
                    keep[j] = False
        final_seg = seg_points[keep]
    else:
        final_seg = seg_points
    print(f"  Filtered bifurcations: {len(final_seg)}")

    # --- 6. Path finding & stenosis detection ---
    all_stenosis_points = []
    all_stenosis_degrees = []

    for i in range(len(final_seg) - 1):
        start_pt = final_seg[i]
        end_pt = final_seg[i + 1]
        try:
            path, path_len = find_path(skeleton, start_pt, end_pt)
        except ValueError:
            continue

        # Average radius along path
        avg_r = np.nanmean([get_radius(radius_lookup, tuple(p)) for p in path])

        # Detect V-shapes
        v_queue = detect_v_shapes(path, radius_lookup)

        # Calculate stenosis degree for each V-shape triplet
        for k in range(1, len(v_queue) - 1, 3):
            r_mid = get_radius(radius_lookup, v_queue[k])
            r_prev = get_radius(radius_lookup, v_queue[k - 1])
            r_next = get_radius(radius_lookup, v_queue[k + 1]) if k + 1 < len(v_queue) else np.nan
            if np.isnan(r_prev) or np.isnan(r_next) or (r_prev + r_next) == 0:
                continue
            nn = 1 - 2 * r_mid / (r_prev + r_next)
            if nn > 0.25 and avg_r > 4:
                all_stenosis_points.append(v_queue[k])
                all_stenosis_degrees.append(nn)

    # --- 7. Deduplicate close stenosis points ---
    if all_stenosis_points:
        pts = np.array(all_stenosis_points)
        degs = np.array(all_stenosis_degrees)
        # Sort by first coordinate (col originally, now row)
        pts_flip = pts[:, ::-1]
        order = np.lexsort((pts_flip[:, 1], pts_flip[:, 0]))
        pts_flip = pts_flip[order]
        degs = degs[order]

        keep = np.ones(len(pts_flip), dtype=bool)
        for i in range(len(pts_flip) - 1):
            if abs(pts_flip[i, 0] - pts_flip[i + 1, 0]) < 10:
                keep[i + 1] = False
        pts_flip = pts_flip[keep]
        degs = degs[keep]
    else:
        pts_flip = np.empty((0, 2))
        degs = np.array([])

    print(f"  Stenosis points found: {len(degs)}")
    for i, d in enumerate(degs):
        severity = "SEVERE" if d > 0.75 else "MODERATE" if d > 0.5 else "MILD"
        print(f"    [{severity}] Point ({pts_flip[i,0]:.0f}, {pts_flip[i,1]:.0f}): {d*100:.1f}%")

    return {
        'points': pts_flip,
        'degrees': degs,
        'skeleton': skeleton,
        'skeleton_coords': (m_coords, n_coords),
        'bifurcations': final_seg,
        'mask': im,
        'original': Im,
    }


def plot_results(result, save_path=None):
    """Visualize stenosis detection results."""
    im = result['mask']
    m, n = result['skeleton_coords']
    bif = result['bifurcations']
    pts = result['points']
    degs = result['degrees']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Centerline
    axes[0].imshow(im, cmap='gray')
    axes[0].plot(n, m, 'r.', markersize=1)
    axes[0].set_title('Centerline Extraction')
    axes[0].axis('off')

    # Plot 2: Bifurcation points
    axes[1].imshow(im, cmap='gray')
    axes[1].plot(n, m, 'r.', markersize=1)
    if len(bif) > 0:
        axes[1].plot(bif[:, 0], bif[:, 1], 'mo', markersize=5)
    axes[1].set_title('Bifurcation Points')
    axes[1].axis('off')

    # Plot 3: Stenosis results
    axes[2].imshow(im, cmap='gray')
    for i in range(len(pts)):
        sx, sy = pts[i, 0], pts[i, 1]
        d = degs[i]
        color = 'r' if d > 0.75 else 'g' if d > 0.5 else 'b'
        axes[2].plot(sy, sx, 'o', markeredgecolor=color, markerfacecolor='none',
                     markersize=11, markeredgewidth=2)
    # Legend
    axes[2].plot([], [], 'ro', markersize=10, label='>75% Severe')
    axes[2].plot([], [], 'go', markersize=10, label='50-75% Moderate')
    axes[2].plot([], [], 'bo', markersize=10, label='25-50% Mild')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_title('Stenosis Detection')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
