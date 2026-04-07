# -*- coding: utf-8 -*-
# Scan-to-CAD Local Pipeline
# Converted from Colab notebook — only paths changed, all functionality preserved.

import subprocess, sys, os, json, time, types

sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ── Paths — must be defined first ─────────────────────────────────────────────
BASE_DIR   = os.path.expanduser('~/ScanToCAD')
PCD_PATH   = f'{BASE_DIR}/latest.pcd'
READY_PATH = f'{BASE_DIR}/output_ready.txt'
OUTPUT_DIR = f'{BASE_DIR}/cad_output'
XYZC_PATH  = f'{BASE_DIR}/scan_prediction.xyzc'
P2CAD_DIR  = f'{BASE_DIR}/point2cad'
SEG_DIR    = f'{BASE_DIR}/point2cad_out'
FLAG_PATH  = f'{BASE_DIR}/.deps_installed'
P2CAD_FLAG = f'{BASE_DIR}/.p2cad_deps_installed'
LOG_PATH   = f'{BASE_DIR}/pipeline.log'

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEG_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')

# ── Install dependencies ───────────────────────────────────────────────────────
if not os.path.exists(FLAG_PATH):
    log('First run — installing dependencies...')

    def pip(pkg):
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages', pkg],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f'pip failed for {pkg}: {result.stderr}')
        else:
            log(f'Installed {pkg}')

    pip('numpy')
    pip('open3d')
    pip('geomdl==5.2.9')
    pip('h5py')
    pip('lap')
    pip('scikit-image')
    pip('scikit-learn')
    pip('scipy')
    pip('six')
    pip('tensorboard-logger')
    pip('trimesh')
    pip('pyvista')
    pip('transforms3d')

    # Install PyTorch with CUDA 12.4 support (compatible with CUDA 13.x)
    log('Installing PyTorch with CUDA support...')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages',
         'torch', 'torchvision',
         '--index-url', 'https://download.pytorch.org/whl/cu124'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f'pip failed for torch+cuda: {result.stderr}')
    else:
        log('Installed torch with CUDA')

    open(FLAG_PATH, 'w').close()
    log('Dependencies installed.')
else:
    log('Dependencies already installed — skipping.')

import numpy as np

# ── Verify GPU ─────────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    log(f'GPU available: {torch.cuda.get_device_name(0)}')
    DEVICE = 'cuda'
else:
    log('WARNING: GPU not available — running on CPU (will be slow)')
    DEVICE = 'cpu'

# ── geomdl numpy compatibility patch ──────────────────────────────────────────
import geomdl
geomdl_path = os.path.dirname(geomdl.__file__)
for root, dirs, files in os.walk(geomdl_path):
    for fname in files:
        if fname.endswith('.py'):
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                src = f.read()
            if 'np.float' in src or 'np.int ' in src or 'np.bool' in src:
                src = (src
                    .replace('np.float', 'float')
                    .replace('np.int ', 'int ')
                    .replace('np.bool', 'bool'))
                with open(fpath, 'w') as f:
                    f.write(src)

# ── lapsolver patch ────────────────────────────────────────────────────────────
from scipy.optimize import linear_sum_assignment

lapsolver = types.ModuleType('lapsolver')
def solve_dense(costs):
    r, c = linear_sum_assignment(costs)
    return r.astype(np.int32), c.astype(np.int32)
lapsolver.solve_dense = solve_dense
sys.modules['lapsolver'] = lapsolver

# ── fitting_utils.py patch ────────────────────────────────────────────────────
parsenet_fitting = f'{BASE_DIR}/parsenet/src/fitting_utils.py'
if os.path.exists(parsenet_fitting):
    with open(parsenet_fitting, 'r') as f:
        src = f.read()
    src = src.replace('from lapsolver import solve_dense', '')
    with open(parsenet_fitting, 'w') as f:
        f.write(src)

# ── Load point cloud written by Unity ─────────────────────────────────────────
import open3d as o3d

if not os.path.exists(PCD_PATH):
    raise FileNotFoundError(f'No PCD at {PCD_PATH} — export from Unity first.')

log(f'Loading {PCD_PATH}...')
pcd = o3d.io.read_point_cloud(PCD_PATH)
log(f'Loaded {len(pcd.points)} points')

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(10)

# ── RANSAC primitive segmentation ─────────────────────────────────────────────
def fit_primitives(pcd, max_primitives=10, distance_thresh=0.01, min_points=100):
    remainder = pcd
    primitives = []
    for i in range(max_primitives):
        if len(remainder.points) < min_points:
            log(f'  Stopping: only {len(remainder.points)} points left')
            break
        plane_model, inliers = remainder.segment_plane(
            distance_threshold=distance_thresh,
            ransac_n=3,
            num_iterations=2000
        )
        inlier_cloud = remainder.select_by_index(inliers)
        remainder    = remainder.select_by_index(inliers, invert=True)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        log(f'  Primitive {i+1}: PLANE | normal={np.round(normal,3)} | {len(inliers)} pts')
        primitives.append(('plane', plane_model, inlier_cloud))
    return primitives, remainder

log('Fitting primitives...')
primitives, leftover = fit_primitives(pcd, max_primitives=8, distance_thresh=0.012)
log(f'\nFound {len(primitives)} primitive(s), {len(leftover.points)} unclassified points')

# ── Export segments ────────────────────────────────────────────────────────────
if os.path.exists(READY_PATH):
    os.remove(READY_PATH)

results = []
for i, (prim_type, model, cloud) in enumerate(primitives):
    seg_path = f'{SEG_DIR}/segment_{i:02d}_{prim_type}.pcd'
    o3d.io.write_point_cloud(seg_path, cloud)
    results.append({
        'id': i,
        'type': prim_type,
        'model': list(model),
        'num_points': len(cloud.points)
    })

with open(f'{SEG_DIR}/primitives.json', 'w') as f:
    json.dump(results, f, indent=2)

if len(leftover.points) > 0:
    o3d.io.write_point_cloud(f'{SEG_DIR}/unclassified.pcd', leftover)

log(f'Saved {len(primitives)} segments to {SEG_DIR}')

# ── Convert to .xyzc for Point2CAD ────────────────────────────────────────────
with open(XYZC_PATH, 'w') as f:
    for segment_id, (prim_type, model, cloud) in enumerate(primitives):
        pts = np.asarray(cloud.points)
        for pt in pts:
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {segment_id}\n')
    if len(leftover.points) > 0:
        leftover_id = len(primitives)
        pts = np.asarray(leftover.points)
        for pt in pts:
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {leftover_id}\n')

log(f'Wrote {XYZC_PATH}')

# ── Clone Point2CAD if not present ────────────────────────────────────────────
if not os.path.exists(P2CAD_DIR):
    log('Cloning Point2CAD...')
    subprocess.run(
        f'git clone https://github.com/prs-eth/point2cad.git {P2CAD_DIR}',
        shell=True, check=True)

# ── Patch io_utils.py — remove pymesh ─────────────────────────────────────────
patched_io_utils = '''import itertools
import json
import numpy as np
import pyvista as pv
import scipy
import trimesh
from collections import Counter
from point2cad.utils import suppress_output_fd


def save_unclipped_meshes(meshes, color_list, out_path):
    non_clipped_meshes = []
    tm_meshes = []
    for s in range(len(meshes)):
        tri = trimesh.Trimesh(
            vertices=np.array(meshes[s]["mesh"].points),
            faces=np.array(meshes[s]["mesh"].faces.reshape(-1, 4)[:, 1:]),
        )
        tri.visual.face_colors = color_list[s]
        non_clipped_meshes.append(tri)
        tm_meshes.append(tri)
    final_non_clipped = trimesh.util.concatenate(non_clipped_meshes)
    final_non_clipped.export(out_path)
    return tm_meshes


def save_clipped_meshes(tm_meshes, out_meshes, color_list, out_path):
    all_vertices = []
    all_faces = []
    face_sources_merged = []
    vert_offset = 0
    for src_id, tm in enumerate(tm_meshes):
        all_vertices.append(tm.vertices)
        all_faces.append(tm.faces + vert_offset)
        face_sources_merged.extend([src_id] * len(tm.faces))
        vert_offset += len(tm.vertices)
    merged_verts = np.concatenate(all_vertices, axis=0)
    merged_faces = np.concatenate(all_faces, axis=0)
    face_sources_merged = np.array(face_sources_merged, dtype=np.int32)
    merged = trimesh.Trimesh(vertices=merged_verts, faces=merged_faces, process=False)
    merged = merged.process(validate=True)
    unique_verts, inv = np.unique(merged.vertices.round(6), axis=0, return_inverse=True)
    new_faces = inv[merged.faces]
    resolved = trimesh.Trimesh(vertices=unique_verts, faces=new_faces, process=False)
    face_sources_resolved = face_sources_merged[:len(resolved.faces)] if len(face_sources_merged) >= len(resolved.faces) else face_sources_merged
    face_adjacency = resolved.face_adjacency
    connected_node_labels = trimesh.graph.connected_component_labels(
        edges=face_adjacency, node_count=len(resolved.faces)
    )
    most_common_groupids = [item[0] for item in Counter(connected_node_labels).most_common()]
    submeshes = [
        trimesh.Trimesh(
            vertices=np.array(resolved.vertices),
            faces=np.array(resolved.faces)[np.where(connected_node_labels == item)],
        )
        for item in most_common_groupids
    ]
    indices_sources = [
        face_sources_resolved[connected_node_labels == item][0]
        for item in np.array(most_common_groupids)
    ]
    clipped_meshes = []
    for p in range(len(out_meshes)):
        one_cluter_points = out_meshes[p]["inpoints"]
        submeshes_cur = [
            x for x, y in zip(submeshes, np.array(indices_sources) == p)
            if y and len(x.faces) > 2
        ]
        if not submeshes_cur:
            clipped_meshes.append(trimesh.Trimesh())
            continue
        nearest_submesh = np.argmin(
            np.array([
                trimesh.proximity.closest_point(item, one_cluter_points)[1]
                for item in submeshes_cur
            ]).transpose(), -1,
        )
        counter_nearest = Counter(nearest_submesh).most_common()
        area_per_point = np.array(
            [submeshes_cur[item[0]].area / item[1] for item in counter_nearest]
        )
        multiplier_area = 2
        nonzero = np.nonzero(area_per_point)[0]
        if len(nonzero) == 0:
            clipped_meshes.append(trimesh.Trimesh())
            continue
        result_indices = np.array(counter_nearest)[:, 0][
            np.logical_and(
                area_per_point < area_per_point[nonzero[0]] * multiplier_area,
                area_per_point != 0,
            )
        ]
        result_submesh_list = [submeshes_cur[item] for item in result_indices]
        clipped_mesh = trimesh.util.concatenate(result_submesh_list)
        clipped_mesh.visual.face_colors = color_list[p]
        clipped_meshes.append(clipped_mesh)
    clipped = trimesh.util.concatenate(clipped_meshes)
    clipped.export(out_path)
    return clipped_meshes


def save_topology(clipped_meshes, out_path):
    filtered_submeshes_pv = [pv.wrap(item) for item in clipped_meshes]
    filtered_submeshes_pv_combinations = list(itertools.combinations(filtered_submeshes_pv, 2))
    intersection_curves = []
    intersections = {}
    for k, pv_pair in enumerate(filtered_submeshes_pv_combinations):
        with suppress_output_fd():
            intersection, _, _ = pv_pair[0].intersection(
                pv_pair[1], split_first=False, split_second=False, progress_bar=False
            )
        if intersection.n_points > 0:
            intersection_curve = {}
            intersection_curve["pv_points"] = intersection.points.tolist()
            intersection_curve["pv_lines"] = intersection.lines.reshape(-1, 3)[:, 1:].tolist()
            intersection_curves.append(intersection_curve)
    intersections["curves"] = intersection_curves
    intersection_corners = []
    for combination_indices in itertools.combinations(range(len(intersection_curves)), 2):
        sample0 = np.array(intersection_curves[combination_indices[0]]["pv_points"])
        sample1 = np.array(intersection_curves[combination_indices[1]]["pv_points"])
        dists = scipy.spatial.distance.cdist(sample0, sample1)
        row_indices, col_indices = np.where(dists == 0)
        if len(row_indices) > 0 and len(col_indices) > 0:
            corners = [
                (sample0[item[0]] + sample1[item[1]]) / 2
                for item in zip(row_indices, col_indices)
            ]
            intersection_corners.extend(corners)
    intersections["corners"] = [arr.tolist() for arr in intersection_corners]
    with open(out_path, "w") as cf:
        json.dump(intersections, cf)
'''

with open(f'{P2CAD_DIR}/point2cad/io_utils.py', 'w') as f:
    f.write(patched_io_utils)
log('Patched io_utils.py — pymesh removed')

# ── Patch get_device() bug ─────────────────────────────────────────────────────
p2c_pkg = f'{P2CAD_DIR}/point2cad'
for fname in os.listdir(p2c_pkg):
    if not fname.endswith('.py'):
        continue
    fpath = os.path.join(p2c_pkg, fname)
    with open(fpath, 'r') as f:
        src = f.read()
    if 'get_device()' in src:
        src = src.replace('.get_device()', '.device')
        with open(fpath, 'w') as f:
            f.write(src)
        log(f'Patched {fname}')

# ── Install Point2CAD dependencies ────────────────────────────────────────────
if not os.path.exists(P2CAD_FLAG):
    log('Installing Point2CAD dependencies...')
    subprocess.run(
        f'{sys.executable} -m pip install -q --break-system-packages -r {P2CAD_DIR}/build/requirements.txt',
        shell=True, check=True)
    open(P2CAD_FLAG, 'w').close()
    log('Point2CAD dependencies installed.')
else:
    log('Point2CAD dependencies already installed — skipping.')

# ── Run Point2CAD with GPU ─────────────────────────────────────────────────────
os.chdir(P2CAD_DIR)
log(f'Running Point2CAD on {DEVICE}...')
result = subprocess.run(
    f'{sys.executable} -m point2cad.main '
    f'--path_in {XYZC_PATH} '
    f'--path_out {OUTPUT_DIR} '
    f'--device {DEVICE}',
    shell=True, capture_output=True, text=True
)

print(result.stdout)
if result.returncode != 0:
    log('STDERR: ' + result.stderr)
    raise RuntimeError('Point2CAD failed')

log('\nOutput files:')
for root, dirs, fnames in os.walk(OUTPUT_DIR):
    for fn in fnames:
        path = os.path.join(root, fn)
        log(f'  {path}  ({os.path.getsize(path):,} bytes)')

# ── Export OBJ for Unity ───────────────────────────────────────────────────────
import trimesh

mesh = trimesh.load(f'{OUTPUT_DIR}/clipped/mesh.ply')
obj_path = f'{BASE_DIR}/cad_output.obj'
mesh.export(obj_path)
log(f'Exported cad_output.obj to {obj_path}')

# ── Signal Unity ───────────────────────────────────────────────────────────────
with open(READY_PATH, 'w') as f:
    f.write('READY')
log('Wrote output_ready.txt — Unity will now load the CAD model')