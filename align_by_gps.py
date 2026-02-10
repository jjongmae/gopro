#!/usr/bin/env python3
"""
GPS 기반 포인트 클라우드 정합 도구 (No-Overlap Optimized)
- 오버랩 없는 데이터를 전제로 한 고속/정밀 정합
- 스케일 안정화 및 연속성 보장
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_ply_vertices(ply_path):
    if not os.path.exists(ply_path): return np.array([]), None
    with open(ply_path, 'r') as f: lines = f.readlines()
    header_end = 0
    has_color = False
    for i, line in enumerate(lines):
        if "element vertex" in line: count = int(line.split()[-1])
        if "property uchar red" in line: has_color = True
        if "end_header" in line:
            header_end = i + 1
            break
    verts, colors = [], []
    for line in lines[header_end:]:
        v = line.split()
        if len(v) < 3: continue
        verts.append([float(v[0]), float(v[1]), float(v[2])])
        if has_color and len(v) >= 6:
            colors.append([int(float(v[3])), int(float(v[4])), int(float(v[5]))])
    return np.array(verts), np.array(colors) if has_color else None

def save_ply(path, points, colors=None):
    if len(points) == 0: return
    print(f"\n✓ 저장 중: {path} ({len(points):,} points)")
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            p = points[i]
            c = f"{colors[i][0]} {colors[i][1]} {colors[i][2]}" if colors is not None else ""
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c}\n")

def get_gps_rotation_matrix(gps_vec):
    f = gps_vec / (np.linalg.norm(gps_vec) + 1e-6)
    world_up = np.array([0, 0, 1])
    r = np.cross(f, world_up)
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-6: r = np.array([1, 0, 0])
    else: r /= norm_r
    real_up = np.cross(r, f)
    return np.column_stack((r, -real_up, f))

def align_chunks(video_id, base_dir='.'):
    base_path = Path(base_dir)
    gps_json = base_path / 'gps_output' / video_id / 'gps_path.json'
    chunks_meta = base_path / 'map_output' / video_id / 'pointcloud' / 'chunks.json'
    output_ply = base_path / 'map_output' / video_id / 'aligned_final.ply'
    
    if not gps_json.exists() or not chunks_meta.exists():
        print(f"필수 파일 누락: {video_id}")
        return

    print(f"=== 정합 시작: {video_id} (No-Overlap Optimized) ===")

    with open(gps_json) as f: gps_pts = {p['frame_idx']: p for p in json.load(f)['points']}
    with open(chunks_meta) as f: chunks = json.load(f)['chunks']
    
    all_pts, all_cols = [], []
    
    # 상태 변수
    prev_vis_pos = None
    prev_gps_pos = None
    last_valid_scale = 1.0 
    
    total_frames = chunks[-1]['frame_end'] + 1
    pbar = tqdm(total=total_frames, desc=f"Processing")

    for chunk in chunks:
        frames = chunk['frames']
        chunk_start = chunk['frame_start']
        
        chunk_dir = base_path / 'map_output' / video_id / 'pointcloud' / f"chunk_{chunk['chunk_idx']:03d}"
        pose_dir = base_path / 'map_output' / video_id / 'depth'
        
        for fname in frames:
            pbar.update(1)
            frame_idx = int(fname.split('_')[-1])
            
            if frame_idx not in gps_pts: continue
            
            # Load Data
            ply_path = chunk_dir / f"{fname}.ply"
            pose_path = pose_dir / f"{fname}_pose.npy"
            if not ply_path.exists() or not pose_path.exists(): continue
            
            pts, cols = load_ply_vertices(str(ply_path))
            if len(pts) == 0: continue
            pose = np.load(pose_path)
            
            # Current Positions
            vis_pos_curr = pose[:3, 3]
            gps_data = gps_pts[frame_idx]
            gps_pos_curr = np.array([gps_data['x'], gps_data['y'], gps_data['z']])
            gps_vec = np.array(gps_data['vec'])
            
            # --- Scale Calculation ---
            # 청크가 바뀌면(frame_idx == chunk_start) 좌표계가 리셋되므로
            # 이전 프레임과의 거리 비교가 불가능함 -> 스케일 계산 건너뛰고 이전 값 유지
            is_boundary = (frame_idx == chunk_start)
            current_scale = last_valid_scale
            
            if not is_boundary and prev_vis_pos is not None:
                d_vis = np.linalg.norm(vis_pos_curr - prev_vis_pos)
                d_gps = np.linalg.norm(gps_pos_curr - prev_gps_pos)
                
                if d_vis > 0.01:
                    raw_scale = d_gps / d_vis
                    # 급격한 변화 방지 (Smoothing)
                    if 0.2 < raw_scale < 5.0:
                        current_scale = 0.9 * last_valid_scale + 0.1 * raw_scale
                        last_valid_scale = current_scale

            # Update State
            prev_vis_pos = vis_pos_curr
            prev_gps_pos = gps_pos_curr

            # --- Alignment ---
            # 1. Local to Camera
            R_cam = pose[:3, :3]
            T_cam = pose[:3, 3]
            pts_local = (pts - T_cam) @ R_cam 
            
            # 2. Camera to GPS World
            if np.linalg.norm(gps_vec) < 1e-6: gps_vec = np.array([0, 1, 0])
            R_gps = get_gps_rotation_matrix(gps_vec)
            
            # 3. Final Transform
            pts_final = (pts_local * current_scale) @ R_gps.T + gps_pos_curr
            
            all_pts.append(pts_final)
            if cols is not None: all_cols.append(cols)

    pbar.close()

    if all_pts:
        save_ply(str(output_ply), np.vstack(all_pts), np.vstack(all_cols) if all_cols else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--base-dir', type=str, default='.')
    args = parser.parse_args()
    if args.video: align_chunks(args.video, args.base_dir)
    else: 
        d = Path(args.base_dir)/'map_output'
        if d.exists(): [align_chunks(x.name, args.base_dir) for x in d.iterdir() if x.is_dir()]