#!/usr/bin/env python3
"""
GoPro 영상에서 MapAnything을 이용해 depth map과 포인트 클라우드를 추출하는 스크립트
(Modified: Default overlap = 0 for GPS-based alignment)
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import json

try:
    from mapanything.models.mapanything import MapAnything
    from mapanything.utils.image import load_images
    from mapanything.utils.inference import (
        preprocess_input_views_for_inference,
        postprocess_model_outputs_for_inference,
    )
except ImportError as e:
    print(f"Error importing mapanything: {e}")
    sys.exit(1)

def extract_frames(video_path, output_dir, frame_skip=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  총 프레임: {total_frames}")
    
    frame_paths = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_skip == 0:
            p = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(p), frame)
            frame_paths.append(p)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    return frame_paths

def process_frames_multiview(model, frame_paths, depth_dir, ply_dir, device="cuda", max_views=50, overlap=0):
    depth_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(frame_paths)
    if max_views is None: max_views = total
    
    # [수정] 오버랩이 0이면 step은 max_views와 같음
    step = max_views - overlap if overlap > 0 else max_views
    
    chunks_metadata = {
        "total_frames": total,
        "max_views": max_views,
        "overlap": overlap,
        "chunks": []
    }
    
    chunk_idx = 0
    # step 단위로 이동하며 처리
    for chunk_start in range(0, total, step):
        chunk_end = min(chunk_start + max_views, total)
        chunk_paths = frame_paths[chunk_start:chunk_end]
        
        chunk_dir = ply_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        chunks_metadata["chunks"].append({
            "chunk_idx": chunk_idx,
            "frame_start": chunk_start,
            "frame_end": chunk_end - 1,
            "frame_count": len(chunk_paths),
            "frames": [p.stem for p in chunk_paths]
        })
        
        print(f"  [Chunk {chunk_idx}] Processing frames {chunk_start} ~ {chunk_end-1} ({len(chunk_paths)} frames)...")
        
        views = load_images([str(p) for p in chunk_paths], resize_mode="fixed_mapping", norm_type="dinov2", verbose=False)
        views = preprocess_input_views_for_inference(views)
        for view in views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor): view[k] = v.to(device)
                    
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                outputs = model(views)
        outputs = postprocess_model_outputs_for_inference(outputs, views)
        
        for j, (fp, out) in enumerate(zip(chunk_paths, outputs)):
            fn = fp.stem
            if "depth_z" in out:
                np.save(depth_dir / f"{fn}_depth.npy", out["depth_z"].cpu().numpy().squeeze())
            if "pts3d" in out:
                pts = out["pts3d"].cpu().numpy().squeeze().reshape(-1, 3)
                cols = (out["img_no_norm"].cpu().numpy().squeeze() * 255).astype(np.uint8).reshape(-1, 3) if "img_no_norm" in out else None
                
                # 유효 포인트 필터링 및 저장
                mask = np.any(pts != 0, axis=-1)
                pts, cols = pts[mask], cols[mask]
                
                with open(chunk_dir / f"{fn}.ply", 'w') as f:
                    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                    for k in range(len(pts)):
                        f.write(f"{pts[k][0]:.6f} {pts[k][1]:.6f} {pts[k][2]:.6f} {cols[k][0]} {cols[k][1]} {cols[k][2]}\n")
                        
            if "intrinsics" in out:
                np.save(depth_dir / f"{fn}_intrinsics.npy", out["intrinsics"].cpu().numpy().squeeze())
            if "camera_poses" in out:
                np.save(depth_dir / f"{fn}_pose.npy", out["camera_poses"].cpu().numpy().squeeze())
                
        torch.cuda.empty_cache()
        chunk_idx += 1
        
    with open(ply_dir / "chunks.json", 'w') as f:
        json.dump(chunks_metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./video')
    parser.add_argument('-o', '--output', default='./map_output')
    parser.add_argument('--single', default=None)
    parser.add_argument('--frame-skip', type=int, default=1)
    parser.add_argument('--max-views', type=int, default=50)
    # [수정] 기본값 0으로 변경
    parser.add_argument('--overlap', type=int, default=0, help='Overlap (Default: 0 for GPS alignment)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'
    
    print("Loading Model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(args.device)
    model.eval()
    
    out_dir = Path(args.output)
    if args.single:
        process_video(Path(args.single), out_dir, model, args.device, args.frame_skip, args.max_views, args.overlap)
    else:
        for vf in sorted(Path(args.input).glob('*.MP4')):
            process_video(vf, out_dir, model, args.device, args.frame_skip, args.max_views, args.overlap)

def process_video(vp, out_dir, model, dev, skip, max_v, ov):
    print(f"\nProcessing {vp.name}...")
    v_out = out_dir / vp.stem
    frames = extract_frames(vp, v_out / "frames", skip)
    if frames:
        process_frames_multiview(model, frames, v_out / "depth", v_out / "pointcloud", dev, max_v, ov)

if __name__ == '__main__':
    main()