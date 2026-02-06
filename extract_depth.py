#!/usr/bin/env python3
"""
GoPro 영상에서 MapAnything을 이용해 depth map과 포인트 클라우드를 추출하는 스크립트
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from mapanything.models.mapanything import MapAnything
    from mapanything.utils.image import load_images
    from mapanything.utils.inference import (
        preprocess_input_views_for_inference,
        postprocess_model_outputs_for_inference,
    )
except ImportError as e:
    print(f"Error importing mapanything: {e}")
    print("\n필요한 라이브러리를 설치하세요:")
    print('pip install "git+https://github.com/facebookresearch/map-anything.git"')
    sys.exit(1)


def extract_frames(video_path, output_dir, frame_skip=1):
    """비디오에서 프레임 추출

    Args:
        video_path: 비디오 파일 경로
        output_dir: 프레임 저장 디렉토리
        frame_skip: 프레임 건너뛰기 간격 (1이면 모든 프레임)

    Returns:
        추출된 프레임 파일 경로 리스트
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: 비디오를 열 수 없습니다: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  비디오 정보: {total_frames} frames, {fps:.2f} FPS")

    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"  프레임 추출 완료: {saved_idx}개")
    return frame_paths


def save_depth_npy(depth, output_path):
    """Depth map을 NPY 형식으로 저장"""
    np.save(output_path, depth)


def save_pointcloud_ply(points, colors, output_path):
    """포인트 클라우드를 PLY 형식으로 저장

    Args:
        points: (N, 3) 형태의 3D 좌표
        colors: (N, 3) 형태의 RGB 색상 (0-255)
        output_path: 저장 경로
    """
    # 유효한 포인트만 필터링 (depth가 0이 아닌 것)
    mask = np.any(points != 0, axis=-1)
    valid_points = points[mask]
    valid_colors = colors[mask] if colors is not None else None

    if len(valid_points) == 0:
        print(f"  Warning: 유효한 포인트가 없습니다")
        return

    with open(output_path, 'w') as f:
        # PLY 헤더
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(valid_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if valid_colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # 포인트 데이터
        for i in range(len(valid_points)):
            x, y, z = valid_points[i]
            if valid_colors is not None:
                r, g, b = valid_colors[i].astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def process_frames_multiview(model, frame_paths, depth_dir, ply_dir, device="cuda", max_views=None, overlap=0):
    """여러 프레임을 한번에 처리하여 정렬된 depth map과 포인트 클라우드 생성 (Multi-view reconstruction)

    Args:
        model: MapAnything 모델
        frame_paths: 프레임 파일 경로 리스트
        depth_dir: depth map 저장 디렉토리
        ply_dir: 포인트 클라우드 저장 디렉토리
        device: 디바이스 (cuda/cpu)
        max_views: 한번에 처리할 최대 뷰 수 (None이면 전체)
        overlap: 청크 간 오버랩 프레임 수
    """
    import json

    depth_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    total = len(frame_paths)

    # max_views가 지정되지 않았거나 total보다 크면 전체 처리
    if max_views is None or max_views >= total:
        max_views = total

    # 오버랩 검증
    if overlap >= max_views:
        print(f"  Warning: overlap({overlap})이 max_views({max_views})보다 크거나 같아 0으로 설정")
        overlap = 0

    # 청크 간 이동 간격 (오버랩 적용)
    step = max_views - overlap if overlap > 0 else max_views

    # 청크 메타데이터 저장용
    chunks_metadata = {
        "total_frames": total,
        "max_views": max_views,
        "overlap": overlap,
        "step": step,
        "chunks": []
    }

    # 청크 단위로 처리
    chunk_idx = 0
    for chunk_start in range(0, total, step):
        chunk_end = min(chunk_start + max_views, total)
        chunk_paths = frame_paths[chunk_start:chunk_end]

        # 청크별 디렉토리 생성
        chunk_dir = ply_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # 청크 메타데이터 기록
        chunk_info = {
            "chunk_idx": chunk_idx,
            "frame_start": chunk_start,
            "frame_end": chunk_end - 1,
            "frame_count": len(chunk_paths),
            "frames": [p.stem for p in chunk_paths]
        }
        chunks_metadata["chunks"].append(chunk_info)

        print(f"  Multi-view 처리 중: 청크 {chunk_idx} (프레임 {chunk_start}~{chunk_end-1}, {len(chunk_paths)}개)")

        # 이미지 로드 (모든 프레임을 한번에)
        views = load_images(
            [str(p) for p in chunk_paths],
            resize_mode="fixed_mapping",
            norm_type="dinov2",
            verbose=False
        )

        # 추론 전처리
        views = preprocess_input_views_for_inference(views)

        # GPU로 이동
        for view in views:
            for key, value in view.items():
                if isinstance(value, torch.Tensor):
                    view[key] = value.to(device)

        # 모델 추론 (모든 뷰를 한번에 - multi-view reconstruction)
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                outputs = model(views)

        # 후처리
        outputs = postprocess_model_outputs_for_inference(outputs, views)

        # 각 프레임별 저장 + 통합 포인트 클라우드 수집
        for j, (frame_path, output) in enumerate(zip(chunk_paths, outputs)):
            frame_name = frame_path.stem
            frame_idx = chunk_start + j

            # Depth map 저장 (NPY)
            if "depth_z" in output:
                depth = output["depth_z"].cpu().numpy().squeeze()
                depth_path = depth_dir / f"{frame_name}_depth.npy"
                save_depth_npy(depth, depth_path)

            # 포인트 클라우드 저장 (PLY) - 월드 좌표계 (pts3d)
            if "pts3d" in output:
                pts3d = output["pts3d"].cpu().numpy().squeeze()  # (H, W, 3)

                # 색상 정보
                if "img_no_norm" in output:
                    colors = (output["img_no_norm"].cpu().numpy().squeeze() * 255).astype(np.uint8)
                else:
                    colors = None

                # (H, W, 3) -> (N, 3)
                pts3d_flat = pts3d.reshape(-1, 3)
                colors_flat = colors.reshape(-1, 3) if colors is not None else None

                # 개별 프레임 PLY 저장 (청크별 디렉토리에)
                ply_path = chunk_dir / f"{frame_name}.ply"
                save_pointcloud_ply(pts3d_flat, colors_flat, ply_path)

            # Intrinsics 저장
            if "intrinsics" in output:
                intrinsics = output["intrinsics"].cpu().numpy().squeeze()
                intrinsics_path = depth_dir / f"{frame_name}_intrinsics.npy"
                np.save(intrinsics_path, intrinsics)

            # 카메라 pose 저장 (있으면)
            if "camera_poses" in output:
                camera_pose = output["camera_poses"].cpu().numpy().squeeze()
                pose_path = depth_dir / f"{frame_name}_pose.npy"
                np.save(pose_path, camera_pose)

        print(f"  처리 완료: 청크 {chunk_idx} ({chunk_end}/{total} 프레임)")

        # GPU 메모리 정리
        torch.cuda.empty_cache()

        chunk_idx += 1

    # 청크 메타데이터 저장
    chunks_json_path = ply_dir / "chunks.json"
    with open(chunks_json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_metadata, f, indent=2, ensure_ascii=False)
    print(f"  청크 메타데이터 저장: {chunks_json_path}")


def process_frames_batch(model, frame_paths, depth_dir, ply_dir, batch_size=1, device="cuda"):
    """프레임을 개별적으로 처리 (single-view, 레거시 호환용)

    Args:
        model: MapAnything 모델
        frame_paths: 프레임 파일 경로 리스트
        depth_dir: depth map 저장 디렉토리
        ply_dir: 포인트 클라우드 저장 디렉토리
        batch_size: 배치 크기
        device: 디바이스 (cuda/cpu)
    """
    depth_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    total = len(frame_paths)

    for i in range(0, total, batch_size):
        batch_paths = frame_paths[i:i+batch_size]

        # 이미지 로드
        views = load_images(
            [str(p) for p in batch_paths],
            resize_mode="fixed_mapping",
            norm_type="dinov2",
            verbose=False
        )

        # 추론 전처리
        views = preprocess_input_views_for_inference(views)

        # GPU로 이동
        for view in views:
            for key, value in view.items():
                if isinstance(value, torch.Tensor):
                    view[key] = value.to(device)

        # 모델 추론
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                outputs = model(views)

        # 후처리
        outputs = postprocess_model_outputs_for_inference(outputs, views)

        # 각 프레임별 저장
        for j, (frame_path, output) in enumerate(zip(batch_paths, outputs)):
            frame_name = frame_path.stem

            # Depth map 저장 (NPY)
            if "depth_z" in output:
                depth = output["depth_z"].cpu().numpy().squeeze()
                depth_path = depth_dir / f"{frame_name}_depth.npy"
                save_depth_npy(depth, depth_path)

            # 포인트 클라우드 저장 (PLY)
            if "pts3d" in output:
                pts3d = output["pts3d"].cpu().numpy().squeeze()  # (H, W, 3)

                # 색상 정보 (원본 이미지에서)
                if "img_no_norm" in output:
                    colors = (output["img_no_norm"].cpu().numpy().squeeze() * 255).astype(np.uint8)
                else:
                    colors = None

                # (H, W, 3) -> (N, 3)
                pts3d_flat = pts3d.reshape(-1, 3)
                colors_flat = colors.reshape(-1, 3) if colors is not None else None

                ply_path = ply_dir / f"{frame_name}.ply"
                save_pointcloud_ply(pts3d_flat, colors_flat, ply_path)

            # Intrinsics 저장 (카메라 내부 파라미터 - 나중에 거리 계산에 필요)
            if "intrinsics" in output:
                intrinsics = output["intrinsics"].cpu().numpy().squeeze()
                intrinsics_path = depth_dir / f"{frame_name}_intrinsics.npy"
                np.save(intrinsics_path, intrinsics)

        print(f"  처리 완료: {min(i+batch_size, total)}/{total} 프레임")


def process_video(video_path, output_dir, model, device="cuda", frame_skip=1, max_views=None, overlap=0):
    """단일 비디오 처리

    Args:
        video_path: 비디오 파일 경로
        output_dir: 출력 디렉토리
        model: MapAnything 모델
        device: 디바이스
        frame_skip: 프레임 건너뛰기 간격
        max_views: 한번에 처리할 최대 뷰 수 (None이면 전체를 한번에)
        overlap: 청크 간 오버랩 프레임 수

    Returns:
        성공 여부
    """
    video_name = video_path.stem
    video_output_dir = output_dir / video_name

    # 디렉토리 구조 생성
    frames_dir = video_output_dir / "frames"
    depth_dir = video_output_dir / "depth"
    ply_dir = video_output_dir / "pointcloud"

    print(f"\n[1/2] 프레임 추출 중...")
    frame_paths = extract_frames(video_path, frames_dir, frame_skip)

    if not frame_paths:
        print(f"  Error: 프레임 추출 실패")
        return False

    print(f"\n[2/2] Multi-view reconstruction 중... (총 {len(frame_paths)}개 프레임)")
    process_frames_multiview(model, frame_paths, depth_dir, ply_dir, device=device, max_views=max_views, overlap=overlap)

    print(f"\n출력 디렉토리: {video_output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='GoPro 영상에서 MapAnything을 이용해 depth map과 포인트 클라우드 추출'
    )
    parser.add_argument(
        '-i', '--input',
        help='입력 폴더 경로 (기본값: ./video)',
        default='./video'
    )
    parser.add_argument(
        '-o', '--output',
        help='출력 폴더 경로 (기본값: ./map_output)',
        default='./map_output'
    )
    parser.add_argument(
        '--single',
        help='단일 영상 파일 처리',
        default=None
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='프레임 건너뛰기 간격 (기본값: 1, 모든 프레임)'
    )
    parser.add_argument(
        '--max-views',
        type=int,
        default=50,
        help='한번에 처리할 최대 뷰 수 (기본값: 50). GPU 메모리에 따라 조절'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=10,
        help='청크 간 오버랩 프레임 수 (기본값: 10). 정합 품질 향상용'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='디바이스 (cuda/cpu, 기본값: cuda)'
    )

    args = parser.parse_args()

    # GPU 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'

    print("=" * 50)
    print("MapAnything Depth Extractor")
    print("=" * 50)

    # 모델 로드 (Hugging Face에서 직접 로드)
    print("\n모델 로딩 중... (처음 실행 시 다운로드에 시간이 걸릴 수 있습니다)")
    model = MapAnything.from_pretrained("facebook/map-anything").to(args.device)
    model.eval()
    print("모델 로딩 완료")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 단일 파일 모드
    if args.single:
        video_path = Path(args.single)
        if not video_path.exists():
            print(f"Error: 비디오 파일을 찾을 수 없습니다: {video_path}")
            sys.exit(1)

        print(f"\n처리 중: {video_path.name}")
        if process_video(video_path, output_dir, model, args.device, args.frame_skip, args.max_views, args.overlap):
            print("\n처리 완료!")
        else:
            sys.exit(1)
        return

    # 배치 모드
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        sys.exit(1)

    video_extensions = ['.mp4', '.MP4', '.mov', '.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))

    if not video_files:
        print(f"비디오 파일을 찾을 수 없습니다: {input_dir}")
        sys.exit(1)

    print(f"\n발견된 비디오: {len(video_files)}개")

    success_count = 0
    for idx, video_file in enumerate(sorted(video_files), 1):
        print(f"\n{'=' * 50}")
        print(f"[{idx}/{len(video_files)}] {video_file.name}")
        print("=" * 50)

        if process_video(video_file, output_dir, model, args.device, args.frame_skip, args.max_views, args.overlap):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"처리 완료: {success_count}/{len(video_files)} 비디오")
    print(f"출력 디렉토리: {output_dir}")


if __name__ == '__main__':
    main()
