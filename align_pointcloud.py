#!/usr/bin/env python3
"""
청크별로 분리된 포인트 클라우드를 Umeyama 알고리즘으로 정합하는 스크립트

오버랩 프레임을 이용하여 인접 청크 간 변환 행렬을 계산하고,
모든 청크를 첫 번째 청크의 좌표계로 통합합니다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_ply(ply_path):
    """PLY 파일에서 포인트와 색상 로드

    Returns:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array or None
    """
    points = []
    colors = []
    has_color = False

    with open(ply_path, 'r') as f:
        # 헤더 파싱
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"올바른 PLY 파일이 아닙니다: {ply_path}")

        vertex_count = 0
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif "property" in line and "red" in line:
                has_color = True
            elif line == "end_header":
                break

        # 데이터 파싱
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])

            if has_color and len(parts) >= 6:
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                colors.append([r, g, b])

    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8) if colors else None

    return points, colors


def save_ply(ply_path, points, colors=None):
    """포인트 클라우드를 PLY 파일로 저장"""
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def umeyama(src, dst):
    """Umeyama 알고리즘으로 최적의 rigid transformation 계산

    src와 dst는 대응점 쌍 (같은 인덱스가 같은 점)
    dst = R @ src + t 를 만족하는 R, t를 찾음

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        scale: scale factor (rigid transformation에서는 1에 가까워야 함)
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    # 중심점 계산
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # 중심점 이동
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # 공분산 행렬
    H = src_centered.T @ dst_centered / n

    # SVD 분해
    U, S, Vt = np.linalg.svd(H)

    # 회전 행렬 계산
    R = Vt.T @ U.T

    # Reflection 보정 (det(R) = -1인 경우)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 스케일 계산 (rigid transformation에서는 사용 안 함)
    src_var = np.sum(src_centered ** 2) / n
    scale = np.sum(S) / src_var if src_var > 0 else 1.0

    # Translation 계산
    t = dst_mean - R @ src_mean

    return R, t, scale


def apply_transform(points, R, t):
    """포인트에 변환 적용"""
    return (R @ points.T).T + t


def compute_rmse(src, dst):
    """두 점 집합 간 RMSE 계산"""
    diff = src - dst
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def find_overlap_frames(chunk_a_info, chunk_b_info):
    """두 청크 간 오버랩 프레임 찾기"""
    frames_a = set(chunk_a_info["frames"])
    frames_b = set(chunk_b_info["frames"])
    overlap = frames_a & frames_b
    return sorted(list(overlap))


def load_chunk_points(chunk_dir, frame_names):
    """청크에서 특정 프레임들의 포인트 로드

    Returns:
        all_points: dict {frame_name: (points, colors)}
    """
    all_points = {}
    for frame_name in frame_names:
        ply_path = chunk_dir / f"{frame_name}.ply"
        if ply_path.exists():
            points, colors = load_ply(ply_path)
            all_points[frame_name] = (points, colors)
    return all_points


def sample_corresponding_points(points_a, points_b, num_samples=10000):
    """두 포인트 클라우드에서 대응점 샘플링

    같은 프레임의 PLY는 같은 픽셀 순서로 저장되어 있으므로,
    같은 인덱스가 대응점임
    """
    n = min(len(points_a), len(points_b))

    if n == 0:
        return np.array([]), np.array([])

    # 유효한 포인트만 필터링 (0이 아닌 점)
    valid_mask = (np.any(points_a[:n] != 0, axis=1) &
                  np.any(points_b[:n] != 0, axis=1))

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.array([]), np.array([])

    # 샘플링
    if len(valid_indices) > num_samples:
        sampled_indices = np.random.choice(valid_indices, num_samples, replace=False)
    else:
        sampled_indices = valid_indices

    return points_a[sampled_indices], points_b[sampled_indices]


def align_chunks(ply_dir, output_path=None, verbose=True):
    """청크들을 정합하여 통합 포인트 클라우드 생성

    Args:
        ply_dir: pointcloud 디렉토리 경로
        output_path: 출력 PLY 파일 경로 (None이면 ply_dir/aligned_combined.ply)
        verbose: 상세 로그 출력 여부

    Returns:
        성공 여부
    """
    ply_dir = Path(ply_dir)
    chunks_json = ply_dir / "chunks.json"

    if not chunks_json.exists():
        print(f"Error: chunks.json을 찾을 수 없습니다: {chunks_json}")
        return False

    # 메타데이터 로드
    with open(chunks_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    chunks = metadata["chunks"]
    num_chunks = len(chunks)

    if num_chunks == 0:
        print("Error: 청크가 없습니다.")
        return False

    if verbose:
        print(f"\n총 {num_chunks}개 청크 발견")
        print(f"오버랩: {metadata['overlap']} 프레임\n")

    if num_chunks == 1:
        print("청크가 1개뿐이므로 정합이 필요 없습니다.")
        # 단일 청크의 모든 포인트 합치기
        chunk_dir = ply_dir / f"chunk_000"
        all_points = []
        all_colors = []

        for frame_name in chunks[0]["frames"]:
            ply_path = chunk_dir / f"{frame_name}.ply"
            if ply_path.exists():
                points, colors = load_ply(ply_path)
                valid_mask = np.any(points != 0, axis=1)
                all_points.append(points[valid_mask])
                if colors is not None:
                    all_colors.append(colors[valid_mask])

        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors) if all_colors else None

        if output_path is None:
            output_path = ply_dir / "aligned_combined.ply"

        save_ply(output_path, combined_points, combined_colors)
        print(f"통합 포인트 클라우드 저장: {output_path} ({len(combined_points):,}개 포인트)")
        return True

    # 첫 번째 청크를 기준으로 설정
    transforms = [None] * num_chunks  # 각 청크의 변환 (첫 번째 청크 좌표계 기준)
    transforms[0] = (np.eye(3), np.zeros(3))  # 첫 번째 청크는 항등 변환

    # 인접 청크 간 변환 계산
    for i in range(num_chunks - 1):
        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]

        chunk_a_dir = ply_dir / f"chunk_{chunk_a['chunk_idx']:03d}"
        chunk_b_dir = ply_dir / f"chunk_{chunk_b['chunk_idx']:03d}"

        # 오버랩 프레임 찾기
        overlap_frames = find_overlap_frames(chunk_a, chunk_b)

        if len(overlap_frames) == 0:
            print(f"Warning: 청크 {i}와 {i+1} 사이에 오버랩 프레임이 없습니다!")
            # 이전 변환 그대로 사용 (드리프트 발생 가능)
            transforms[i + 1] = transforms[i]
            continue

        if verbose:
            print(f"청크 {i} -> {i+1}: 오버랩 프레임 {len(overlap_frames)}개")

        # 오버랩 프레임에서 대응점 수집
        all_src_points = []
        all_dst_points = []

        for frame_name in overlap_frames:
            ply_a = chunk_a_dir / f"{frame_name}.ply"
            ply_b = chunk_b_dir / f"{frame_name}.ply"

            if ply_a.exists() and ply_b.exists():
                points_a, _ = load_ply(ply_a)
                points_b, _ = load_ply(ply_b)

                # 대응점 샘플링
                src_sampled, dst_sampled = sample_corresponding_points(
                    points_b, points_a, num_samples=5000
                )

                if len(src_sampled) > 0:
                    all_src_points.append(src_sampled)
                    all_dst_points.append(dst_sampled)

        if not all_src_points:
            print(f"Warning: 청크 {i}와 {i+1} 사이에 유효한 대응점이 없습니다!")
            transforms[i + 1] = transforms[i]
            continue

        # 모든 대응점 합치기
        src_points = np.vstack(all_src_points)
        dst_points = np.vstack(all_dst_points)

        if verbose:
            print(f"  대응점 수: {len(src_points):,}")

        # Umeyama 알고리즘으로 변환 계산
        # src (청크 B) -> dst (청크 A) 변환
        R_local, t_local, scale = umeyama(src_points, dst_points)

        # RMSE 계산
        transformed = apply_transform(src_points, R_local, t_local)
        rmse = compute_rmse(transformed, dst_points)

        if verbose:
            print(f"  RMSE: {rmse:.6f}, Scale: {scale:.6f}")

        # 누적 변환 계산 (청크 0 좌표계 기준)
        R_prev, t_prev = transforms[i]
        R_cumulative = R_prev @ R_local
        t_cumulative = R_prev @ t_local + t_prev

        transforms[i + 1] = (R_cumulative, t_cumulative)

    # 모든 청크의 포인트 변환 및 합치기
    if verbose:
        print("\n포인트 클라우드 통합 중...")

    all_points = []
    all_colors = []
    processed_frames = set()  # 중복 방지

    for i, chunk in enumerate(chunks):
        chunk_dir = ply_dir / f"chunk_{chunk['chunk_idx']:03d}"
        R, t = transforms[i]

        for frame_name in chunk["frames"]:
            # 이미 처리된 프레임은 건너뛰기 (오버랩 영역 중복 방지)
            if frame_name in processed_frames:
                continue
            processed_frames.add(frame_name)

            ply_path = chunk_dir / f"{frame_name}.ply"
            if not ply_path.exists():
                continue

            points, colors = load_ply(ply_path)

            # 유효한 포인트만 필터링
            valid_mask = np.any(points != 0, axis=1)
            valid_points = points[valid_mask]
            valid_colors = colors[valid_mask] if colors is not None else None

            # 변환 적용
            transformed_points = apply_transform(valid_points, R, t)

            all_points.append(transformed_points)
            if valid_colors is not None:
                all_colors.append(valid_colors)

    # 최종 통합
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors) if all_colors else None

    # 저장
    if output_path is None:
        output_path = ply_dir / "aligned_combined.ply"

    save_ply(output_path, combined_points, combined_colors)

    print(f"\n정합 완료!")
    print(f"  총 포인트: {len(combined_points):,}개")
    print(f"  출력 파일: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='청크별 포인트 클라우드를 Umeyama 알고리즘으로 정합'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='pointcloud 디렉토리 경로 (chunks.json이 있는 디렉토리)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='출력 PLY 파일 경로 (기본값: input/aligned_combined.ply)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='상세 로그 출력 안 함'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 입력 경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    print("=" * 50)
    print("포인트 클라우드 정합 (Umeyama Algorithm)")
    print("=" * 50)

    success = align_chunks(input_path, output_path, verbose=not args.quiet)

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
