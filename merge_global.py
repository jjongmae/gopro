#!/usr/bin/env python3
"""
전역 UTM 좌표 포인트 클라우드 병합 스크립트
- 여러 aligned_global.ply 파일을 하나로 합침
- 스트리밍 방식으로 대용량 파일 처리 가능
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def count_ply_vertices(ply_path):
    """PLY 파일의 정점 수와 색상 여부 확인"""
    vertex_count = 0
    has_color = False

    with open(ply_path, 'r') as f:
        for line in f:
            if "element vertex" in line:
                vertex_count = int(line.split()[-1])
            if "property uchar red" in line:
                has_color = True
            if "end_header" in line:
                break

    return vertex_count, has_color


def iter_ply_points(ply_path):
    """PLY 파일에서 포인트를 한 줄씩 yield (메모리 효율적)"""
    with open(ply_path, 'r') as f:
        in_data = False
        for line in f:
            if "end_header" in line:
                in_data = True
                continue
            if in_data:
                parts = line.strip().split()
                if len(parts) >= 3:
                    yield parts  # [x, y, z, r, g, b, ...]


def merge_global_plys_streaming(base_dir='.', output_name='merged_global.ply',
                                 max_height_above_ground=None, cell_size=5.0):
    """스트리밍 방식으로 병합 (대용량 처리 가능)"""
    base_path = Path(base_dir)
    map_dir = base_path / 'map_output'

    if not map_dir.exists():
        print(f"디렉토리 없음: {map_dir}")
        return

    # aligned_global.ply 파일 찾기
    global_files = sorted(map_dir.glob('*/aligned_global.ply'))

    if not global_files:
        print("aligned_global.ply 파일이 없습니다.")
        print("먼저 convert_to_global.py를 실행하세요.")
        return

    print(f"=== 포인트 클라우드 병합 (스트리밍) ===")
    print(f"발견된 파일: {len(global_files)}개\n")

    # 전체 포인트 수 및 색상 여부 확인
    total_points = 0
    has_color = False
    file_info = []

    for ply_file in global_files:
        count, has_c = count_ply_vertices(str(ply_file))
        file_info.append((ply_file, count, has_c))
        total_points += count
        if has_c:
            has_color = True
        video_id = ply_file.parent.name
        print(f"  {video_id}: {count:,} points")

    print(f"\n총 {total_points:,} points")

    # 높이 필터링이 필요한 경우: 2-pass 처리
    if max_height_above_ground is not None:
        print(f"\n=== Pass 1: 지면 높이 계산 ===")
        print(f"셀 크기: {cell_size}m")

        # Pass 1: 각 셀의 최저 Z값 수집
        cell_min_z = defaultdict(lambda: float('inf'))
        min_x, min_y = float('inf'), float('inf')

        # 먼저 전체 범위 확인
        for ply_file, count, _ in tqdm(file_info, desc="범위 확인"):
            for parts in iter_ply_points(str(ply_file)):
                x, y = float(parts[0]), float(parts[1])
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y

        print(f"최소 좌표: X={min_x:.2f}, Y={min_y:.2f}")

        # 셀별 최저 Z값 계산
        for ply_file, count, _ in tqdm(file_info, desc="지면 계산"):
            for parts in iter_ply_points(str(ply_file)):
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                cell_x = int((x - min_x) / cell_size)
                cell_y = int((y - min_y) / cell_size)
                cell_key = (cell_x, cell_y)
                if z < cell_min_z[cell_key]:
                    cell_min_z[cell_key] = z

        print(f"셀 개수: {len(cell_min_z):,}")

        # Pass 2: 필터링하면서 저장
        print(f"\n=== Pass 2: 필터링 및 저장 ===")
        print(f"지면 기준 최대 높이: {max_height_above_ground}m")

        output_path = base_path / output_name
        temp_path = base_path / (output_name + '.tmp')

        kept_count = 0
        removed_count = 0

        with open(temp_path, 'w') as f_out:
            # 헤더는 나중에 작성 (포인트 수를 알아야 함)
            f_out.write("PLACEHOLDER_HEADER\n")

            for ply_file, count, _ in tqdm(file_info, desc="필터링"):
                for parts in iter_ply_points(str(ply_file)):
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    cell_x = int((x - min_x) / cell_size)
                    cell_y = int((y - min_y) / cell_size)
                    cell_key = (cell_x, cell_y)

                    ground_z = cell_min_z[cell_key]
                    height_above = z - ground_z

                    if height_above <= max_height_above_ground:
                        # 포인트 유지
                        if has_color and len(parts) >= 6:
                            f_out.write(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}\n")
                        else:
                            f_out.write(f"{parts[0]} {parts[1]} {parts[2]}\n")
                        kept_count += 1
                    else:
                        removed_count += 1

        # 최종 파일 작성 (올바른 헤더 포함)
        print(f"\n최종 파일 작성 중...")
        with open(output_path, 'w') as f_out:
            f_out.write(f"ply\nformat ascii 1.0\nelement vertex {kept_count}\n")
            f_out.write("property float x\nproperty float y\nproperty float z\n")
            if has_color:
                f_out.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f_out.write("end_header\n")

            # temp 파일에서 데이터 복사
            with open(temp_path, 'r') as f_in:
                f_in.readline()  # PLACEHOLDER_HEADER 스킵
                for line in tqdm(f_in, total=kept_count, desc="저장"):
                    f_out.write(line)

        # temp 파일 삭제
        os.remove(temp_path)

        print(f"\n제거됨: {removed_count:,} points ({removed_count/(kept_count+removed_count)*100:.1f}%)")
        print(f"남은 포인트: {kept_count:,}")

    else:
        # 필터링 없이 단순 병합
        output_path = base_path / output_name

        print(f"\n저장 중: {output_path}")

        with open(output_path, 'w') as f_out:
            f_out.write(f"ply\nformat ascii 1.0\nelement vertex {total_points}\n")
            f_out.write("property float x\nproperty float y\nproperty float z\n")
            if has_color:
                f_out.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f_out.write("end_header\n")

            for ply_file, count, _ in tqdm(file_info, desc="병합"):
                for parts in iter_ply_points(str(ply_file)):
                    if has_color and len(parts) >= 6:
                        f_out.write(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}\n")
                    else:
                        f_out.write(f"{parts[0]} {parts[1]} {parts[2]}\n")

    print(f"\n완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='전역 UTM 포인트 클라우드 병합')
    parser.add_argument('--base-dir', type=str, default='.', help='기본 디렉토리')
    parser.add_argument('--output', type=str, default='merged_global.ply', help='출력 파일명')
    parser.add_argument('--max-height', type=float, default=None,
                        help='지면 기준 최대 높이(m). 예: 3 = 지면에서 3m 이상은 제거 (나무 등)')
    parser.add_argument('--cell-size', type=float, default=5.0,
                        help='지면 계산용 그리드 셀 크기(m). 기본값: 5m')
    args = parser.parse_args()

    merge_global_plys_streaming(args.base_dir, args.output, args.max_height, args.cell_size)


if __name__ == '__main__':
    main()
