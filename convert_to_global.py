#!/usr/bin/env python3
"""
포인트 클라우드를 전역 UTM 좌표로 변환하는 후처리 스크립트
- 기존 aligned_final.ply를 읽어서 UTM 좌표로 변환
- CloudCompare에서 여러 비디오를 열면 지도처럼 자동 정렬됨
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path

try:
    from pyproj import Proj, Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("경고: pyproj가 설치되지 않았습니다. 'pip install pyproj'로 설치해주세요.")


def get_utm_zone(lon):
    """경도로부터 UTM 존 번호 계산"""
    return int((lon + 180) / 6) + 1


def latlon_to_utm(lat, lon):
    """위경도를 UTM 좌표(미터)로 변환"""
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj가 필요합니다: pip install pyproj")

    zone = get_utm_zone(lon)
    hemisphere = 'north' if lat >= 0 else 'south'

    # WGS84 -> UTM 변환기 생성
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84 위경도
        f"+proj=utm +zone={zone} +{hemisphere} +ellps=WGS84",
        always_xy=True
    )

    easting, northing = transformer.transform(lon, lat)
    return easting, northing, zone


def load_ply_vertices(ply_path):
    """PLY 파일에서 정점과 색상 로드"""
    if not os.path.exists(ply_path):
        return np.array([]), None

    with open(ply_path, 'r') as f:
        lines = f.readlines()

    header_end = 0
    has_color = False
    vertex_count = 0

    for i, line in enumerate(lines):
        if "element vertex" in line:
            vertex_count = int(line.split()[-1])
        if "property uchar red" in line:
            has_color = True
        if "end_header" in line:
            header_end = i + 1
            break

    verts, colors = [], []
    for line in lines[header_end:]:
        v = line.split()
        if len(v) < 3:
            continue
        verts.append([float(v[0]), float(v[1]), float(v[2])])
        if has_color and len(v) >= 6:
            colors.append([int(float(v[3])), int(float(v[4])), int(float(v[5]))])

    return np.array(verts), np.array(colors) if has_color and colors else None


def save_ply(path, points, colors=None):
    """PLY 파일 저장"""
    if len(points) == 0:
        return

    print(f"저장 중: {path} ({len(points):,} points)")

    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            p = points[i]
            if colors is not None:
                c = f"{colors[i][0]} {colors[i][1]} {colors[i][2]}"
            else:
                c = ""
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c}\n")


def convert_to_global(video_id, base_dir='.'):
    """aligned_final.ply를 전역 UTM 좌표로 변환"""
    base_path = Path(base_dir)

    # 파일 경로
    gps_json = base_path / 'gps_output' / video_id / 'gps_path.json'
    input_ply = base_path / 'map_output' / video_id / 'aligned_final.ply'
    output_ply = base_path / 'map_output' / video_id / 'aligned_global.ply'

    # 파일 존재 확인
    if not gps_json.exists():
        print(f"GPS 파일 없음: {gps_json}")
        return None

    if not input_ply.exists():
        print(f"입력 파일 없음: {input_ply}")
        return None

    # GPS origin 로드
    with open(gps_json) as f:
        gps_data = json.load(f)

    origin = gps_data['origin']
    origin_lat = origin['lat']
    origin_lon = origin['lon']
    origin_alt = origin['alt']

    print(f"\n=== {video_id} 변환 시작 ===")
    print(f"Origin: {origin_lat:.6f}°N, {origin_lon:.6f}°E, {origin_alt:.1f}m")

    # Origin을 UTM으로 변환
    utm_e, utm_n, utm_zone = latlon_to_utm(origin_lat, origin_lon)
    print(f"UTM Zone {utm_zone}: E {utm_e:.2f}, N {utm_n:.2f}")

    # PLY 로드
    points, colors = load_ply_vertices(str(input_ply))
    if len(points) == 0:
        print("포인트 없음")
        return None

    print(f"로드됨: {len(points):,} points")

    # 로컬 좌표를 전역 UTM 좌표로 변환
    # 로컬 좌표계: x=동쪽, y=북쪽 (align_by_gps.py에서 GPS 방향 기준)
    global_points = points.copy()
    global_points[:, 0] += utm_e  # X -> Easting
    global_points[:, 1] += utm_n  # Y -> Northing
    global_points[:, 2] += origin_alt  # Z -> 절대 고도

    # 저장
    save_ply(str(output_ply), global_points, colors)
    print(f"완료: {output_ply}")

    return {
        'video_id': video_id,
        'utm_zone': utm_zone,
        'origin_utm': (utm_e, utm_n, origin_alt),
        'point_count': len(points)
    }


def main():
    parser = argparse.ArgumentParser(description='포인트 클라우드를 전역 UTM 좌표로 변환')
    parser.add_argument('--video', type=str, help='특정 비디오 ID (미지정시 전체 처리)')
    parser.add_argument('--base-dir', type=str, default='.', help='기본 디렉토리')
    args = parser.parse_args()

    if not HAS_PYPROJ:
        print("\n오류: pyproj가 필요합니다.")
        print("설치: pip install pyproj")
        return

    results = []

    if args.video:
        # 단일 비디오 처리
        result = convert_to_global(args.video, args.base_dir)
        if result:
            results.append(result)
    else:
        # 전체 비디오 처리
        map_dir = Path(args.base_dir) / 'map_output'
        if map_dir.exists():
            for video_dir in sorted(map_dir.iterdir()):
                if video_dir.is_dir():
                    result = convert_to_global(video_dir.name, args.base_dir)
                    if result:
                        results.append(result)

    # 결과 요약
    if results:
        print("\n" + "=" * 50)
        print("변환 완료 요약")
        print("=" * 50)

        zones = set(r['utm_zone'] for r in results)
        if len(zones) == 1:
            print(f"UTM Zone: {zones.pop()} (모든 파일 동일)")
        else:
            print(f"경고: 여러 UTM Zone 사용됨: {zones}")
            print("  -> 서로 다른 zone의 파일은 CloudCompare에서 위치가 맞지 않을 수 있음")

        total_points = sum(r['point_count'] for r in results)
        print(f"총 {len(results)}개 비디오, {total_points:,} points 변환됨")
        print("\nCloudCompare에서 aligned_global.ply 파일들을 열면 자동 정렬됩니다!")


if __name__ == '__main__':
    main()
