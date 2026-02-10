#!/usr/bin/env python3
"""
GPS 전처리 스크립트 (preprocess_gps.py)

1. Shapefile(.shp)에서 GPS 데이터를 읽어옵니다.
2. 이동 평균(Moving Average) 필터를 적용해 노이즈를 제거합니다.
3. 위도/경도를 미터 단위(Local ENU 좌표계)로 변환합니다.
4. 각 프레임별 위치와 진행 방향(Heading)을 계산하여 JSON으로 저장합니다.
"""

import os
import argparse
import json
import shapefile  # pyshp
import numpy as np
from pathlib import Path

def latlon_to_meters(lat, lon, lat0, lon0):
    """
    위도/경도를 기준점(lat0, lon0)에 대한 미터 단위(x, y)로 변환
    (간이 근사 계산 사용)
    """
    R = 6378137.0  # 지구 반지름 (미터)
    rad = np.pi / 180.0
    
    dlat = (lat - lat0) * rad
    dlon = (lon - lon0) * rad
    
    # x: 동쪽 방향 거리 (Longitude), y: 북쪽 방향 거리 (Latitude)
    x = R * np.cos(lat0 * rad) * dlon
    y = R * dlat
    
    return x, y

def smooth_data(data, window_size=5):
    """이동 평균 필터 적용"""
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    # 가장자리 처리를 위해 padding
    pad_size = window_size // 2
    padded = np.pad(data, (pad_size, pad_size), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    # 크기 맞추기 (가끔 convolve 결과가 1~2개 차이나는 경우 방지)
    if len(smoothed) > len(data):
        smoothed = smoothed[:len(data)]
    elif len(smoothed) < len(data):
        # 부족한 경우 원본 데이터로 채움 (거의 발생 안 함)
        diff = len(data) - len(smoothed)
        smoothed = np.concatenate((smoothed, data[-diff:]))
        
    return smoothed

def process_gps(video_id, base_dir='.'):
    base_path = Path(base_dir)
    shp_path = base_path / 'gps_output' / video_id / f'{video_id}_gps.shp'
    output_json = base_path / 'gps_output' / video_id / 'gps_path.json'
    
    if not shp_path.exists():
        print(f"✗ Shapefile을 찾을 수 없습니다: {shp_path}")
        return False
    
    print(f"처리 중: {video_id}")
    
    # 1. Shapefile 읽기
    sf = shapefile.Reader(str(shp_path))
    records = sf.records()
    shapes = sf.shapes()  # (lon, lat) 순서임에 주의
    
    # 데이터 추출
    frames = []
    lats = []
    lons = []
    alts = []
    
    for i, (record, shape) in enumerate(zip(records, shapes)):
        rec_dict = record.as_dict()
        frame_idx = rec_dict.get('frame_idx', i)
        
        lon, lat = shape.points[0]
        alt = rec_dict.get('altitude', 0)
        
        frames.append(frame_idx)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
    
    # 2. 노이즈 제거 (Smoothing)
    print("  - GPS 노이즈 제거 (Smoothing)...")
    lats_smooth = smooth_data(lats, window_size=10)
    lons_smooth = smooth_data(lons, window_size=10)
    alts_smooth = smooth_data(alts, window_size=10)
    
    # 3. 미터 변환 (첫 번째 점을 원점 (0,0,0)으로 설정)
    lat0, lon0, alt0 = lats_smooth[0], lons_smooth[0], alts_smooth[0]
    
    processed_points = []
    
    for i in range(len(frames)):
        x, y = latlon_to_meters(lats_smooth[i], lons_smooth[i], lat0, lon0)
        z = alts_smooth[i] - alt0
        
        processed_points.append({
            "frame_idx": int(frames[i]),
            "lat": float(lats_smooth[i]),
            "lon": float(lons_smooth[i]),
            "alt": float(alts_smooth[i]),
            "x": float(x), # East (Meters)
            "y": float(y), # North (Meters)
            "z": float(z)  # Up (Meters)
        })
    
    # 4. Heading(진행 방향) 및 속도 벡터 계산
    for i in range(len(processed_points) - 1):
        curr = processed_points[i]
        next_p = processed_points[i+1]
        
        dx = next_p['x'] - curr['x']
        dy = next_p['y'] - curr['y']
        dz = next_p['z'] - curr['z']
        
        # 방위각 (북쪽 0도 기준 시계방향? 여기선 수학적 각도 사용: East=0, CCW)
        # 하지만 단순히 벡터 정렬용이므로 atan2(dy, dx) 그대로 사용
        heading_rad = np.arctan2(dy, dx)
        
        curr['heading_rad'] = float(heading_rad)
        curr['vec'] = [float(dx), float(dy), float(dz)]

    # 마지막 프레임은 이전 프레임 정보 복사
    processed_points[-1]['heading_rad'] = processed_points[-2]['heading_rad']
    processed_points[-1]['vec'] = processed_points[-2]['vec']
    
    # 5. JSON 저장
    output_data = {
        "video_id": video_id,
        "origin": {"lat": lat0, "lon": lon0, "alt": alt0},
        "points": processed_points
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"  ✓ 저장 완료: {output_json}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='특정 비디오 ID')
    parser.add_argument('--base-dir', type=str, default='.')
    args = parser.parse_args()
    
    if args.video:
        process_gps(args.video, args.base_dir)
    else:
        # map_output 폴더 기준으로 모든 비디오 찾기
        base = Path(args.base_dir)
        map_dir = base / 'gps_output'
        if map_dir.exists():
            for d in map_dir.iterdir():
                if d.is_dir():
                    process_gps(d.name, args.base_dir)