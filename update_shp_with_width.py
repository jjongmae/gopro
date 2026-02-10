#!/usr/bin/env python3
"""
도로 폭 측정 데이터를 Shapefile에 추가하는 스크립트

이 스크립트는 map_output 디렉토리의 width_measurements.csv 파일을 읽어서
해당하는 GPS Shapefile에 road_width 속성을 추가합니다.

사용법:
    python update_shp_with_width.py
    또는
    python update_shp_with_width.py --video-id GH013057
"""

import os
import csv
import shapefile
import shutil
from pathlib import Path
import argparse


def extract_frame_number(frame_str):
    """
    프레임 문자열에서 숫자를 추출합니다.
    
    Args:
        frame_str: 'frame_000000' 형식의 문자열
        
    Returns:
        int: 프레임 번호 (예: 0)
    """
    # 'frame_' 접두사를 제거하고 숫자로 변환
    return int(frame_str.replace('frame_', ''))


def read_width_measurements(csv_path):
    """
    CSV 파일에서 도로 폭 측정 데이터를 읽습니다.
    
    Args:
        csv_path: CSV 파일 경로
        
    Returns:
        dict: {frame_idx: road_width} 형식의 딕셔너리
    """
    measurements = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_str = row['Frame']
            frame_idx = extract_frame_number(frame_str)
            road_width = float(row['Distance_Meter'])
            measurements[frame_idx] = road_width
            
    print(f"✓ CSV에서 {len(measurements)}개의 측정 데이터를 읽었습니다.")
    return measurements


def update_shapefile_with_width(shp_path, measurements, output_path=None):
    """
    Shapefile에 road_width 필드를 추가하고 측정값을 업데이트합니다.
    
    Args:
        shp_path: 원본 Shapefile 경로
        measurements: {frame_idx: road_width} 딕셔너리
        output_path: 출력 Shapefile 경로 (None이면 원본을 백업하고 덮어씀)
    """
    # 출력 경로 설정
    if output_path is None:
        # 원본 파일명 기반으로 새 파일명 생성 (예: video_id_gps_width.shp)
        output_path = shp_path.replace('.shp', '_width.shp')
        print(f"✓ 새 파일로 저장합니다: {output_path}")
    
    # 원본 Shapefile 읽기
    sf = shapefile.Reader(shp_path)
    
    # 기존 필드 정보 가져오기
    fields = sf.fields[1:]  # 첫 번째 DeletionFlag 필드 제외
    field_names = [field[0] for field in fields]
    
    # road_width 필드가 이미 있는지 확인
    if 'road_width' in field_names:
        print("⚠ 'road_width' 필드가 이미 존재합니다. 값을 업데이트합니다.")
        road_width_exists = True
    else:
        print("✓ 새로운 'road_width' 필드를 추가합니다.")
        road_width_exists = False
    
    # 새 Shapefile 작성
    # temp 파일로 먼저 작성
    temp_shp_path = output_path.replace('.shp', '_temp.shp')
    w = shapefile.Writer(temp_shp_path)
    
    # 필드 정의 복사
    for field in fields:
        w.field(*field)
    
    # road_width 필드 추가 (아직 없는 경우)
    if not road_width_exists:
        w.field('road_width', 'N', decimal=4, size=10)  # 숫자 필드, 소수점 4자리
    
    # 레코드와 지오메트리 복사
    records = sf.records()
    shapes = sf.shapes()
    
    matched_count = 0
    unmatched_count = 0
    
    for i, (record, shape) in enumerate(zip(records, shapes)):
        # 기존 레코드를 딕셔너리로 변환
        rec_dict = record.as_dict()
        frame_idx = rec_dict['frame_idx']
        
        # 측정값이 있는 경우에만 저장 (매칭 안 되면 삭제)
        if frame_idx in measurements:
            # 지오메트리 추가
            w.shape(shape)
            
            # 레코드 값 준비
            if road_width_exists:
                # 기존 필드가 있으면 값 업데이트
                rec_values = list(record)
                road_width_idx = field_names.index('road_width')
                rec_values[road_width_idx] = measurements[frame_idx]
                w.record(*rec_values)
            else:
                # 새 필드 추가
                rec_values = list(record)
                rec_values.append(measurements[frame_idx])
                w.record(*rec_values)
                
            matched_count += 1
        else:
            # 매칭되지 않으면 건너뜀 (삭제됨)
            unmatched_count += 1
    
    w.close()
    sf.close()
    
    # 임시 파일을 최종 파일로 이동
    # .shp, .shx, .dbf 파일 이동
    for ext in ['.shp', '.shx', '.dbf']:
        temp_file = output_path.replace('.shp', f'_temp{ext}')
        final_file = output_path.replace('.shp', ext)
        if os.path.exists(temp_file):
            shutil.move(temp_file, final_file)
    
    # .prj 파일 복사 (투영 정보)
    prj_src = shp_path.replace('.shp', '.prj')
    prj_dst = output_path.replace('.shp', '.prj')
    if os.path.exists(prj_src) and prj_src != prj_dst:
        shutil.copy2(prj_src, prj_dst)
        
    # .cpg 파일이 있다면 복사 (인코딩 정보)
    cpg_src = shp_path.replace('.shp', '.cpg')
    cpg_dst = output_path.replace('.shp', '.cpg')
    if os.path.exists(cpg_src) and cpg_src != cpg_dst:
        shutil.copy2(cpg_src, cpg_dst)
    
    print(f"✓ Shapefile 업데이트 완료:")
    print(f"  - 저장된 레코드: {matched_count}개")
    print(f"  - 삭제된 레코드(매칭 실패): {unmatched_count}개")
    print(f"  - 출력 파일: {output_path}")


def process_video(video_id, base_dir='.'):
    """
    특정 비디오 ID에 대해 CSV와 Shapefile을 처리합니다.
    
    Args:
        video_id: 비디오 ID (예: 'GH013057')
        base_dir: 프로젝트 기본 디렉토리
    """
    base_path = Path(base_dir)
    
    # CSV 파일 경로
    csv_path = base_path / 'map_output' / video_id / 'measurements' / 'width_measurements.csv'
    
    # Shapefile 경로
    shp_path = base_path / 'gps_output' / video_id / f'{video_id}_gps.shp'
    
    # 파일 존재 확인
    if not csv_path.exists():
        print(f"✗ CSV 파일을 찾을 수 없습니다: {csv_path}")
        return False
    
    if not shp_path.exists():
        print(f"✗ Shapefile을 찾을 수 없습니다: {shp_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"비디오 ID: {video_id}")
    print(f"{'='*60}")
    print(f"CSV 파일: {csv_path}")
    print(f"Shapefile: {shp_path}")
    print()
    
    # 측정 데이터 읽기
    measurements = read_width_measurements(csv_path)
    
    # Shapefile 업데이트 (새 파일로 저장)
    output_shp_path = str(shp_path).replace('.shp', '_width.shp')
    update_shapefile_with_width(str(shp_path), measurements, output_path=output_shp_path)
    
    print(f"\n✓ {video_id} 처리 완료!\n")
    return True


def process_all_videos(base_dir='.'):
    """
    map_output 디렉토리의 모든 비디오를 처리합니다.
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
    """
    base_path = Path(base_dir)
    map_output_dir = base_path / 'map_output'
    
    if not map_output_dir.exists():
        print(f"✗ map_output 디렉토리를 찾을 수 없습니다: {map_output_dir}")
        return
    
    # map_output의 모든 하위 디렉토리 찾기
    video_dirs = [d for d in map_output_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("✗ 처리할 비디오 디렉토리가 없습니다.")
        return
    
    print(f"총 {len(video_dirs)}개의 비디오 디렉토리를 찾았습니다.")
    
    success_count = 0
    fail_count = 0
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        csv_file = video_dir / 'measurements' / 'width_measurements.csv'
        
        # CSV 파일이 있는 경우만 처리
        if csv_file.exists():
            if process_video(video_id, base_dir):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"⊘ {video_id}: width_measurements.csv 파일이 없어 건너뜁니다.")
    
    print(f"\n{'='*60}")
    print(f"전체 처리 완료")
    print(f"{'='*60}")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")


def main():
    parser = argparse.ArgumentParser(
        description='도로 폭 측정 데이터를 Shapefile에 추가합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 모든 비디오 처리
  python update_shp_with_width.py
  
  # 특정 비디오만 처리
  python update_shp_with_width.py --video-id GH013057
  
  # 다른 디렉토리에서 실행
  python update_shp_with_width.py --base-dir /path/to/project
        """
    )
    
    parser.add_argument(
        '-v', '--video',
        type=str,
        help='처리할 비디오 이름 (예: GH013057). 지정하지 않으면 모든 비디오를 처리합니다.'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='프로젝트 기본 디렉토리 (기본값: 현재 디렉토리)'
    )
    
    args = parser.parse_args()
    
    if args.video:
        # 특정 비디오만 처리
        process_video(args.video, args.base_dir)
    else:
        # 모든 비디오 처리
        process_all_videos(args.base_dir)


if __name__ == '__main__':
    main()
