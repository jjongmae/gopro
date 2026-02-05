#!/usr/bin/env python3
"""
GoPro 영상에서 GPS/GIS 정보를 추출하는 스크립트
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from gopro2gpx.gpmf import Parser
    from gopro2gpx.gopro2gpx import BuildGPSPoints
    from gopro2gpx.config import Config
except ImportError as e:
    print(f"Error importing gopro2gpx: {e}")
    print("\n필요한 라이브러리를 설치하세요:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def extract_gps_data(video_path):
    """GoPro 영상에서 GPS 데이터 추출"""
    print(f"Processing: {video_path}")

    try:
        # gopro2gpx 라이브러리를 사용하여 GPS 데이터 추출
        config = Config(
            input_file=str(video_path),
            outputfile=None,
            format="GPX",
            verbose=False,
            skip=True
        )
        parser = Parser(config)
        data = parser.readFromMP4()
        points = BuildGPSPoints(data, skip=config.skip)

        if len(points) == 0:
            return None

        gps_data = []
        for idx, point in enumerate(points):
            gps_entry = {
                'frame_idx': idx,
                'timestamp': point.time.isoformat() if point.time else None,
                'latitude': point.latitude,
                'longitude': point.longitude,
                'altitude': point.elevation,
                'speed': point.speed,
            }
            gps_data.append(gps_entry)

        return gps_data

    except Exception as e:
        print(f"  Error: {e}")
        return None


def save_to_json(data, output_path):
    """JSON 형식으로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")


def save_to_csv(data, output_path):
    """CSV 형식으로 저장"""
    import csv

    if not data:
        print("No data to save")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved to: {output_path}")


def save_to_shp(data, output_path):
    """SHP(Shapefile) 형식으로 저장"""
    import shapefile

    if not data:
        print("저장할 데이터가 없습니다")
        return

    # Shapefile writer 생성 (Point 타입)
    w = shapefile.Writer(str(output_path).replace('.shp', ''))

    # 필드 정의
    w.field('frame_idx', 'N', decimal=0)  # 프레임 인덱스
    w.field('timestamp', 'C', size=30)  # 문자열
    w.field('latitude', 'N', decimal=8)  # 숫자
    w.field('longitude', 'N', decimal=8)
    w.field('altitude', 'N', decimal=2)
    w.field('speed', 'N', decimal=4)

    # 포인트 추가
    for point in data:
        if point['latitude'] and point['longitude']:
            w.point(point['longitude'], point['latitude'])
            w.record(
                point['frame_idx'],
                point['timestamp'] or '',
                point['latitude'],
                point['longitude'],
                point['altitude'] or 0,
                point['speed'] or 0
            )

    w.close()

    # PRJ 파일 생성 (WGS84 좌표계)
    prj_path = str(output_path).replace('.shp', '.prj')
    with open(prj_path, 'w') as prj:
        prj.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                  'SPHEROID["WGS_1984",6378137,298.257223563]],'
                  'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')

    print(f"Saved to: {output_path}")


def save_to_gpx(data, output_path):
    """GPX 형식으로 저장"""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    gpx = Element('gpx', {
        'version': '1.1',
        'creator': 'GoPro GPS Extractor',
        'xmlns': 'http://www.topografix.com/GPX/1/1'
    })

    trk = SubElement(gpx, 'trk')
    SubElement(trk, 'name').text = 'GoPro Track'
    trkseg = SubElement(trk, 'trkseg')

    for point in data:
        if point['latitude'] and point['longitude']:
            trkpt = SubElement(trkseg, 'trkpt', {
                'lat': str(point['latitude']),
                'lon': str(point['longitude'])
            })

            if point['altitude']:
                SubElement(trkpt, 'ele').text = str(point['altitude'])

            if point['timestamp']:
                SubElement(trkpt, 'time').text = point['timestamp']

            # frame_idx를 extensions에 추가
            extensions = SubElement(trkpt, 'extensions')
            SubElement(extensions, 'frame_idx').text = str(point['frame_idx'])

    xml_str = minidom.parseString(tostring(gpx)).toprettyxml(indent='  ')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"Saved to: {output_path}")


def process_video(video_path, output_dir, output_format):
    """단일 영상 파일 처리"""
    gps_data = extract_gps_data(str(video_path))

    if not gps_data:
        print(f"  ⚠ No GPS data found in {video_path.name}")
        return False

    print(f"  ✓ Extracted {len(gps_data)} GPS points")

    output_path = output_dir / f"{video_path.stem}_gps.{output_format}"

    if output_format == 'json':
        save_to_json(gps_data, output_path)
    elif output_format == 'csv':
        save_to_csv(gps_data, output_path)
    elif output_format == 'gpx':
        save_to_gpx(gps_data, output_path)
    elif output_format == 'shp':
        save_to_shp(gps_data, output_path)

    return True


def process_batch(input_dir, output_dir, output_format):
    """input 폴더의 모든 영상 처리"""
    video_extensions = ['.mp4', '.MP4', '.mov', '.MOV']
    video_files = []

    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"\n=== Found {len(video_files)} video file(s) ===\n")

    success_count = 0
    for video_file in sorted(video_files):
        print(f"[{success_count + 1}/{len(video_files)}] {video_file.name}")
        if process_video(video_file, output_dir, output_format):
            success_count += 1
        print()

    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {success_count}/{len(video_files)} files")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='GoPro 영상에서 GPS/GIS 정보 추출'
    )
    parser.add_argument(
        '-i', '--input',
        help='입력 폴더 경로 (기본값: ./video)',
        default='./video'
    )
    parser.add_argument(
        '-o', '--output',
        help='출력 폴더 경로 (기본값: ./shp)',
        default='./shp'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'csv', 'gpx', 'shp'],
        default='shp',
        help='출력 형식 (기본값: shp)'
    )
    parser.add_argument(
        '--single',
        help='단일 영상 파일 처리 (배치 모드 대신)',
        default=None
    )

    args = parser.parse_args()

    # 단일 파일 모드
    if args.single:
        video_path = Path(args.single)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing single file: {video_path.name}\n")
        if process_video(video_path, output_dir, args.format):
            print("✓ Success")
        else:
            sys.exit(1)
        return

    # 배치 모드 (기본)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print(f"Please create the directory and add GoPro video files.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    process_batch(input_dir, output_dir, args.format)


if __name__ == '__main__':
    main()
