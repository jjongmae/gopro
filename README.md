# GoPro GPS Extractor

GoPro 영상에서 GPS/GIS 정보를 추출하는 Python 스크립트입니다.

## 설치

### 시스템 패키지 (필수)
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Python 의존성
```bash
# venv 환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 폴더 구조

```
gopro/
├── video/          # GoPro 영상 파일을 여기에 넣으세요
├── shp/            # 추출된 GPS 데이터가 여기에 저장됩니다
├── extract_gps.py
└── requirements.txt
```

## 사용법

### 배치 모드 (기본) - video 폴더의 모든 영상 처리

```bash
# 기본 사용 (SHP 형식)
python extract_gps.py

# JSON 형식으로 저장
python extract_gps.py -f json

# CSV 형식으로 저장
python extract_gps.py -f csv

# GPX 형식으로 저장
python extract_gps.py -f gpx

# 입력/출력 폴더 지정
python extract_gps.py -i ./my_videos -o ./my_output
```

### 단일 파일 모드

```bash
# 단일 파일 처리
python extract_gps.py --single video.MP4

# 출력 형식 지정
python extract_gps.py --single video.MP4 -f gpx -o ./output
```

## 출력 형식

- **JSON**: GPS 데이터를 JSON 배열로 저장
- **CSV**: 스프레드시트에서 열 수 있는 CSV 형식
- **GPX**: GPS 교환 형식 (Google Earth, QGIS 등에서 사용 가능)
- **SHP**: Shapefile 형식 (QGIS, ArcGIS 등 GIS 소프트웨어에서 사용 가능, WGS84 좌표계)

## 추출되는 데이터

- frame_idx: 비디오 프레임 인덱스 (객체 인식 결과와 매칭용)
- timestamp: 시간 정보
- latitude: 위도
- longitude: 경도
- altitude: 고도 (미터)
- speed: 속도 (m/s, 미터/초)
  - km/h로 변환하려면 × 3.6 (예: 8.913 m/s = 32.09 km/h)

## 지원 파일 형식

- `.mp4`, `.MP4`
- `.mov`, `.MOV`
