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
├── input/          # GoPro 영상 파일을 여기에 넣으세요
├── output/         # 추출된 GPS 데이터가 여기에 저장됩니다
├── extract_gps.py
└── requirements.txt
```

## 사용법

### 배치 모드 (기본) - input 폴더의 모든 영상 처리

```bash
# 기본 사용 (JSON 형식)
python extract_gps.py

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

## 추출되는 데이터

- timestamp: 시간 정보
- latitude: 위도
- longitude: 경도
- altitude: 고도 (미터)
- speed_2d: 2D 속도
- speed_3d: 3D 속도

## 지원 파일 형식

- `.mp4`, `.MP4`
- `.mov`, `.MOV`
