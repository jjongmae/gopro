# GoPro GPS & Depth Extractor

GoPro 영상에서 GPS 정보와 depth map을 추출하는 Python 스크립트입니다.

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

# 기본 의존성 설치
pip install -r requirements.txt

# MapAnything 설치 (depth 추출용)
pip install "git+https://github.com/facebookresearch/map-anything.git"
```

## 폴더 구조

```
gopro/
├── video/              # GoPro 영상 파일을 여기에 넣으세요
├── gps_output/         # GPS 데이터 출력
│   └── 동영상이름/
│       └── 동영상이름_gps.shp
├── map_output/         # Depth map 출력
│   └── 동영상이름/
│       ├── frames/     # 원본 프레임 이미지
│       ├── depth/      # Depth map (.npy) + Intrinsics
│       └── pointcloud/ # 포인트 클라우드 (.ply)
├── extract_gps.py      # GPS 추출 스크립트
├── extract_depth.py    # Depth map 추출 스크립트
└── requirements.txt
```

## 사용법

### 1. GPS 추출 (extract_gps.py)

```bash
# 기본 사용 (video 폴더의 모든 영상 → SHP 형식)
python extract_gps.py

# JSON/CSV/GPX 형식으로 저장
python extract_gps.py -f json
python extract_gps.py -f csv
python extract_gps.py -f gpx

# 단일 파일 처리
python extract_gps.py --single video/GH013057.MP4

# 입력/출력 폴더 지정
python extract_gps.py -i ./my_videos -o ./my_output
```

### 2. Depth Map 추출 (extract_depth.py)

```bash
# 기본 사용 (video 폴더의 모든 영상)
python extract_depth.py

# 단일 파일 처리
python extract_depth.py --single video/GH013057.MP4

# 옵션
python extract_depth.py -i ./video -o ./map_output --frame-skip 1 --device cuda
```

**옵션:**
- `-i, --input`: 입력 폴더 (기본값: ./video)
- `-o, --output`: 출력 폴더 (기본값: ./map_output)
- `--single`: 단일 파일 처리
- `--frame-skip`: 프레임 건너뛰기 간격 (기본값: 1, 모든 프레임)
- `--device`: cuda 또는 cpu (기본값: cuda)

## 출력 형식

### GPS 출력
- **SHP**: Shapefile 형식 (QGIS, ArcGIS 등 GIS 소프트웨어에서 사용 가능, WGS84 좌표계)
- **JSON**: GPS 데이터를 JSON 배열로 저장
- **CSV**: 스프레드시트에서 열 수 있는 CSV 형식
- **GPX**: GPS 교환 형식 (Google Earth, QGIS 등에서 사용 가능)

### Depth Map 출력
- **frames/**: 비디오에서 추출한 원본 프레임 이미지 (.jpg)
- **depth/**: 각 프레임의 depth map (.npy) + 카메라 intrinsics (.npy)
- **pointcloud/**: 3D 포인트 클라우드 (.ply)

## 추출되는 데이터

### GPS 데이터
| 필드 | 설명 |
|------|------|
| frame_idx | 비디오 프레임 인덱스 (depth map과 매칭용) |
| timestamp | 시간 정보 |
| latitude | 위도 |
| longitude | 경도 |
| altitude | 고도 (미터) |
| speed | 속도 (m/s) - km/h로 변환: × 3.6 |

### Depth 데이터
| 파일 | 설명 |
|------|------|
| frame_XXXXXX_depth.npy | 픽셀별 거리 (미터) |
| frame_XXXXXX_intrinsics.npy | 카메라 내부 파라미터 (fx, fy, cx, cy) |
| frame_XXXXXX.ply | 3D 포인트 클라우드 (색상 포함) |

## GPS와 Depth 매핑

타임랩스 영상의 경우 GPS frame_idx와 비디오 프레임이 1:1 대응됩니다:
- GPS frame_idx 0 → depth/frame_000000_depth.npy
- GPS frame_idx 1 → depth/frame_000001_depth.npy
- ...

## 도로 너비 측정 (예정)

depth map과 intrinsics를 사용하여 이미지의 두 점 사이 실제 거리를 계산할 수 있습니다:

```python
import numpy as np

# 데이터 로드
depth = np.load('depth/frame_000000_depth.npy')
intrinsics = np.load('depth/frame_000000_intrinsics.npy')
fx, fy = intrinsics[0, 0], intrinsics[1, 1]
cx, cy = intrinsics[0, 2], intrinsics[1, 2]

# 픽셀 좌표 (x1, y1), (x2, y2)에서 3D 좌표 계산
z1 = depth[y1, x1]
X1 = (x1 - cx) * z1 / fx
Y1 = (y1 - cy) * z1 / fy

z2 = depth[y2, x2]
X2 = (x2 - cx) * z2 / fx
Y2 = (y2 - cy) * z2 / fy

# 두 점 사이 거리 = 도로 너비
width = np.sqrt((X2-X1)**2 + (Y2-Y1)**2 + (z2-z1)**2)
```

## 지원 파일 형식

- `.mp4`, `.MP4`
- `.mov`, `.MOV`
