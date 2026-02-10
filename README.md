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
│       ├── 동영상이름_gps.shp
│       └── gps_path.json       # 전처리된 GPS 경로 데이터
├── map_output/         # Depth map 출력
│   └── 동영상이름/
│       ├── frames/     # 원본 프레임 이미지
│       ├── depth/      # Depth map (.npy) + Intrinsics
│       └── pointcloud/ # 포인트 클라우드 (.ply)
│       └── aligned_final.ply   # 최종 정합된 포인트 클라우드
├── extract_gps.py      # GPS 추출 스크립트
├── extract_depth.py    # Depth map 추출 스크립트
├── preprocess_gps.py   # GPS 데이터 전처리 스크립트
├── align_by_gps.py     # GPS 기반 정합 스크립트
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

### 3. 도로 너비 측정 GUI (measure_width.py)

추출된 Depth Map을 이용하여 화면상에서 직접 점을 찍어 거리를 측정하는 GUI 도구입니다.

```bash
# 실행
python measure_width.py
```

**주요 기능:**
- 📏 **거리 측정**: 이미지에서 두 점을 클릭하여 실제 거리(미터) 측정
- 💾 **자동 저장**: 측정 즉시 자동으로 CSV 파일에 저장
- 🎨 **Depth 맵 시각화**: 컬러맵으로 depth 정보를 오버레이 표시
- 🔄 **측정값 복원**: 이전/다음 프레임 이동 시 저장된 측정값 자동 표시
- 🔧 **자동 보간**: Depth 값이 0인 픽셀은 주변 5x5 영역의 평균값으로 자동 보간

**사용 방법:**
1. 프로그램 실행 후 **"폴더 열기"** 버튼 클릭
2. `map_output/비디오이름` 폴더 선택 (예: `map_output/GH013057`)
3. 리스트에서 이미지 선택
4. 이미지 상에서 측정할 **두 지점 클릭** (빨간 점과 노란 선 생성)
5. 화면에 거리(미터) 즉시 표시 및 자동 저장
6. **"Depth 맵 표시"** 버튼으로 depth 정보 시각화 가능
7. 이전/다음 버튼으로 프레임 이동 시 저장된 측정값 자동 표시

**출력 파일:**
- `map_output/비디오이름/measurements/width_measurements.csv`
  - Frame: 프레임 이름
  - P1_X, P1_Y: 첫 번째 점 좌표
  - P2_X, P2_Y: 두 번째 점 좌표
  - P1_Depth, P2_Depth: 각 점의 깊이(미터)
  - Distance_Meter: 측정된 거리(미터)

### 4. Shapefile에 도로 폭 병합 (update_shp_with_width.py)

측정된 도로 폭 데이터(CSV)를 GPS Shapefile에 병합하여 하나의 파일로 만듭니다.

```bash
# 기본 사용 (모든 비디오 처리)
python update_shp_with_width.py

# 특정 비디오 처리 (비디오 이름 지정)
python update_shp_with_width.py --video GH013057
# 또는
python update_shp_with_width.py -v GH013057
```

**주요 기능:**
- **데이터 매핑:** CSV의 프레임 번호와 Shapefile의 `frame_idx`를 자동으로 매칭
- **자동 필터링:** 도로 폭 측정값이 없는 GPS 포인트는 Shapefile에서 **자동으로 삭제**합니다. (측정된 지점만 남김)
- **백업 생성:** 원본 Shapefile은 `_backup.shp`로 자동 백업됩니다.

**결과물:**
- `gps_output/비디오이름/비디오이름_gps.shp` 파일에 `road_width` 필드가 추가되고, 측정된 값(미터)이 저장됩니다.

### 5. GPS 기반 포인트 클라우드 정합

기존의 단순 정합 방식(`align_pointcloud.py`) 대신, GPS 경로 데이터를 기반으로 더욱 정확한 대규모 정합을 수행합니다. 이 과정은 **전처리**와 **정합** 두 단계로 나뉩니다.

#### 5-1. GPS 데이터 전처리 (preprocess_gps.py)

Shapefile(.shp) 형식의 GPS 데이터를 읽어 노이즈를 제거하고, 미터 단위의 로컬 좌표계로 변환합니다.

```bash
# 기본 사용 (모든 비디오 처리)
python preprocess_gps.py

# 특정 비디오 처리
python preprocess_gps.py --video GH013057
```

**기능:**
- **노이즈 제거**: 이동 평균 필터(Moving Average)를 적용하여 GPS 튀는 현상 보정
- **좌표 변환**: 위도/경도를 미터 단위(Local ENU)로 변환
- **데이터 생성**: 정합에 필요한 `gps_path.json` 파일 생성

#### 5-2. 정합 실행 (align_by_gps.py)

전처리된 GPS 경로(`gps_path.json`)를 기반으로 분할된 포인트 클라우드 청크들을 하나의 전체 지도로 통합합니다.

```bash
# 기본 사용 (모든 비디오 처리)
python align_by_gps.py

# 특정 비디오 처리
python align_by_gps.py --video GH013057
```

**기능:**
- **No-Overlap 최적화**: 오버랩이 없는 영상에서도 GPS 경로를 따라 자연스럽게 이어지도록 정합
- **스케일 보정**: GPS 이동 거리와 Visual Odometry 이동 거리를 비교하여 스케일 자동 보정
- **좌표계 통합**: 모든 청크를 실제 지리적 위치(GPS) 기반의 좌표계로 변환

**출력 파일:**
- `map_output/비디오이름/aligned_final.ply`: GPS 좌표계로 통합된 최종 포인트 클라우드


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
| 필드       | 설명                                                 |
| ---------- | ---------------------------------------------------- |
| frame_idx  | 비디오 프레임 인덱스 (depth map과 매칭용)            |
| timestamp  | 시간 정보                                            |
| latitude   | 위도                                                 |
| longitude  | 경도                                                 |
| altitude   | 고도 (미터)                                          |
| speed      | 속도 (m/s) - km/h로 변환: × 3.6                      |
| road_width | 측정된 도로 폭 (미터) - 측정값 없을 시 레코드 삭제됨 |

### Depth 데이터
| 파일                        | 설명                                  |
| --------------------------- | ------------------------------------- |
| frame_XXXXXX_depth.npy      | 픽셀별 거리 (미터)                    |
| frame_XXXXXX_intrinsics.npy | 카메라 내부 파라미터 (fx, fy, cx, cy) |
| frame_XXXXXX.ply            | 3D 포인트 클라우드 (색상 포함)        |

## GPS와 Depth 매핑

타임랩스 영상의 경우 GPS frame_idx와 비디오 프레임이 1:1 대응됩니다:
- GPS frame_idx 0 → depth/frame_000000_depth.npy
- GPS frame_idx 1 → depth/frame_000001_depth.npy
- ...


## 지원 파일 형식

- `.mp4`, `.MP4`
- `.mov`, `.MOV`


