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

### 5. 포인트 클라우드 정합 (align_pointcloud.py)

청크별로 분리된 포인트 클라우드를 Umeyama 알고리즘으로 정합하여 하나의 통합된 포인트 클라우드를 생성합니다.

```bash
# 기본 사용
python align_pointcloud.py -i map_output/비디오이름/pointcloud

# 출력 경로 지정
python align_pointcloud.py -i map_output/비디오이름/pointcloud -o output.ply

# 상세 로그 숨기기
python align_pointcloud.py -i map_output/비디오이름/pointcloud -q
```

**주요 기능:**
- **Umeyama 알고리즘**: 오버랩 프레임을 이용하여 인접 청크 간 최적의 rigid transformation 계산
- **자동 정합**: 모든 청크를 첫 번째 청크의 좌표계로 자동 통합
- **중복 제거**: 오버랩 영역의 중복 포인트 자동 제거
- **색상 보존**: 원본 포인트의 RGB 색상 정보 유지

**옵션:**
- `-i, --input`: pointcloud 디렉토리 경로 (chunks.json이 있는 디렉토리) [필수]
- `-o, --output`: 출력 PLY 파일 경로 (기본값: input/aligned_combined.ply)
- `-q, --quiet`: 상세 로그 출력 안 함

**처리 과정:**
1. `chunks.json` 메타데이터에서 청크 정보 로드
2. 인접 청크 간 오버랩 프레임에서 대응점 추출
3. Umeyama 알고리즘으로 변환 행렬(회전 + 이동) 계산
4. 누적 변환을 적용하여 모든 청크를 첫 번째 청크 좌표계로 통합
5. 중복 프레임 제거 후 최종 PLY 파일 저장

**출력 파일:**
- `map_output/비디오이름/pointcloud/aligned_combined.ply`: 정합된 통합 포인트 클라우드

**참고:**
- 청크가 1개만 있는 경우 정합 없이 모든 포인트를 단순 병합합니다.
- 오버랩 프레임이 없는 청크 간에는 이전 변환을 그대로 사용하여 드리프트가 발생할 수 있습니다.


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

