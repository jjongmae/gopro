#!/usr/bin/env python3
"""
GoPro 영상에서 추출된 Depth Map을 이용하여 도로 너비를 측정하는 GUI 도구
PyQt5를 사용하여 구현되었습니다.
"""

import sys
import os
import csv
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QListWidget, QSplitter)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import cv2

class ImageLabel(QLabel):
    """이미지를 표시하고 마우스 클릭 이벤트를 처리하는 위젯"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []  # 선택된 점들 [(x, y), (x, y)]
        self.original_image = None # 원본 OpenCV 이미지
        self.depth_overlay = None  # Depth 맵 오버레이 이미지
        self.show_depth = False    # Depth 맵 표시 여부
        self.pixmap_scale = 1.0 # 화면 표시 배율
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.parent_window = None

    def set_image(self, cv_image):
        """OpenCV 이미지를 설정"""
        self.original_image = cv_image
        self.depth_overlay = None  # 새 이미지 로드 시 depth 오버레이 초기화
        self.points = []  # 이미지 변경 시 점 초기화
        self.update_display()

    def update_display(self):
        """현재 창 크기에 맞춰 이미지를 리사이징하여 표시"""
        if self.original_image is None:
            self.setText("폴더를 열어 이미지를 불러오세요")
            return

        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Depth 오버레이 합성
        if self.show_depth and self.depth_overlay is not None:
            # Depth 오버레이를 원본 이미지 크기에 맞게 리사이즈
            depth_resized = cv2.resize(self.depth_overlay, 
                                      (rgb_image.shape[1], rgb_image.shape[0]), 
                                      interpolation=cv2.INTER_LINEAR)
            # 알파 블렌딩 (60% 원본, 40% depth)
            rgb_image = cv2.addWeighted(rgb_image, 0.6, depth_resized, 0.4, 0)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # QImage 생성
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # 라벨 크기에 맞춰 스케일링 (비율 유지)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

        # 스케일 비율 계산 (원본 좌표 복원용)
        self.pixmap_scale = scaled_pixmap.width() / w
        
    def set_depth_map(self, depth_map):
        """Depth 맵을 컬러맵으로 변환하여 오버레이 생성"""
        if depth_map is None:
            self.depth_overlay = None
            return
        
        # 유효한 depth 값만 추출 (0보다 큰 값)
        valid_depths = depth_map[depth_map > 0]
        
        if len(valid_depths) == 0:
            print("[WARNING] Depth 맵에 유효한 값이 없습니다.")
            self.depth_overlay = None
            return
        
        # 정규화를 위한 min/max 계산 (유효한 값만 사용)
        vmin = np.percentile(valid_depths, 5)   # 하위 5% 제거 (노이즈 방지)
        vmax = np.percentile(valid_depths, 95)  # 상위 5% 제거
        
        # Depth 맵 정규화 (0~255)
        depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        mask = depth_map > 0  # 유효한 영역 마스크
        
        # 유효한 영역만 정규화
        depth_normalized[mask] = np.clip(
            (depth_map[mask] - vmin) / (vmax - vmin) * 255, 0, 255
        ).astype(np.uint8)
        
        # 컬러맵 적용 (TURBO: 파랑(먼 곳) -> 빨강(가까운 곳))
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        
        # 유효하지 않은 영역(depth=0)은 검은색으로 표시
        depth_colored[~mask] = [0, 0, 0]
        
        # BGR -> RGB 변환
        self.depth_overlay = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
    def toggle_depth_overlay(self):
        """Depth 오버레이 표시/숨김 토글"""
        self.show_depth = not self.show_depth
        self.update_display()
        self.update()  # 점 다시 그리기
        
        # 레터박싱(여백) 고려한 오프셋 계산 -> mousePressEvent에서 처리
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 그려진 pixmap의 실제 위치 계산 (Center 정렬 때문)
        pm = self.pixmap()
        x_offset = (self.width() - pm.width()) // 2
        y_offset = (self.height() - pm.height()) // 2

        # 점 그리기
        painter.setPen(QPen(Qt.red, 5))
        painter.setBrush(Qt.red)
        
        display_points = []
        for pt in self.points:
            # 원본 좌표 -> 화면 좌표 변환
            sx = int(pt[0] * self.pixmap_scale) + x_offset
            sy = int(pt[1] * self.pixmap_scale) + y_offset
            display_points.append(QPoint(sx, sy))
            painter.drawEllipse(QPoint(sx, sy), 5, 5)

        # 선 그리기
        if len(display_points) == 2:
            painter.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))
            painter.drawLine(display_points[0], display_points[1])

    def mousePressEvent(self, event):
        if self.original_image is None:
            return

        if event.button() == Qt.LeftButton:
            # 점이 이미 2개면 초기화 후 다시 시작
            if len(self.points) >= 2:
                self.points = []
                if self.parent_window:
                    self.parent_window.clear_measurement()

            # 화면 클릭 좌표
            click_x = event.x()
            click_y = event.y()

            # Pixmap 내부 좌표로 변환
            pm = self.pixmap()
            if not pm: return
            
            x_offset = (self.width() - pm.width()) // 2
            y_offset = (self.height() - pm.height()) // 2
            
            # 이미지 영역 밖 클릭 무시
            if not (x_offset <= click_x < x_offset + pm.width() and 
                    y_offset <= click_y < y_offset + pm.height()):
                return

            # 원본 이미지 좌표로 복원
            img_x = int((click_x - x_offset) / self.pixmap_scale)
            img_y = int((click_y - y_offset) / self.pixmap_scale)

            # 좌표 클램핑 (안전장치)
            h, w = self.original_image.shape[:2]
            img_x = max(0, min(img_x, w - 1))
            img_y = max(0, min(img_y, h - 1))

            self.points.append((img_x, img_y))
            self.update() # 다시 그리기 시그널

            # 2개의 점이 모이면 계산 요청
            if len(self.points) == 2 and self.parent_window:
                self.parent_window.calculate_distance(self.points[0], self.points[1])

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GoPro 도로 폭 측정기")
        self.setGeometry(100, 100, 1200, 800)

        # 데이터 경로 및 상태
        self.current_video_dir = None
        self.frames_dir = None
        self.depth_dir = None
        self.image_files = []
        self.current_idx = 0
        
        # UI 초기화
        self.init_ui()

    def init_ui(self):
        # 메인 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 레이아웃: 좌측(리스트), 중앙(이미지), 하단(컨트롤)
        main_layout = QHBoxLayout(central_widget)
        
        # --- 좌측: 파일 리스트 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_open = QPushButton("폴더 열기 (map_output/비디오명)")
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_open.setMinimumHeight(40)
        left_layout.addWidget(self.btn_open)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_file_selected)
        left_layout.addWidget(self.list_widget)
        
        left_panel.setFixedWidth(250)
        main_layout.addWidget(left_panel)
        
        # --- 우측: 이미지 뷰어 및 정보 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 이미지 뷰어
        self.image_label = ImageLabel()
        self.image_label.parent_window = self
        self.image_label.setStyleSheet("background-color: #222; color: #fff;")
        self.image_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.image_label, stretch=1)

        # 정보 표시창 (계산 결과)
        self.info_label = QLabel("이미지를 불러온 후 두 점을 클릭하여 거리를 측정하세요.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc;")
        right_layout.addWidget(self.info_label)

        # 컨트롤 패널
        control_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("<< 이전")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("다음 >>")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_toggle_depth = QPushButton("Depth 맵 표시")
        self.btn_toggle_depth.clicked.connect(self.toggle_depth_view)
        self.btn_toggle_depth.setCheckable(True)
        self.btn_toggle_depth.setStyleSheet("""
            QPushButton {
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_toggle_depth)
        control_layout.addWidget(self.btn_next)
        
        right_layout.addLayout(control_layout)
        
        main_layout.addWidget(right_panel)

    def open_folder(self):
        """폴더 선택 다이얼로그"""
        # 기본적으로 map_output 폴더에서 시작하면 좋음
        start_dir = "./map_output" if os.path.exists("./map_output") else "."
        folder = QFileDialog.getExistingDirectory(self, "비디오 결과 폴더 선택", start_dir)
        
        if folder:
            self.load_data(folder)

    def load_data(self, video_dir_path):
        """선택된 폴더에서 프레임 및 depth 데이터 로드"""
        video_dir = Path(video_dir_path)
        frames_dir = video_dir / "frames"
        depth_dir = video_dir / "depth"

        if not frames_dir.exists() or not depth_dir.exists():
            QMessageBox.critical(self, "오류", 
                               f"올바른 데이터 폴더가 아닙니다.\n아래 폴더가 있어야 합니다:\n- {frames_dir}\n- {depth_dir}")
            return

        self.current_video_dir = video_dir
        self.frames_dir = frames_dir
        self.depth_dir = depth_dir
        
        # 이미지 파일 리스트 (.jpg)
        self.image_files = sorted(list(frames_dir.glob("*.jpg")))
        
        if not self.image_files:
            QMessageBox.warning(self, "경고", "frames 폴더에 이미지가 없습니다.")
            return

        # UI 업데이트
        self.list_widget.clear()
        for f in self.image_files:
            self.list_widget.addItem(f.name)
            
        self.setWindowTitle(f"GoPro 도로 폭 측정기 - {video_dir.name}")
        self.current_idx = 0
        self.list_widget.setCurrentRow(0) # 첫 번째 아이템 선택 -> on_file_selected 호출됨

    def on_file_selected(self, row):
        """리스트에서 파일 선택 시"""
        if 0 <= row < len(self.image_files):
            self.current_idx = row
            self.load_current_image()

    def prev_image(self):
        if self.image_files and self.current_idx > 0:
            self.list_widget.setCurrentRow(self.current_idx - 1)

    def next_image(self):
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.list_widget.setCurrentRow(self.current_idx + 1)

    def load_current_image(self):
        """현재 인덱스의 이미지를 로드하여 표시"""
        if not self.image_files:
            return
            
        img_path = str(self.image_files[self.current_idx])
        img = cv2.imread(img_path)
        
        if img is not None:
            self.image_label.set_image(img)
            
            # Depth 맵 로드
            frame_name = self.image_files[self.current_idx].stem
            depth_path = self.depth_dir / f"{frame_name}_depth.npy"
            
            if depth_path.exists():
                try:
                    depth_map = np.load(depth_path)
                    self.image_label.set_depth_map(depth_map)
                    print(f"[INFO] Depth 맵 로드 완료: {depth_path.name}")
                except Exception as e:
                    print(f"[ERROR] Depth 맵 로드 실패: {e}")
                    self.image_label.set_depth_map(None)
            else:
                print(f"[WARNING] Depth 맵 파일 없음: {depth_path}")
                self.image_label.set_depth_map(None)
            
            # Depth 맵 표시 상태가 활성화되어 있으면 화면 업데이트
            if self.image_label.show_depth:
                self.image_label.update_display()
                self.image_label.update()
            
            # 저장된 측정값 로드 (있으면 표시, 없으면 초기화)
            self.load_saved_measurement()
        else:
            self.info_label.setText(f"오류: 이미지를 불러올 수 없습니다 ({os.path.basename(img_path)})")

    def clear_measurement(self):
        """측정 결과 초기화"""
        self.info_label.setText("두 점을 클릭하여 거리를 측정하세요.")
        self.current_measurement = None
    
    def toggle_depth_view(self):
        """Depth 맵 오버레이 표시/숨김 토글"""
        self.image_label.toggle_depth_overlay()
        
        # 버튼 텍스트 업데이트
        if self.image_label.show_depth:
            self.btn_toggle_depth.setText("Depth 맵 숨김")
        else:
            self.btn_toggle_depth.setText("Depth 맵 표시")

    def load_saved_measurement(self):
        """저장된 측정값이 있으면 로드하여 표시"""
        if not self.current_video_dir or not self.image_files:
            self.clear_measurement()
            return
        
        frame_name = self.image_files[self.current_idx].stem
        csv_path = self.current_video_dir / "measurements" / "width_measurements.csv"
        
        if not csv_path.exists():
            self.clear_measurement()
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Frame'] == frame_name:
                        # 저장된 측정값 발견
                        distance = float(row['Distance_Meter'])
                        p1_depth = float(row['P1_Depth'])
                        p2_depth = float(row['P2_Depth'])
                        
                        # 좌표 정보 복원
                        p1_x = int(row['P1_X'])
                        p1_y = int(row['P1_Y'])
                        p2_x = int(row['P2_X'])
                        p2_y = int(row['P2_Y'])
                        
                        # ImageLabel에 점 표시
                        self.image_label.points = [(p1_x, p1_y), (p2_x, p2_y)]
                        self.image_label.update()  # 점과 선 다시 그리기
                        
                        # 화면에 거리 표시
                        if p1_depth <= 0 or p2_depth <= 0:
                            info_text = f"거리: {distance:.2f}m\n⚠ 유효하지 않은 Depth 값 포함 (측정 부정확)"
                        else:
                            info_text = f"거리: {distance:.2f}m"
                        
                        self.info_label.setText(info_text)
                        return
            
            # 저장된 값이 없으면 초기화
            self.clear_measurement()
            
        except Exception as e:
            print(f"[ERROR] 저장된 측정값 로드 실패: {e}")
            self.clear_measurement()



    def calculate_distance(self, p1, p2):
        """두 점 사이의 실제 3D 거리 계산"""
        # 현재 프레임 파일명에서 _depth.npy, _intrinsics.npy 파일명 유추
        frame_name = self.image_files[self.current_idx].stem # frame_XXXXXX
        
        depth_path = self.depth_dir / f"{frame_name}_depth.npy"
        intrinsics_path = self.depth_dir / f"{frame_name}_intrinsics.npy"
        
        if not depth_path.exists() or not intrinsics_path.exists():
            self.info_label.setText("오류: Depth 또는 Intrinsics 파일이 없습니다.")
            return
            
        try:
            # 데이터 로드
            depth_map = np.load(depth_path)
            intrinsics = np.load(intrinsics_path)
            
            # Depth Map 해상도
            dh, dw = depth_map.shape[:2]
            
            # 원본 이미지 해상도 (현재 로드된 이미지 기준)
            if self.image_label.original_image is None:
                return
            oh, ow = self.image_label.original_image.shape[:2]
            
            # 좌표 스케일링 (Image Space -> Depth Space)
            sx = dw / ow
            sy = dh / oh
            
            # 원본 좌표
            x1_img, y1_img = p1
            x2_img, y2_img = p2
            
            # Depth 좌표로 변환
            x1 = int(x1_img * sx)
            y1 = int(y1_img * sy)
            x2 = int(x2_img * sx)
            y2 = int(y2_img * sy)
            
            # 좌표 유효성 검사 (Clamping)
            x1 = max(0, min(x1, dw - 1))
            y1 = max(0, min(y1, dh - 1))
            x2 = max(0, min(x2, dw - 1))
            y2 = max(0, min(y2, dh - 1))
            
            # Intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            # 디버깅 로그
            print(f"\n[DEBUG] Calculating Distance...")
            print(f"  Frame: {frame_name}")
            print(f"  Depth Path: {depth_path}")
            print(f"  Depth Shape: {depth_map.shape}")
            print(f"  Intrinsics Shape: {intrinsics.shape}")
            print(f"  Image Res: ({ow}, {oh}) -> Depth Res: ({dw}, {dh})")
            print(f"  Scale Factor: x={sx:.3f}, y={sy:.3f}")
            print(f"  Clicked: p1({x1_img}, {y1_img}), p2({x2_img}, {y2_img})")
            print(f"  Mapped:  p1({x1}, {y1}), p2({x2}, {y2})")
            
            # 깊이값 가져오기 (보간 포함)
            def get_depth_with_interpolation(depth_map, y, x, window_size=5):
                """depth 값이 0이면 주변 픽셀 평균으로 보간"""
                depth = depth_map[y, x]
                
                if depth > 0:
                    return depth, False  # 유효한 값, 보간 안함
                
                # depth가 0이면 주변 픽셀에서 유효한 값 찾기
                half_window = window_size // 2
                h, w = depth_map.shape[:2]
                
                # 윈도우 범위 계산
                y_min = max(0, y - half_window)
                y_max = min(h, y + half_window + 1)
                x_min = max(0, x - half_window)
                x_max = min(w, x + half_window + 1)
                
                # 주변 영역 추출
                window = depth_map[y_min:y_max, x_min:x_max]
                valid_depths = window[window > 0]
                
                if len(valid_depths) > 0:
                    # 유효한 값들의 평균 사용
                    interpolated = float(np.mean(valid_depths))
                    print(f"    [보간] ({x}, {y}) depth=0 -> {interpolated:.2f}m (주변 {len(valid_depths)}개 픽셀 평균)")
                    return interpolated, True  # 보간된 값
                else:
                    # 주변에도 유효한 값이 없음
                    print(f"    [경고] ({x}, {y}) 주변에 유효한 depth 값 없음")
                    return 0.0, True  # 보간 실패
            
            z1, z1_interpolated = get_depth_with_interpolation(depth_map, y1, x1)
            z2, z2_interpolated = get_depth_with_interpolation(depth_map, y2, x2)
            
            print(f"  Depth Values: z1={z1}, z2={z2}")
            
            # 경고 메시지 표시
            warning_msg = ""
            if z1 <= 0 or z2 <= 0:
                warning_msg = " [경고: 유효하지 않은 Depth 값(0) 포함됨]"
                print("  [WARNING] Point has zero depth! Distance calculation will be incorrect.")

            # 3D 좌표 복원 (Back-projection) - Depth Space 좌표 사용
            X1 = (x1 - cx) * z1 / fx
            Y1 = (y1 - cy) * z1 / fy
            Z1 = z1
            
            X2 = (x2 - cx) * z2 / fx
            Y2 = (y2 - cy) * z2 / fy
            Z2 = z2
            
            # 유클리드 거리 계산
            dist = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
            
            print(f"  Calculated Distance: {dist:.4f} m")
            
            self.current_measurement = {
                'frame': frame_name,
                'p1_img': (x1_img, y1_img), # 원본 이미지 좌표 저장
                'p2_img': (x2_img, y2_img),
                'p1_depth_idx': (x1, y1),   # Depth 맵 인덱스
                'p2_depth_idx': (x2, y2),
                'p1_3d': (float(X1), float(Y1), float(Z1)),
                'p2_3d': (float(X2), float(Y2), float(Z2)),
                'distance': float(dist)
            }
            
            # 정보 표시 (간결한 포맷)
            if warning_msg:
                # 경고가 있는 경우
                info_text = f"거리: {dist:.2f}m\n⚠ 유효하지 않은 Depth 값 포함 (측정 부정확)"
            else:
                # 정상적인 경우
                info_text = f"거리: {dist:.2f}m"
            
            self.info_label.setText(info_text)
            
            # 자동 저장
            self.save_measurement()
            
        except Exception as e:
            print(f"Calculation Error: {e}")
            self.info_label.setText(f"계산 오류: {str(e)}")

    def save_measurement(self):
        """측정 결과를 파일로 저장 (자동 저장, 조용히 실행)"""
        if not self.current_measurement or not self.current_video_dir:
            return
            
        # 저장 폴더 생성: map_output/{video}/measurements
        save_dir = self.current_video_dir / "measurements"
        save_dir.mkdir(exist_ok=True)
        
        csv_path = save_dir / "width_measurements.csv"
        
        try:
            # 기존 데이터 읽기
            existing_data = []
            frame_name = self.current_measurement['frame']
            
            if csv_path.exists():
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_data = [row for row in reader if row['Frame'] != frame_name]
            
            # 새 데이터 추가
            m = self.current_measurement
            new_row = {
                'Frame': m['frame'],
                'P1_X': m['p1_img'][0],
                'P1_Y': m['p1_img'][1],
                'P2_X': m['p2_img'][0],
                'P2_Y': m['p2_img'][1],
                'P1_Depth': f"{m['p1_3d'][2]:.4f}",
                'P2_Depth': f"{m['p2_3d'][2]:.4f}",
                'Distance_Meter': f"{m['distance']:.4f}"
            }
            existing_data.append(new_row)
            
            # 파일에 쓰기
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Frame', 'P1_X', 'P1_Y', 'P2_X', 'P2_Y', 
                             'P1_Depth', 'P2_Depth', 'Distance_Meter']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_data)
            
            print(f"[INFO] 측정값 저장 완료: {frame_name} -> {m['distance']:.2f}m")
            
        except Exception as e:
            print(f"[ERROR] 파일 저장 중 오류: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
