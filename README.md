# 어류 크기측정 실시간 산출

Jetson 환경에서 HIKVISION IP 카메라의 RTSP 스트림을 입력받아 **어체 크기(Size Measurement)** 를 실시간으로 측정하고, 지정한 시간 동안의 평균 체장·체고를 cm 단위로 CSV에 누적 저장하는 프로그램,
동시에 bbox·keypoint와 측정 지점이 오버레이된 영상을 MP4로 저장 가능

YOLO Pose 모델로 어체의 keypoint를 검출하고, **keypoint 쌍 사이의 거리**를 치수로 사용. 렌즈 왜곡 보정(`calibration.npz`)과 ROI 기반 부분 추론을 적용해 측정 정확도와 처리 속도를 함께 확보

---

# 아키텍처
## ![AI 기반 어류 크기측정](docs/Size_Estimate_Architecture.png)

## 1. 주요 기능

- HIKVISION IP 카메라 RTSP 스트림 실시간 입력 처리 (파일 경로·웹캠 index도 지원)
- YOLO Pose 기반 어체 keypoint 검출 (Keypoint-based Morphometric Analysis)
- ROI 영역 크롭 후 부분 추론 (Region-of-Interest Cropped Inference)
- `calibration.npz` 기반 렌즈 왜곡 보정 (Lens Distortion Correction)
- keypoint 쌍 거리 기반 체장·체고 산출 및 픽셀 → cm 실척 환산
- 지정 시간(기본 120초) 동안의 전체 관측치 평균 및 표준편차 산출
- bbox·keypoint·측정 지점·ROI 경계가 오버레이된 MP4 영상 저장 (옵션)
- CSV 결과 누적 저장 (append)
- `jetson_robot_api` 서버로 시작/종료 flag 전송

---

## 2. 실행 환경

### 하드웨어

- NVIDIA Jetson 보드
- HIKVISION IP 카메라

### 소프트웨어

- Python 3.x
- OpenCV (FFMPEG 백엔드 포함 — RTSP 디코딩에 필요)
- NumPy
- ultralytics (YOLO Pose `.predict()` API 사용)
- requests

필요 패키지:

```bash
pip install -r requirements.txt
```

Jetson 환경에서는 OpenCV가 RTSP 디코딩을 지원하는지 확인 필요

```bash
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i ffmpeg
```

ultralytics/torch가 Jetson의 CUDA를 정상적으로 잡는지도 함께 확인 필요

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. 파일 구조 예시

```text
project/
├── main.py        
├── size_measurement.py               
├── best.pt                          
├── calibration.npz                 
├── README.md
└── results/
    ├── size_measurement.csv
    └── size_measurement_overlay.mp4
```

`main.py`와 `size_measurement.py`는 **반드시 같은 경로**에 함께 있어야 함

`best.pt`는 저장소에 포함되어 있지 않으므로, 대상 어종을 학습한 YOLO Pose 가중치 파일을 별도로 준비해서 배치해야 함

---

## 4. 설정값

실행 시점에 바꾸는 값은 CLI 인자로, 모델 구조에 종속된 값은 `size_measurement.py` 상단 상수로 분리되어 있음

### 4.1 CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--weights` | (필수) | YOLO Pose `.pt` 가중치 경로 |
| `--tank-id` | (필수) | 수조 ID. 로봇 flag 전송과 CSV 기록에 사용 (예: `tank-03`) |
| `--source` | — | RTSP URL 전체 / 파일 경로 / 웹캠 index. `--camera-ip`와 택일 |
| `--camera-ip` | — | HIKVISION 카메라 IP. RTSP URL을 자동 생성. `--source`와 택일 |
| `--camera-user` | `admin` | HIKVISION 계정 |
| `--camera-password` | — | HIKVISION 비밀번호. `--camera-ip` 사용 시 필수 |
| `--camera-channel` | `1` | 채널 번호 |
| `--camera-stream` | `main` | `main`(주 스트림, 카메라 최대 해상도) / `sub`(보조 스트림, 통상 640x360급) |
| `--roi` | 없음 | `x1,y1,x2,y2` 형식. 이 영역만 잘라서 추론. 생략 시 전체 프레임 |
| `--calib` | 없음 | `calibration.npz` 경로. 지정 시 keypoint 좌표에 왜곡 보정 적용 |
| `--scale-px` | `133` | 실척 환산 기준 픽셀 길이 |
| `--scale-cm` | `20` | `--scale-px`에 해당하는 실제 길이(cm) |
| `--duration` | `120` | 추론 동작 시간(초) |
| `--imgsz` | `1280` | YOLO 추론 입력 크기 |
| `--device` | `0` | Jetson GPU는 `0`, CPU만 쓰려면 `cpu` |
| `--csv` | `results/size_measurement.csv` | 결과를 append할 CSV 경로 |
| `--overlay-path` | 없음 | 지정 시 오버레이 영상을 이 경로에 저장 (`.mp4` / `.avi`) |
| `--overlay-fps` | 없음 | 오버레이 fps. 생략 시 소스 fps, 그마저 신뢰할 수 없으면 15fps |

### 4.2 코드 상수 (`size_measurement.py`)

```python
MODEL_TYPE = "size_measurement"
ROBOT_API_BASE = os.environ.get("ROBOT_API_BASE", "http://localhost:8100")

KEYPOINT_GROUP_A = (0, 1)   # "1~2번 keypoint" -> width
KEYPOINT_GROUP_B = (2, 3)   # "3~4번 keypoint" -> height

CONF_THRESHOLD = 0.25

# 카메라 고정 기준 값
DEFAULT_SCALE_PX = 133
DEFAULT_SCALE_CM = 20
```

| 설정값 | 설명 |
|---|---|
| `MODEL_TYPE` | `jetson_robot_api`의 ModelType과 동일해야 하는 식별자 |
| `ROBOT_API_BASE` | flag를 보낼 서버 주소. 환경변수 `ROBOT_API_BASE`로 재정의 가능 |
| `KEYPOINT_GROUP_A` | width(체장)를 재는 keypoint 쌍의 0-based 인덱스 |
| `KEYPOINT_GROUP_B` | height(체고)를 재는 keypoint 쌍의 0-based 인덱스 |
| `CONF_THRESHOLD` | 이 값 미만인 keypoint는 관측치에서 제외 |
| `DEFAULT_SCALE_PX` / `DEFAULT_SCALE_CM` | 실척 환산 기본 기준값 |

문서상 "1~2번 / 3~4번 keypoint"는 1-based 표기로 보고 0-based 배열의 `(0,1)`, `(2,3)`에 매핑되어 있음. 학습된 모델의 keypoint 순서가 다르면 이 두 상수만 수정하면 됨

실제 순서가 불확실하면 다음으로 먼저 확인

```python
print(res.keypoints.xy.shape)   # (개체 수, keypoint 수, 2)
```

---

## 5. 크기 산출 방식

### 5.1 ROI 크롭 후 추론 (Region-of-Interest Cropped Inference)

`--roi`가 지정되면 해당 영역만 잘라내어 YOLO에 입력

```text
Calibration을 통해 왜곡 보정이 어느정도 이루어 졌으나 완벽하게 왜곡이 보정된 것이 아니므로,
카메라가 촬영하고 있는 물리적 거리가 먼 외곽영역은 추론에서 제외하여 크기 정확도를 향상
```

전체 4K를 입력하면 3840 → 1280으로 **0.33배** 축소되지만, ROI만 입력하면 1500 → 1280으로 **0.85배**만 축소됨. 개체가 화면에서 차지하는 픽셀이 약 2.6배 늘어나 작은 개체 검출과 치수 정밀도에 유리하며, 연산량도 함께 감소

ROI는 첫 프레임에서 실제 스트림 해상도에 맞춰 자동 클램프되며, 요청 면적의 절반 이상이 잘려나가면 경고를 출력

### 5.2 렌즈 왜곡 보정 (Lens Distortion Correction)

`--calib`로 `calibration.npz`를 지정하면 검출된 keypoint 좌표에 `cv2.undistortPoints`를 적용

프레임 전체를 remap하지 않고 점 몇 개만 변환하므로 연산 비용이 거의 없음. 대신 오버레이 영상은 원본(왜곡된) 화면 그대로 남고, **CSV에 저장되는 좌표와 치수만 보정값**이 됨

npz 키 이름은 `mtx`/`camera_matrix`/`cameraMatrix`/`K`, `dist`/`dist_coeffs`/`distCoeffs`/`D` 순으로 자동 탐색

npz에 `image_size` 키가 있으면 첫 프레임 해상도에 맞춰 카메라 행렬 K를 자동 스케일. 캘리브레이션 해상도와 스트림 해상도가 다를 때(예: 4K로 캘리브레이션 후 `sub` 스트림 사용) 보정이 틀어지는 것을 방지

```python
np.savez("calibration.npz", mtx=K, dist=D, image_size=np.array([w, h]))
```

보정을 적용하지 않으면 렌즈 왜곡 때문에 **같은 크기의 개체라도 화면 가장자리에서 중앙과 다른 픽셀 길이로 측정**됨

### 5.3 keypoint 쌍 거리 기반 치수 (Keypoint-based Morphometric Analysis)

한 개체에서 검출된 keypoint 중 지정한 두 쌍의 유클리드 거리를 치수로 사용

```python
width  = |kp1 - kp2|    # KEYPOINT_GROUP_A
height = |kp3 - kp4|    # KEYPOINT_GROUP_B
```

거리는 **ROI 오프셋 복원과 왜곡 보정이 모두 끝난 좌표**에서 계산. 두 점 중 하나라도 `CONF_THRESHOLD` 미만이면 그 개체의 해당 치수는 관측치에서 제외

bbox의 width/height를 쓰지 않는 이유는, 어체가 기울어져 있을 때 bbox가 실제 체장보다 커지기 때문. keypoint 간 거리는 자세와 무관하게 실제 치수를 측정

### 5.4 실척 환산

```text
px_per_cm = scale_px / scale_cm = 100 / 10 = 10.0
길이(cm) = 길이(px) / px_per_cm
```

CSV에는 cm 값과 px 값, 그리고 환산 계수 `px_per_cm`을 함께 기록. 나중에 기준이 잘못되었다고 판단되면 **CSV만 가지고 재환산 가능**

### 5.5 관측치 집계

실행 기간 동안 검출된 **모든 프레임 · 모든 개체**의 치수를 프레임/개체 구분 없이 한 풀에 모아 평균과 표준편차를 산출

프레임 단위 평균이나 개체별 평균이 필요하면 `size_measurement.py`의 `SampleCollector` 클래스만 교체하면 됨

---

## 6. 출력 파일

### 6.1 CSV 로그

`--csv`로 지정한 경로에 **append** 방식으로 누적. 파일이 없으면 헤더와 함께 새로 생성

CSV 컬럼:

| 컬럼명 | 설명 |
|---|---|
| `tank_id` | 이번 실행의 대상 수조 ID |
| `start_time` | 추론 시작 시각 (KST, ISO 8601) |
| `end_time` | 추론 종료 시각 (KST, ISO 8601) |
| `roi` | 적용한 ROI 영역 (`x1\|y1\|x2\|y2`). 미사용 시 빈 칸 |
| `undistorted` | 왜곡 보정 적용 여부 (1 / 0) |
| `avg_width_cm` | **평균 체장(cm)** |
| `avg_height_cm` | **평균 체고(cm)** |
| `std_width_cm` | 체장 표준편차(cm) |
| `std_height_cm` | 체고 표준편차(cm) |
| `avg_width_px` | 평균 체장(px) — 재환산용 원본값 |
| `avg_height_px` | 평균 체고(px) — 재환산용 원본값 |
| `px_per_cm` | 적용한 실척 환산 계수 |
| `kp1_2_avg_x`, `kp1_2_avg_y` | kp1~2 중점의 평균 좌표 |
| `kp3_4_avg_x`, `kp3_4_avg_y` | kp3~4 중점의 평균 좌표 |
| `frame_count` | 처리한 총 프레임 수 |
| `kp1_2_sample_count` | kp1~2 관측치 개수 |
| `kp3_4_sample_count` | kp3~4 관측치 개수 |

`kp*_sample_count`는 신뢰도를 판단하는 지표. 값이 지나치게 작으면 검출이 거의 안 된 것이므로 평균값을 신뢰하기 어려움

### 6.2 MP4 영상 (옵션)

`--overlay-path`를 지정한 경우에만 저장. 화면에는 다음이 표시됨

- 빨간 사각형 — ROI 경계
- YOLO 기본 렌더링 — bbox, 클래스 라벨, keypoint, skeleton
- 노란 점 — kp1~2 중점 (`kp1-2 avg`)
- 자홍 점 — kp3~4 중점 (`kp3-4 avg`)
- 하늘색 선 — 두 중점을 잇는 선

ROI 밖은 원본 그대로 남아 탐지 영역과 비탐지 영역이 한눈에 구분됨

오버레이를 켜면 프레임마다 렌더링·인코딩 비용이 추가되므로, Jetson에서 실시간 처리가 빠듯하면 운영 시에는 끄고 검증 시에만 사용하는 것을 권장

---

## 7. 실행 방법

### 7.1 IP로 RTSP URL 자동 생성

```bash
python3 main.py \
    --weights best.pt \
    --tank-id tank-03 \
    --camera-ip 192.168.10.199 \
    --camera-user admin \
    --camera-password "$CAMERA_PW" \
    --roi 1500,1000,3000,2160 \
    --calib calibration.npz \
    --csv results/size_measurement.csv \
    --overlay-path results/size_measurement_overlay.mp4
```

비밀번호는 코드나 문서에 직접 적지 말고 환경변수로 넘기는 것을 권장

### 7.2 RTSP URL 직접 지정

```bash
python3 main.py \
    --weights best.pt \
    --tank-id tank-03 \
    --source "rtsp://admin:****@192.168.0.64:554/Streaming/Channels/101" \
    --csv results/size_measurement.csv
```

실행 후 다음과 같은 로그가 출력됨

```text
[INFO] RTSP URL: rtsp://admin:****@192.168.10.199:554/...
[START] size_measurement 추론 시작 (tank=tank-03, 120초)
[INFO] 실척 기준: 133 px = 20 cm (6.6500 px/cm)
[INFO] 실제 스트림 해상도: 3840x2160
[INFO] ROI (1500,1000)-(3000,2160) 1500x1160 -> imgsz 1280 (축소율 0.85x, 전체 프레임 대비 0.33x)
[INFO] 왜곡 보정 적용 (3840x2160)
  ... 30 프레임, 남은 시간 약 118초
  ...
[END] size_measurement 추론 종료 (tank=tank-03)
[RESULT] 평균 width  20.15 cm (±0.86, 134.0 px, n=428)
[RESULT] 평균 height 10.02 cm (±0.43, 66.6 px, n=428)
[CSV] 결과 저장: /home/jetson/results/size_measurement.csv
```

---

## 8. 종료 방법

`--duration`으로 지정한 시간(기본 120초)이 지나면 **자동으로 종료**되며, CSV 저장과 종료 flag 전송이 수행됨

중간에 멈춰야 하면 `Ctrl+C`를 사용. 이 경우 `finally` 블록이 실행되어 카메라 해제와 영상 파일 마무리는 이루어지지만, CSV 저장은 수행되지 않음

---

