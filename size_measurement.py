# -*- coding: utf-8 -*-
"""크기측정(Size Measurement) 추론 로직.

infer_size_measurement.py가 이 모듈을 import해서 실행한다.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

Point = Tuple[float, float]

# ── 설정 ────────────────────────────────────────────────────────────────

MODEL_TYPE = "size_measurement"  # jetson_robot_api ModelType과 동일한 값
ROBOT_API_BASE = os.environ.get("ROBOT_API_BASE", "http://localhost:8100")

DEFAULT_DURATION_SEC = 120

# 1-based "1~2번 / 3~4번 keypoint" -> 0-based 배열 인덱스.
# 모델의 keypoint 순서가 다르면 여기만 수정.
KEYPOINT_GROUP_A: Tuple[int, int] = (0, 1)
KEYPOINT_GROUP_B: Tuple[int, int] = (2, 3)

CONF_THRESHOLD = 0.25  # 이 값 미만인 keypoint는 관측치에서 제외

# 실척 환산 기준: 이 픽셀 길이가 이 실제 길이(cm)에 해당한다.
# 카메라-어체 거리가 고정되어 있을 때만 유효하다 (단안이라 거리가 변하면 통째로 틀어짐).
DEFAULT_SCALE_PX = 133.0
DEFAULT_SCALE_CM = 20.0

# OpenCV는 BGR 순서
COLOR_KP_A = (0, 255, 255)    # 노랑 — kp1~2 평균점
COLOR_KP_B = (255, 0, 255)    # 자홍 — kp3~4 평균점
COLOR_LINE = (255, 255, 0)    # 하늘 — 두 평균점 연결선
COLOR_ROI = (0, 0, 255)       # 빨강 — ROI 경계

FALLBACK_FPS = 15.0           # RTSP가 fps를 못 줄 때 사용할 값
KST = timezone(timedelta(hours=9))


@dataclass
class RunConfig:
    weights_path: str
    source: str
    tank_id: str
    csv_path: str = "results/size_measurement.csv"
    duration_sec: int = DEFAULT_DURATION_SEC
    device: str = "0"
    imgsz: int = 1280
    roi: Optional[Tuple[int, int, int, int]] = None   # None이면 전체 프레임
    calib_path: Optional[str] = None                  # calibration.npz 경로
    scale_px: float = DEFAULT_SCALE_PX                 # 실척 환산: scale_px 픽셀이
    scale_cm: float = DEFAULT_SCALE_CM                 #            scale_cm 센티미터
    overlay_path: Optional[str] = None
    overlay_fps: Optional[float] = None
    show_progress_every: int = 30

    @property
    def px_per_cm(self) -> Optional[float]:
        if not self.scale_px or not self.scale_cm:
            return None
        return self.scale_px / self.scale_cm


@dataclass
class RunResult:
    start_time: datetime
    end_time: datetime
    kp_a_avg: Optional[Point]
    kp_b_avg: Optional[Point]
    frame_count: int
    kp_a_sample_count: int
    kp_b_sample_count: int
    avg_width_px: Optional[float] = None      # kp1~2 거리 평균
    avg_height_px: Optional[float] = None     # kp3~4 거리 평균
    std_width_px: Optional[float] = None
    std_height_px: Optional[float] = None
    px_per_cm: Optional[float] = None

    def _to_cm(self, px: Optional[float]) -> Optional[float]:
        if px is None or not self.px_per_cm:
            return None
        return px / self.px_per_cm

    @property
    def avg_width_cm(self) -> Optional[float]:
        return self._to_cm(self.avg_width_px)

    @property
    def avg_height_cm(self) -> Optional[float]:
        return self._to_cm(self.avg_height_px)

    @property
    def std_width_cm(self) -> Optional[float]:
        return self._to_cm(self.std_width_px)

    @property
    def std_height_cm(self) -> Optional[float]:
        return self._to_cm(self.std_height_px)


# ── ROI ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Roi:
    """프레임 내 사각 영역. 좌표는 원본 프레임 기준."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_tuple(cls, box) -> Optional["Roi"]:
        return None if box is None else cls(*map(int, box))

    def clamp(self, width: int, height: int) -> "Roi":
        """프레임 범위를 벗어난 좌표를 잘라냄.

        요청 면적의 절반 이상이 잘려나가면 경고한다. 대개 --camera-stream sub
        (보조 스트림, 640x360급)로 받으면서 4K 기준 ROI 좌표를 준 경우다.
        """
        x1 = max(0, min(self.x1, width - 1))
        y1 = max(0, min(self.y1, height - 1))
        out = Roi(x1, y1,
                  max(x1 + 1, min(self.x2, width)),
                  max(y1 + 1, min(self.y2, height)))

        want = (self.x2 - self.x1) * (self.y2 - self.y1)
        got = (out.x2 - out.x1) * (out.y2 - out.y1)
        if want > 0 and got < want * 0.5:
            print(f"[WARN] ROI가 프레임({width}x{height})을 크게 벗어나 "
                  f"{self} -> {out} 로 잘렸습니다. "
                  f"4K 기준 좌표라면 --camera-stream main 을 쓰세요.")
        return out

    @property
    def offset(self) -> np.ndarray:
        return np.array([self.x1, self.y1], dtype=np.float32)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1:self.y2, self.x1:self.x2]

    def paste(self, frame: np.ndarray, patch: np.ndarray) -> None:
        frame[self.y1:self.y2, self.x1:self.x2] = patch

    def __str__(self) -> str:
        return (f"({self.x1},{self.y1})-({self.x2},{self.y2}) "
                f"{self.x2 - self.x1}x{self.y2 - self.y1}")


def parse_roi_arg(text: Optional[str]):
    """"x1,y1,x2,y2" CLI 인자를 튜플로 변환."""
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--roi 형식이 잘못됨: {text!r} (예: 1500,1000,3000,2160)")
    return tuple(int(p) for p in parts)


# ── 렌즈 왜곡 보정 ──────────────────────────────────────────────────────

_CALIB_K_KEYS = ("mtx", "camera_matrix", "cameraMatrix", "K")
_CALIB_D_KEYS = ("dist", "dist_coeffs", "distCoeffs", "D")


class Undistorter:
    """calibration.npz로 keypoint 좌표만 보정한다.

    프레임 전체를 remap하지 않고 검출된 점 몇 개만 변환하므로 비용이 거의 없다.
    대신 오버레이 영상은 원본(왜곡된) 화면 그대로 남고, CSV에 저장되는 좌표만
    보정된 값이 된다.

    주의: npz는 특정 해상도에서 촬영한 체커보드로 만들어진다. 스트림 해상도가
    그와 다르면 K를 비율만큼 스케일해야 하며, 이 클래스는 첫 프레임에서 그
    스케일링을 자동 수행한다(--camera-stream sub 처럼 해상도가 내려가는 경우).
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.K0 = self._pick(data, _CALIB_K_KEYS, npz_path)
        self.D = self._pick(data, _CALIB_D_KEYS, npz_path)
        # npz에 캘리브레이션 당시 해상도가 있으면 스케일 보정에 쓴다
        self.calib_size = None
        for key in ("image_size", "img_size", "shape"):
            if key in data:
                self.calib_size = tuple(int(v) for v in np.ravel(data[key])[:2])
                break
        self.K = self.K0
        self._fitted = False

    @staticmethod
    def _pick(data, names, path):
        for n in names:
            if n in data:
                return np.asarray(data[n], dtype=np.float64)
        raise KeyError(
            f"{path}에 {names} 중 어느 키도 없습니다. "
            f"실제 키: {list(data.keys())}"
        )

    def fit_to_frame(self, width: int, height: int) -> None:
        """첫 프레임 해상도에 맞춰 K를 스케일 (한 번만 수행)."""
        if self._fitted:
            return
        self._fitted = True
        if not self.calib_size:
            print("[INFO] 왜곡 보정 적용 (npz에 캘리브레이션 해상도 정보 없음 — "
                  "스트림 해상도가 캘리브레이션과 같다고 가정)")
            return
        cw, ch = self.calib_size
        if (cw, ch) == (width, height):
            print(f"[INFO] 왜곡 보정 적용 ({width}x{height})")
            return
        sx, sy = width / cw, height / ch
        self.K = self.K0.copy()
        self.K[0, 0] *= sx; self.K[0, 2] *= sx
        self.K[1, 1] *= sy; self.K[1, 2] *= sy
        print(f"[INFO] 왜곡 보정 적용 — K를 {cw}x{ch} -> {width}x{height} 로 스케일")

    def __call__(self, pt: Optional[Point]) -> Optional[Point]:
        """원본 프레임 기준 한 점을 보정된 좌표로 변환."""
        if pt is None:
            return None
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
        dst = cv2.undistortPoints(src, self.K, self.D, P=self.K)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])


# ── 로봇 flag ───────────────────────────────────────────────────────────


def _notify(endpoint: str, tank_id: str) -> None:
    try:
        requests.post(
            f"{ROBOT_API_BASE}/api/v1/model-runs/{endpoint}",
            json={"model_type": MODEL_TYPE, "tank_id": tank_id},
            timeout=3,
        )
    except requests.RequestException as exc:
        # flag 전송 실패는 추론을 막지 않음
        print(f"[WARN] {endpoint} flag 전송 실패 (계속 진행): {exc}")


# ── 카메라 ──────────────────────────────────────────────────────────────


def build_hikvision_rtsp_url(ip: str, username: str, password: str,
                             channel: int = 1, stream: str = "main",
                             port: int = 554) -> str:
    """HIKVISION 표준 RTSP 경로로 URL 생성.

    stream="main" -> 01 (주 스트림, 카메라 최대 해상도)
    stream="sub"  -> 02 (보조 스트림, 통상 640x360급)

    계측이 목적이면 반드시 main을 쓸 것. sub로 받으면 4K 기준으로 잡은 ROI
    좌표가 프레임을 벗어나고, 개체당 픽셀 수가 부족해 측정 오차가 커진다.

    기종·펌웨어에 따라 경로가 다를 수 있음. 열리지 않으면 ISAPI
    (http://<ip>/ISAPI/Streaming/channels)에서 실제 경로를 확인해
    --source 로 직접 지정할 것.
    """
    stream_no = "01" if stream == "main" else "02"
    return (f"rtsp://{username}:{password}@{ip}:{port}"
            f"/Streaming/Channels/{channel}{stream_no}")


def open_video_source(source: str) -> cv2.VideoCapture:
    # UDP는 패킷 유실로 디코딩이 깨지기 쉬워 TCP 전송을 강제
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(
            f"영상 스트림을 열 수 없습니다: {source}\n"
            "  - RTSP라면 IP/포트/계정/채널/스트림 번호를 확인하세요.\n"
            "  - 카메라와 Jetson이 같은 네트워크(554 포트 접근 가능)인지 확인하세요."
        )
    return cap


# ── 집계 ────────────────────────────────────────────────────────────────


def valid_points(kp_xy: np.ndarray, kp_conf: Optional[np.ndarray],
                 idx_pair: Sequence[int],
                 conf_threshold: float) -> List[Point]:
    """idx_pair 중 신뢰도를 통과한 점들만 (x, y) 리스트로."""
    return [(float(kp_xy[i][0]), float(kp_xy[i][1])) for i in idx_pair
            if i < len(kp_xy)
            and (kp_conf is None or float(kp_conf[i]) >= conf_threshold)]


def mid_point(pts: Sequence[Point]) -> Optional[Point]:
    """점들의 평균 좌표. 비었으면 None."""
    if not pts:
        return None
    arr = np.asarray(pts, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def pair_distance(pts: Sequence[Point]) -> Optional[float]:
    """두 점 사이 거리(px). 점이 2개가 아니면 None."""
    if len(pts) != 2:
        return None
    (x1, y1), (x2, y2) = pts
    return float(np.hypot(x2 - x1, y2 - y1))


def avg_point(kp_xy: np.ndarray, kp_conf: Optional[np.ndarray],
              idx_pair: Sequence[int], conf_threshold: float) -> Optional[Point]:
    """하위 호환용 — idx_pair 두 점의 평균 좌표."""
    return mid_point(valid_points(kp_xy, kp_conf, idx_pair, conf_threshold))


@dataclass
class SampleCollector:
    """프레임/개체 구분 없이 모든 관측치를 한 풀에 모아 평균.

    kp_a / kp_b 는 각 keypoint 쌍의 중점, width / height 는 쌍 사이의 거리(px).
    프레임 단위나 개체 단위 평균이 필요하면 이 클래스만 교체하면 된다.
    """

    kp_a: List[Point] = field(default_factory=list)
    kp_b: List[Point] = field(default_factory=list)
    width: List[float] = field(default_factory=list)    # kp1~2 거리
    height: List[float] = field(default_factory=list)   # kp3~4 거리

    def add(self, pt_a: Optional[Point], pt_b: Optional[Point],
            w: Optional[float] = None, h: Optional[float] = None) -> None:
        if pt_a is not None:
            self.kp_a.append(pt_a)
        if pt_b is not None:
            self.kp_b.append(pt_b)
        if w is not None:
            self.width.append(w)
        if h is not None:
            self.height.append(h)

    def averages(self):
        return (tuple(np.mean(self.kp_a, axis=0)) if self.kp_a else None,
                tuple(np.mean(self.kp_b, axis=0)) if self.kp_b else None)

    @staticmethod
    def _mean_std(values: List[float]):
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    def width_stats(self):
        return self._mean_std(self.width)

    def height_stats(self):
        return self._mean_std(self.height)


# ── 오버레이 ────────────────────────────────────────────────────────────


def _resolve_fps(cap: cv2.VideoCapture, overlay_fps: Optional[float]) -> float:
    if overlay_fps and overlay_fps > 0:
        return overlay_fps
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps and 0 < src_fps < 240:   # RTSP가 이상한 값을 주는 경우 방어
        return float(src_fps)
    print(f"[WARN] 소스 fps를 신뢰할 수 없어({src_fps}) {FALLBACK_FPS}로 고정합니다.")
    return FALLBACK_FPS


def _open_writer(path_str: str, shape, fps: float) -> cv2.VideoWriter:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if path.suffix.lower() == ".avi"
                                      else "mp4v"))
    h, w = shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"오버레이 영상 파일을 열 수 없습니다: {path_str}")
    return writer


def _draw_markers(frame: np.ndarray, pt_a: Optional[Point],
                  pt_b: Optional[Point], radius: int = 6,
                  thickness: int = 2) -> None:
    """CSV에 저장되는 kp1~2 / kp3~4 평균점과 연결선을 그림."""
    to_int = lambda p: (int(round(p[0])), int(round(p[1])))
    a = to_int(pt_a) if pt_a else None
    b = to_int(pt_b) if pt_b else None

    for pt, color, label in ((a, COLOR_KP_A, "kp1-2 avg"),
                             (b, COLOR_KP_B, "kp3-4 avg")):
        if pt is None:
            continue
        cv2.circle(frame, pt, radius, color, -1)
        cv2.putText(frame, label, (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if a and b:
        cv2.line(frame, a, b, COLOR_LINE, thickness)


def _draw_roi(frame: np.ndarray, roi: Roi, thickness: int = 3) -> None:
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), COLOR_ROI, thickness)
    cv2.putText(frame, "ROI", (roi.x1 + 10, roi.y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_ROI, thickness, cv2.LINE_AA)


# ── CSV ─────────────────────────────────────────────────────────────────

CSV_HEADER = [
    "tank_id", "start_time", "end_time", "roi", "undistorted",
    "avg_width_cm", "avg_height_cm", "std_width_cm", "std_height_cm",
    "avg_width_px", "avg_height_px", "px_per_cm",
    "kp1_2_avg_x", "kp1_2_avg_y", "kp3_4_avg_x", "kp3_4_avg_y",
    "frame_count", "kp1_2_sample_count", "kp3_4_sample_count",
]


def _r(value: Optional[float], nd: int = 2):
    """None은 빈 칸으로, 숫자는 소수점 nd자리로."""
    return "" if value is None else round(value, nd)


def save_result(csv_path: str, tank_id: str, result: RunResult,
                roi=None, undistorted: bool = False) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(CSV_HEADER)
        w.writerow([
            tank_id,
            result.start_time.isoformat(),
            result.end_time.isoformat(),
            "" if roi is None else "|".join(map(str, roi)),
            int(undistorted),
            _r(result.avg_width_cm), _r(result.avg_height_cm),
            _r(result.std_width_cm), _r(result.std_height_cm),
            _r(result.avg_width_px, 1), _r(result.avg_height_px, 1),
            _r(result.px_per_cm, 4),
            *(result.kp_a_avg or ("", "")),
            *(result.kp_b_avg or ("", "")),
            result.frame_count,
            result.kp_a_sample_count,
            result.kp_b_sample_count,
        ])
    print(f"[CSV] 결과 저장: {path.resolve()}")


# ── 추론 루프 ───────────────────────────────────────────────────────────


def _process_frame(model: YOLO, frame: np.ndarray, cfg: RunConfig,
                   roi: Optional[Roi], collector: SampleCollector,
                   render: bool,
                   undistort: Optional["Undistorter"] = None
                   ) -> Optional[np.ndarray]:
    """한 프레임 추론 + 관측치 수집. render=True면 오버레이 프레임을 반환.

    ROI가 설정되면 잘라낸 영역만 추론한다. 입력이 작아져 개체가 상대적으로 크게
    들어가므로 작은 개체 검출에 유리하고 연산량도 준다. 수집하는 좌표는 ROI
    오프셋을 더해 원본 프레임 기준으로 맞춘 뒤, undistort가 주어지면 렌즈 왜곡까지
    보정해서 저장한다. 오버레이에 그리는 마커는 보정 전 좌표라 화면과 일치한다.
    """
    target = roi.crop(frame) if roi else frame
    results = model.predict(target, imgsz=cfg.imgsz, device=cfg.device,
                            verbose=False)

    lw = max(2, round(min(target.shape[:2]) / 250)) if render else 0
    annotated = None

    for r in results:
        patch = r.plot(line_width=lw, kpt_radius=lw + 2) if render else None

        if r.keypoints is not None:
            xy_all = r.keypoints.xy.cpu().numpy()      # (개체, keypoint, 2)
            conf_all = (r.keypoints.conf.cpu().numpy()
                        if r.keypoints.conf is not None else None)

            ox, oy = roi.offset if roi else (0.0, 0.0)

            def to_measure_space(pts):
                """crop 좌표 -> 원본 좌표 -> (옵션) 왜곡 보정."""
                out = [(x + ox, y + oy) for x, y in pts]
                return [undistort(p) for p in out] if undistort else out

            for i in range(xy_all.shape[0]):
                kp_conf = conf_all[i] if conf_all is not None else None
                raw_a = valid_points(xy_all[i], kp_conf, KEYPOINT_GROUP_A,
                                     CONF_THRESHOLD)
                raw_b = valid_points(xy_all[i], kp_conf, KEYPOINT_GROUP_B,
                                     CONF_THRESHOLD)

                # 마커는 crop 좌표계에 그대로 그림 (화면과 일치해야 하므로 보정 전)
                if patch is not None:
                    _draw_markers(patch, mid_point(raw_a), mid_point(raw_b))

                # 치수는 보정된 좌표에서 재야 렌즈 왜곡 오차가 빠진다
                pts_a, pts_b = to_measure_space(raw_a), to_measure_space(raw_b)
                collector.add(
                    mid_point(pts_a), mid_point(pts_b),
                    pair_distance(pts_a),   # width  = kp1~2 거리
                    pair_distance(pts_b),   # height = kp3~4 거리
                )

        if render:
            if roi:
                annotated = frame.copy()
                roi.paste(annotated, patch)
                _draw_roi(annotated, roi)
            else:
                annotated = patch

    return annotated


def run(cfg: RunConfig) -> RunResult:
    """cfg.duration_sec 동안 스트림을 읽어 추론을 반복하고 CSV에 저장."""
    model = YOLO(cfg.weights_path)
    cap = open_video_source(cfg.source)

    render = cfg.overlay_path is not None
    fps = _resolve_fps(cap, cfg.overlay_fps) if render else 0.0
    if render:
        print(f"[INFO] 오버레이 저장: {cfg.overlay_path} ({fps:.1f} fps)")

    roi = Roi.from_tuple(cfg.roi)
    undistort = Undistorter(cfg.calib_path) if cfg.calib_path else None
    writer: Optional[cv2.VideoWriter] = None
    collector = SampleCollector()
    frame_count = 0

    start_time = datetime.now(KST)
    _notify("start", cfg.tank_id)
    print(f"[START] {MODEL_TYPE} 추론 시작 (tank={cfg.tank_id}, {cfg.duration_sec}초)")
    if cfg.px_per_cm:
        print(f"[INFO] 실척 기준: {cfg.scale_px:g} px = {cfg.scale_cm:g} cm "
              f"({cfg.px_per_cm:.4f} px/cm)")

    try:
        deadline = time.monotonic() + cfg.duration_sec
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 프레임 획득 실패, 재시도")
                time.sleep(0.2)
                continue

            # ROI와 K는 첫 프레임에서 실제 해상도에 맞춰 한 번만 보정
            if frame_count == 0:
                h, w = frame.shape[:2]
                print(f"[INFO] 실제 스트림 해상도: {w}x{h}")
                if roi is not None:
                    roi = roi.clamp(w, h)
                    cw, ch = roi.x2 - roi.x1, roi.y2 - roi.y1
                    # ultralytics는 긴 변이 imgsz가 되도록 letterbox 축소한다
                    scale = cfg.imgsz / max(cw, ch)
                    print(f"[INFO] ROI {roi} -> imgsz {cfg.imgsz} "
                          f"(축소율 {scale:.2f}x, 전체 프레임 대비 "
                          f"{(cfg.imgsz / max(w, h)):.2f}x)")
                if undistort is not None:
                    undistort.fit_to_frame(w, h)

            frame_count += 1
            annotated = _process_frame(model, frame, cfg, roi, collector,
                                       render, undistort)

            if render and annotated is not None:
                if writer is None:
                    # 첫 프레임에서 writer를 열어야 영상이 처음부터 끊기지 않음
                    writer = _open_writer(cfg.overlay_path, annotated.shape, fps)
                writer.write(annotated)

            if cfg.show_progress_every and frame_count % cfg.show_progress_every == 0:
                print(f"  ... {frame_count} 프레임, "
                      f"남은 시간 약 {max(0, int(deadline - time.monotonic()))}초")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[OVERLAY] 저장 완료: {Path(cfg.overlay_path).resolve()}")
        end_time = datetime.now(KST)
        _notify("complete", cfg.tank_id)
        print(f"[END] {MODEL_TYPE} 추론 종료 (tank={cfg.tank_id})")

    kp_a_avg, kp_b_avg = collector.averages()
    avg_w, std_w = collector.width_stats()
    avg_h, std_h = collector.height_stats()

    result = RunResult(
        start_time=start_time,
        end_time=end_time,
        kp_a_avg=kp_a_avg,
        kp_b_avg=kp_b_avg,
        frame_count=frame_count,
        kp_a_sample_count=len(collector.kp_a),
        kp_b_sample_count=len(collector.kp_b),
        avg_width_px=avg_w,
        avg_height_px=avg_h,
        std_width_px=std_w,
        std_height_px=std_h,
        px_per_cm=cfg.px_per_cm,
    )

    if result.avg_width_cm is not None:
        print(f"[RESULT] 평균 width  {result.avg_width_cm:.2f} cm "
              f"(±{result.std_width_cm:.2f}, {avg_w:.1f} px, n={len(collector.width)})")
    if result.avg_height_cm is not None:
        print(f"[RESULT] 평균 height {result.avg_height_cm:.2f} cm "
              f"(±{result.std_height_cm:.2f}, {avg_h:.1f} px, n={len(collector.height)})")

    save_result(cfg.csv_path, cfg.tank_id, result, cfg.roi,
                undistorted=undistort is not None)
    return result
