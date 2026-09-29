#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from size_measurement import (
    DEFAULT_DURATION_SEC,
    DEFAULT_SCALE_CM,
    DEFAULT_SCALE_PX,
    RunConfig,
    build_hikvision_rtsp_url,
    parse_roi_arg,
    run,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="크기측정(Size Measurement) YOLO Pose 추론 + CSV 저장")

    p.add_argument("--weights", required=True, help="YOLO Pose .pt 가중치 경로")
    p.add_argument("--tank-id", required=True, help="수조 ID (예: tank-03)")
    p.add_argument("--csv", default="results/size_measurement.csv",
                   help="결과를 append할 CSV 경로")
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION_SEC,
                   help=f"추론 동작 시간(초), 기본 {DEFAULT_DURATION_SEC}")
    p.add_argument("--device", default="0", help="Jetson GPU: '0', CPU: 'cpu'")
    p.add_argument("--imgsz", type=int, default=1280, help="추론 입력 크기")

    # 영상 소스 — URL 직접 지정 또는 IP로 자동 생성 중 택일
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="RTSP URL 전체 / 파일 경로 / 웹캠 index")
    src.add_argument("--camera-ip", help="HIKVISION 카메라 IP (RTSP URL 자동 생성)")

    p.add_argument("--camera-user", default="admin", help="HIKVISION 계정")
    p.add_argument("--camera-password", help="HIKVISION 비밀번호")
    p.add_argument("--camera-channel", type=int, default=1, help="채널 번호")
    p.add_argument("--camera-stream", choices=["main", "sub"], default="main",
                   help="main(주 스트림, 4K급 — 계측용 기본값) / "
                        "sub(보조 스트림, 640x360급 — 지연은 적지만 계측엔 부적합)")

    p.add_argument("--roi", default=None, metavar="x1,y1,x2,y2",
                   help="이 영역만 추론 (예: 1500,1000,3000,2160). 생략 시 전체 프레임")
    p.add_argument("--calib", default=None, metavar="PATH",
                   help="calibration.npz 경로. 지정 시 keypoint 좌표에 왜곡 보정 적용")

    p.add_argument("--scale-px", type=float, default=DEFAULT_SCALE_PX,
                   help=f"실척 기준 픽셀 길이 (기본 {DEFAULT_SCALE_PX:g})")
    p.add_argument("--scale-cm", type=float, default=DEFAULT_SCALE_CM,
                   help=f"--scale-px 에 해당하는 실제 길이(cm) (기본 {DEFAULT_SCALE_CM:g})")

    p.add_argument("--overlay-path", default=None,
                   help="지정 시 bbox/keypoint + 평균점을 그린 영상을 저장 (.mp4/.avi)")
    p.add_argument("--overlay-fps", type=float, default=None,
                   help="오버레이 fps. 생략 시 소스 fps, 그마저 없으면 15fps")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.camera_ip:
        if not args.camera_password:
            raise SystemExit("--camera-ip 사용 시 --camera-password 가 필요합니다.")
        source = build_hikvision_rtsp_url(
            ip=args.camera_ip,
            username=args.camera_user,
            password=args.camera_password,
            channel=args.camera_channel,
            stream=args.camera_stream,
        )
        print(f"[INFO] RTSP URL: rtsp://{args.camera_user}:****@{args.camera_ip}:554/...")
    else:
        source = args.source

    result = run(RunConfig(
        weights_path=args.weights,
        source=source,
        tank_id=args.tank_id,
        csv_path=args.csv,
        duration_sec=args.duration,
        device=args.device,
        imgsz=args.imgsz,
        roi=parse_roi_arg(args.roi),
        calib_path=args.calib,
        scale_px=args.scale_px,
        scale_cm=args.scale_cm,
        overlay_path=args.overlay_path,
        overlay_fps=args.overlay_fps,
    ))
    print(f"[DONE] {result}")


if __name__ == "__main__":
    main()
