"""Executavel do modulo independente de monitoramento visual."""

from __future__ import annotations

import argparse
import ctypes
import time

import cv2
import numpy as np

from modules.visual_attention_monitor import (
    AttentionStateMachine,
    CognitiveDistractionDetector,
    EventLogger,
    FaceLandmarkExtractor,
    FivePointCalibrator,
    MonitorConfig,
)


def screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        return 1920, 1080


def parse_aoi(value: str, width: int, height: int) -> tuple[float, float, float, float]:
    if value == "full":
        return 0.0, 0.0, float(width), float(height)
    try:
        left, top, right, bottom = (float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("AOI deve ser 'full' ou left,top,right,bottom") from error
    if max(left, top, right, bottom) <= 1.0:
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise argparse.ArgumentTypeError("AOI fora dos limites da tela")
    return left, top, right, bottom


def calibration_image(width: int, height: int, point: tuple[int, int], index: int) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.circle(image, point, 15, (20, 20, 220), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Calibracao {index + 1}/5 - fixe o olhar no ponto",
        (max(20, width // 2 - 300), 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return image


def run_calibration(cap, extractor, calibrator) -> None:
    window = "Calibracao do olhar"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    message = "Posicione o rosto e pressione ESPACO para iniciar"
    while True:
        while True:
            ok, frame = cap.read()
            if ok:
                observation = extractor.extract(frame, time.monotonic())
                ready = observation is not None and observation.detected
                preview = frame.copy()
                cv2.putText(preview, message, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview, "Rosto detectado" if ready else "Rosto nao detectado",
                            (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (40, 190, 40) if ready else (20, 40, 230), 2, cv2.LINE_AA)
                cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt
            if key == ord(" ") and ok and ready:
                break
        try:
            collect_calibration(cap, extractor, calibrator, window)
        except RuntimeError as error:
            print(f"Calibracao incompleta: {error}. Pressione ESPACO para tentar novamente.")
            calibrator.__init__(calibrator.screen_width, calibrator.screen_height)
            message = "Calibracao incompleta. ESPACO para tentar novamente"
            continue
        cv2.destroyWindow(window)
        return


def collect_calibration(cap, extractor, calibrator, window) -> None:
    started = time.monotonic()
    calibrator.start(started)
    while not calibrator.is_complete(time.monotonic()):
        ok, frame = cap.read()
        now = time.monotonic()
        if not ok:
            continue
        observation = extractor.extract(frame, now)
        calibrator.add(observation)
        index = calibrator.point_index(now)
        cv2.imshow(window, calibration_image(
            calibrator.screen_width, calibrator.screen_height, calibrator.points[index], index
        ))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            raise KeyboardInterrupt
    calibrator.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor visual de atencao/dispersao via webcam")
    parser.add_argument("--participant-id", required=True, help="Identificador pseudonimizado")
    parser.add_argument("--camera", type=int, default=0, help="Indice OpenCV da webcam")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--aoi", default="full", help="full ou left,top,right,bottom; aceita 0..1")
    parser.add_argument("--output", default="monitoring_logs")
    parser.add_argument("--no-preview", action="store_true", help="Nao exibe a janela da webcam")
    parser.add_argument("--cognitive-threshold", type=float, default=0.0004)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    display_width, display_height = screen_size()
    aoi = parse_aoi(args.aoi, display_width, display_height)
    config = MonitorConfig(cognitive_variance_threshold=args.cognitive_threshold)
    backend = cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a webcam de indice {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    extractor = FaceLandmarkExtractor()
    calibrator = FivePointCalibrator(display_width, display_height)
    session_started = time.monotonic()
    logger = EventLogger(args.participant_id, args.output, session_started)
    machine = AttentionStateMachine(config, logger)
    cognitive = CognitiveDistractionDetector(config)

    print("Iniciando calibracao de 15 segundos (ESC/Q cancela)...")
    try:
        run_calibration(cap, extractor, calibrator)
        print(f"Monitoramento iniciado. CSV: {logger.csv_path} | JSON: {logger.json_path}")
        while True:
            frame_started = time.monotonic()
            ok, frame = cap.read()
            now = time.monotonic()
            observation = extractor.extract(frame, now) if ok else None
            prediction = None
            variance = None
            if observation and observation.detected:
                prediction = calibrator.predict(observation, aoi, config)
                cognitive_flag, variance = cognitive.update(
                    now,
                    prediction.x / display_width,
                    prediction.y / display_height,
                    prediction.on_task and observation.ear >= config.ear_threshold,
                )
                machine.update(
                    now,
                    True,
                    not prediction.on_task,
                    observation.ear < config.ear_threshold,
                    cognitive_flag,
                )
            else:
                cognitive.update(now, 0.0, 0.0, False)
                machine.update(now, False, False, False, False)

            processing = time.monotonic() - frame_started
            if not args.no_preview and ok:
                color = (40, 190, 40) if machine.state.value == "EM_FOCO" else (20, 40, 230)
                cv2.putText(frame, machine.state.value, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                metrics = f"EAR: {observation.ear:.3f}" if observation and observation.ear is not None else "Face/iris: ausente"
                if prediction:
                    metrics += f" | gaze=({prediction.x:.0f},{prediction.y:.0f})"
                if variance is not None:
                    metrics += f" | V={variance:.6f}"
                cv2.putText(frame, metrics, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"latencia={processing:.3f}s", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imshow("Monitor de atencao - Q para encerrar", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        print("Monitoramento cancelado.")
    finally:
        machine.close(time.monotonic())
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
