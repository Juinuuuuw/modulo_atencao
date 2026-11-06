# -*- coding: utf-8 -*-
"""Ponte NAOqi, compatível com Python 2.7 do NAO e SDK Python 2.7 externo.

Sem OpenCV/MediaPipe no robô. Captura RGB, envio JPEG com PIL/Pillow e análise no servidor.
Não envia comandos de movimento ao robô.
"""
from __future__ import print_function
import argparse
import getpass
import io
import json
import os
import threading
import time
import traceback
import uuid
try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen


def make_request(url, data, headers):
    # Python 2 promove o pacote HTTP a unicode quando o caminho veio do JSON.
    # A concatenação com RGB binário então tenta decodificá-lo como UTF-8.
    if not isinstance(url, str):
        url = url.encode("ascii")
    return Request(url, data=data, headers={str(k): str(v) for k, v in headers.items()})


def encode_image(image, quality=80, raw_rgb=False):
    raw = image[6]
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if raw_rgb:
        return raw, "application/x-rgb"
    from PIL import Image
    output = io.BytesIO()
    Image.frombytes("RGB", (image[0], image[1]), raw).save(output, format="JPEG", quality=quality)
    return output.getvalue(), "image/jpeg"


class FrameReader(threading.Thread):
    """Le o sensor continuamente; guarda apenas o quadro mais recente."""
    def __init__(self, video, motion, subscription):
        threading.Thread.__init__(self)
        self.daemon = True
        self.video, self.motion, self.subscription = video, motion, subscription
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                image = self.video.getImageRemote(self.subscription)
                captured = time.time()
                if not image or image[2] != 3:
                    raise ValueError("NAO sem imagem RGB valida")
                pose = self.motion.getAngles(["HeadYaw", "HeadPitch"], True)
                with self.lock:
                    self.latest = image, captured, pose
        except Exception as error:
            with self.lock:
                self.error = error

    def get_latest(self):
        with self.lock:
            if self.error is not None:
                raise self.error
            return self.latest

    def close(self):
        self.stop_event.set()
        self.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="Conector da câmera interna do NAO")
    parser.add_argument("--server", required=True)
    parser.add_argument("--robot", default="127.0.0.1", help="IP do NAO; local quando executado nele")
    parser.add_argument("--port", type=int, default=9559)
    parser.add_argument("--name", default="NAO")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Qualidade JPEG (60 a 95)")
    parser.add_argument("--raw-rgb", action="store_true", help="Enviar RGB sem compressao (mais trafego)")
    parser.add_argument("--key-file", help="Arquivo local com o codigo de acesso, para execucao sem terminal")
    parser.add_argument("--node-id-file", help="Arquivo de identidade para manter a selecao apos reiniciar o conector")
    args = parser.parse_args()
    if not 1 <= args.fps <= 15:
        parser.error("Use 1 a 15 FPS")
    if not 60 <= args.jpeg_quality <= 95:
        parser.error("Use qualidade JPEG entre 60 e 95")
    if not args.raw_rgb:
        try:
            from PIL import Image
        except ImportError:
            parser.error("Pillow/PIL ausente: instale no ambiente do conector ou use --raw-rgb")
    args.server = args.server.rstrip("/")
    if args.key_file:
        with open(args.key_file, "r") as stream:
            key = stream.read().strip()
        if not key:
            parser.error("Arquivo de acesso vazio")
    else:
        key = os.environ.get("MONITOR_ACCESS_KEY") or getpass.getpass("Codigo de acesso do servidor: ")
    from naoqi import ALProxy
    video = ALProxy("ALVideoDevice", args.robot, args.port)
    motion = ALProxy("ALMotion", args.robot, args.port)
    node_id = uuid.uuid4().hex
    if args.node_id_file:
        if os.path.exists(args.node_id_file):
            with open(args.node_id_file, "r") as stream:
                node_id = uuid.UUID(stream.read().strip()).hex
        else:
            with open(args.node_id_file, "w") as stream:
                stream.write(node_id)

    def api(path, payload=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
        values = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        values.update(headers or {})
        response = urlopen(make_request(args.server + path, data, values), timeout=5)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()

    def sync():
        estimates = []
        for _ in range(3):
            before = time.time()
            remote = api("/api/time")["monotonic"]
            after = time.time()
            estimates.append((after - before, remote - (before + after) / 2))
        rtt, offset = min(estimates)
        return offset, rtt / 2

    subscription, selected, sequence = None, None, 0
    reader, last_capture = None, None
    registered, last_sync, last_heartbeat = False, 0, 0
    offset, uncertainty, error_message = 0, 1, ""
    try:
        while True:
            started = time.time()
            try:
                if not registered:
                    api("/api/nodes/register", {"id": node_id, "name": args.name, "cameras": [
                        {"id": "0", "name": "NAO - camera superior", "kind": "nao"},
                        {"id": "1", "name": "NAO - camera inferior", "kind": "nao"}]})
                    registered = True
                    print("NAO conectado. Selecione a camera superior no painel.")
                if started - last_sync > 30:
                    offset, uncertainty = sync()
                    last_sync = started
                if started - last_heartbeat > 1:
                    desired = api("/api/nodes/" + node_id + "/heartbeat", {"error": error_message})["cameras"]
                    if len(desired) > 1:
                        raise ValueError("Selecione apenas uma camera interna por NAO")
                    camera = str(desired[0]) if desired else None
                    if camera != selected:
                        if reader:
                            reader.close()
                            reader = None
                        if subscription:
                            video.unsubscribe(subscription)
                            subscription = None
                        selected = None
                        if camera is not None:
                            # kVGA=2, kRGBColorSpace=11; índices superior=0/inferior=1.
                            subscription = video.subscribeCamera("attention_" + node_id, int(camera), 2, 11, args.fps)
                            selected = camera
                            reader = FrameReader(video, motion, subscription)
                            reader.start()
                            last_capture = None
                    last_heartbeat = started
                latest = reader.get_latest() if reader else None
                if latest and latest[1] != last_capture:
                    image, captured, pose = latest
                    last_capture = captured
                    sequence = max(sequence + 1, int(time.time() * 1000000))
                    encoded, content_type = encode_image(image, args.jpeg_quality, args.raw_rgb)
                    api("/api/nodes/" + node_id + "/frame/" + selected, raw=encoded, headers={
                        "Content-Type": content_type, "X-Width": str(image[0]), "X-Height": str(image[1]),
                        "X-Capture-Time": str(captured + offset), "X-Sync-Uncertainty": str(uncertainty),
                        "X-Sequence": str(sequence), "X-Head-Pose": json.dumps(pose),
                        "X-Device-Time": str(image[4] + image[5] / 1000000.0)})
                error_message = ""
            except Exception as error:
                error_message = str(error)
                print("Falha de conexao/captura:", error_message)
                traceback.print_exc()
                registered, last_sync = False, 0
                if reader:
                    reader.close()
                    reader = None
                if subscription:
                    try:
                        video.unsubscribe(subscription)
                    except Exception:
                        pass
                subscription, selected = None, None
                last_heartbeat = 0
                time.sleep(2)
            time.sleep(max(0.01, 1.0 / args.fps - (time.time() - started)))
    except KeyboardInterrupt:
        print("Encerrando conector NAO")
    finally:
        if reader:
            reader.close()
        if subscription:
            video.unsubscribe(subscription)


if __name__ == "__main__":
    main()
