"""Downloads da validação humana (imagens por sessão e de todas) e CSV de todos os participantes."""
import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import create_app, stream_zip  # noqa: E402


def make_session(output, session_id, participant, started, status="complete", frames=2):
    path = Path(output) / session_id
    path.mkdir(parents=True)
    (path / "session.json").write_text(json.dumps({
        "id": session_id, "participant": participant, "started_utc": started, "duration": 1.0,
        "status": status}), encoding="utf-8")
    (path / "annotations.json").write_text("{}", encoding="utf-8")
    lines = []
    for role in ("notebook", "validation", "nao"):
        (path / role).mkdir()
        for i in range(frames):
            name = f"{i:012d}.jpg"
            (path / role / name).write_bytes(f"{session_id}-{role}-{i}".encode())
            lines.append(json.dumps({"role": role, "t": i * 0.1, "file": f"{role}/{name}", "sequence": i,
                                     "received_t": i * 0.1 + 0.01, "sync_uncertainty_s": 0.002}))
    (path / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prediction = {"t": 0.2, "target": "NAO", "score": 0.9, "reason": "teste", "cameras": {}}
    (path / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    return path


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        out = self.directory.name
        make_session(out, "a" * 32, "P002", "2026-11-12T13:00:00+00:00")
        make_session(out, "b" * 32, "P009", "2026-11-12T13:10:00+00:00")
        make_session(out, "c" * 32, "P009", "2026-11-12T13:20:00+00:00")
        self.app = create_app(out, "test-access", start_worker=False)
        # Depois de subir: ao iniciar, o servidor marca como interrompida uma sessão "recording" antiga.
        make_session(out, "d" * 32, "P020", "2026-11-12T14:00:00+00:00", status="recording")
        self.client = self.app.test_client()
        self.headers = {"Authorization": "Bearer test-access"}

    def tearDown(self):
        self.app.extensions["experiment"].close()
        self.directory.cleanup()

    def get_zip(self, url):
        response = self.client.get(url, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.data[:200])
        self.assertEqual(response.mimetype, "application/zip")
        archive = zipfile.ZipFile(io.BytesIO(response.get_data()))
        self.assertIsNone(archive.testzip())
        return response, archive

    def test_session_images_zip(self):
        response, archive = self.get_zip(f"/api/export/sessions/{'a' * 32}/images.zip")
        self.assertIn("P002_", response.headers["Content-Disposition"])
        names = archive.namelist()
        prefix = names[0].split("/")[0]
        self.assertTrue(prefix.startswith("P002_"))
        for folder in ("Notebook", "Apresentacao_TV", "NAO"):
            self.assertIn(f"{prefix}/{folder}/000000000001.jpg", names)
        self.assertEqual(archive.read(f"{prefix}/Apresentacao_TV/000000000000.jpg"), f"{'a' * 32}-validation-0".encode())
        rows = list(csv.DictReader(io.StringIO(archive.read(f"{prefix}/quadros.csv").decode("utf-8-sig"))))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["camera"] for r in rows}, {"Notebook", "Apresentacao_TV", "NAO"})
        self.assertIn(rows[0]["arquivo"], [n[len(prefix) + 1:] for n in names])

    def test_recording_session_is_refused(self):
        response = self.client.get(f"/api/export/sessions/{'d' * 32}/images.zip", headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_all_images_zip_by_participant(self):
        _, archive = self.get_zip("/api/export/images.zip")
        names = archive.namelist()
        # Uma sessão: direto na pasta do participante.
        self.assertIn("P002/Notebook/000000000000.jpg", names)
        self.assertIn("P002/quadros.csv", names)
        # Duas sessões: uma subpasta por sessão.
        p009 = sorted({n.split("/")[1] for n in names if n.startswith("P009/")})
        self.assertEqual(len(p009), 2)
        self.assertTrue(all(name.endswith(("_bbbbbbbb", "_cccccccc")) for name in p009))
        self.assertIn(f"P009/{p009[0]}/NAO/000000000001.jpg", names)
        # Sessão ainda gravando fica de fora.
        self.assertFalse(any(n.startswith("P020") for n in names))

    def test_all_predictions_csv(self):
        response = self.client.get("/api/export/predictions.csv", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        text = response.get_data().decode("utf-8")
        self.assertTrue(text.startswith("﻿"))
        rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
        self.assertEqual([r["participant"] for r in rows], ["P002", "P009", "P009"])
        self.assertEqual(rows[0]["target"], "NAO")
        self.assertEqual(rows[0]["task"], "DENTRO")
        self.assertIn("session_id", rows[0])

    def test_downloads_need_access_code(self):
        for url in ("/api/export/images.zip", "/api/export/predictions.csv",
                    f"/api/export/sessions/{'a' * 32}/images.zip"):
            self.assertEqual(self.client.get(url).status_code, 401)


class StreamZipTests(unittest.TestCase):
    def test_zip64_many_entries_streamed(self):
        """Todas as sessões passam de 65.535 imagens: o ZIP enviado aos poucos precisa de ZIP64."""
        count = 66000
        chunks = list(stream_zip(((f"x/{i}.jpg", b"jpg") for i in range(count)), chunk=1 << 16))
        self.assertGreater(len(chunks), 2)   # saiu em partes, não de uma vez
        archive = zipfile.ZipFile(io.BytesIO(b"".join(chunks)))
        self.assertEqual(len(archive.namelist()), count)
        self.assertEqual(archive.read("x/65999.jpg"), b"jpg")
        self.assertTrue(all(info.flag_bits & 0x08 == 0 for info in archive.infolist()[:5]))   # sem descritor


if __name__ == "__main__":
    unittest.main()
