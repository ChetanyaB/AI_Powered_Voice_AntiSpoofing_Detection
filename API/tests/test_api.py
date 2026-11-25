"""Test cases for the voice anti-spoofing service (run from API/: pytest -v)."""
import glob
import os
import sys

import numpy as np
import pytest
import soundfile as sf

API = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API)
os.chdir(API)

from fastapi.testclient import TestClient  # noqa: E402

from app.app import app  # noqa: E402
from app.src.deepfake import infa_deepfake, infa_deepfake_with_scores  # noqa: E402
from streamlit_app import extract_audio_metadata  # noqa: E402

client = TestClient(app)
REAL = sorted(glob.glob(os.path.join(API, "samples", "real", "*")))
SYNTH = sorted(glob.glob(os.path.join(API, "samples", "synthetic", "*")))


def test_tc01_health_endpoint():
    assert client.get("/").json() == {"Fast API": "API is working"}


@pytest.mark.parametrize("path", REAL[:1] + SYNTH[:1], ids=lambda p: os.path.basename(p))
def test_tc02_api_returns_label_and_scores(path):
    with open(path, "rb") as f:
        r = client.post("/depfake", files={"audio_file": (os.path.basename(path), f, "audio/wav")})
    body = r.json()
    assert r.status_code == 200 and body["status"] == 1
    assert body["Message"] in {"FAKE", "REAL"}
    assert abs(body["scores"]["FAKE"] + body["scores"]["REAL"] - 1) < 1e-3


def test_tc03_corrupt_file_gives_readable_error():
    r = client.post("/depfake", files={"audio_file": ("bad.wav", b"not audio", "audio/wav")})
    body = r.json()
    assert body["status"] == 0 and "Could not read audio" in body["Message"]   # was {} before the fix


def test_tc04_supported_formats_wav_and_ogg():
    for path in [REAL[0], SYNTH[0]]:          # .ogg and .wav
        status, label, _ = infa_deepfake_with_scores(path)
        assert status == 1 and label in {"FAKE", "REAL"}


def test_tc05_stereo_44k_input_handled(tmp_path):
    t = np.linspace(0, 2, 88200, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1) * 0.3
    p = tmp_path / "stereo.wav"
    sf.write(p, stereo, 44100)
    assert infa_deepfake(str(p))[0] == 1
    meta = extract_audio_metadata(str(p))
    assert meta["sr"] == 16000 and abs(meta["duration"] - 2.0) < 0.05


def test_tc06_very_short_clip():
    p = os.path.join(API, "samples", "_short.wav")
    sf.write(p, np.random.randn(8000).astype("float32") * 0.01, 16000)   # 0.5 s
    try:
        status, label, _ = infa_deepfake_with_scores(p)
        assert status == 1 and label in {"FAKE", "REAL"}
    finally:
        os.remove(p)


def test_tc07_prediction_is_deterministic():
    a = infa_deepfake_with_scores(REAL[0])[2]
    b = infa_deepfake_with_scores(REAL[0])[2]
    assert a == b


def test_tc08_metadata_pitch_in_human_range():
    meta = extract_audio_metadata(REAL[0])
    assert 60 < meta["avg_pitch"] < 300 and meta["avg_energy"] > 0


def test_tc09_recorded_audio_uses_the_model():
    src = open(os.path.join(API, "streamlit_app.py")).read()
    assert "force_real_result" not in src     # recordings were hard-coded to REAL before
