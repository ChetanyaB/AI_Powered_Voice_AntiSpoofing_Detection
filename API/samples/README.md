# Sample clips for testing

| Folder | Clips | Source |
|---|---|---|
| `real/` | 3 human audiobook recordings (LibriSpeech, 3 different speakers) | [LibriSpeech ASR corpus](https://www.openslr.org/12), CC BY 4.0, via the [librosa example data](https://github.com/librosa/data) |
| `synthetic/` | 3 machine-generated voices | Generated with espeak-ng (`make_synthetic_samples.py`) |

These clips are used by `evaluate_samples.py` and `tests/test_api.py`. espeak-ng is a classic formant synthesiser, **not** a neural deepfake, so these clips are only an out-of-domain sanity check.
