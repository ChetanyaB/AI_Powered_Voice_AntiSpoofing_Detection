"""Tiny espeak-ng TTS via ctypes (robotic synthetic speech for a sanity check)."""
import ctypes, os, sys, numpy as np, soundfile as sf, espeakng_loader as el
lib = ctypes.CDLL(el.get_library_path())
sr = lib.espeak_Initialize(2, 0, os.path.dirname(el.get_data_path()).encode(), 0)
buf = []
CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
def cb(wav, n, ev):
    if n > 0: buf.extend(wav[:n])
    return 0
cbf = CB(cb); lib.espeak_SetSynthCallback(cbf)
for voice, text, out in [(v, t, o) for v, t, o in [l.split("|") for l in sys.argv[1:]]]:
    buf.clear(); lib.espeak_SetVoiceByName(voice.encode())
    b = text.encode()
    lib.espeak_Synth(b, len(b) + 1, 0, 1, 0, 0, None, None); lib.espeak_Synchronize()
    sf.write(out, np.array(buf, dtype=np.int16), sr); print(out, sr, len(buf) / sr)
