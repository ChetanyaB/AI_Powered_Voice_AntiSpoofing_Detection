"""Run the deployed model on every clip in samples/real and samples/synthetic.

Usage: python evaluate_samples.py [samples_dir]
Writes ../docs/evaluation/sample_check.json
"""
import glob
import json
import os
import sys

from app.src.deepfake import infa_deepfake_with_scores

HERE = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "samples")
rows = []
for truth, folder in [("REAL", "real"), ("FAKE", "synthetic")]:
    for f in sorted(glob.glob(os.path.join(root, folder, "*"))):
        if f.endswith((".wav", ".ogg", ".flac", ".mp3")):
            status, label, scores = infa_deepfake_with_scores(f)
            rows.append({"file": f"{folder}/{os.path.basename(f)}", "truth": truth, "predicted": label,
                         "p_fake": scores.get("FAKE"), "p_real": scores.get("REAL"), "correct": label == truth})
            print(f"{rows[-1]['file']:40s} truth={truth:4s} pred={label:4s} P(fake)={scores.get('FAKE')}")
acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
print(f"\nAccuracy on {len(rows)} out-of-domain clips: {acc:.0%}")
out = os.path.join(HERE, "..", "docs", "evaluation", "sample_check.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"accuracy": acc, "clips": rows}, open(out, "w"), indent=2)
