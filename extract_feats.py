"""Cache 40-dim log-mel features for every utterance in a manifest.

    python extract_feats.py manifest.tsv feats_real
    python extract_feats.py synth_manifest.tsv feats_synth   (later)
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf


def feat(job):
    utt, wav, out = job
    p = os.path.join(out, utt + ".npy")
    if os.path.exists(p):
        return 0
    try:
        y, sr = sf.read(wav)
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        m = librosa.feature.melspectrogram(
            y=y, sr=16000, n_fft=400, hop_length=160, n_mels=40
        )
        np.save(p, np.log(m + 1e-6).T.astype(np.float16))
        return 0
    except Exception as e:
        print("FAILED", utt, e)
        return 1


if __name__ == "__main__":
    manifest, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    jobs = [(r["utt_id"], r["wav"], out)
            for r in csv.DictReader(open(manifest), delimiter="\t")]
    with ProcessPoolExecutor(max_workers=8) as ex:
        failed = sum(ex.map(feat, jobs, chunksize=32))
    print(f"done: {len(jobs) - failed} ok, {failed} failed")
