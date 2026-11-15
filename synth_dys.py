"""Severity-controlled synthetic speech from a PRETRAINED FastPitch (LJSpeech).

v2: the sentence is cut into phrases ONLY at the pause positions, and each phrase
is synthesised as one piece (natural coarticulation/prosody inside a phrase) with
a duration scale (length_scale) that grows with severity s in [0,1]. Phrases are
joined with silences whose number/length follow the paper's equations:
    P  = floor(alpha * s * Lb)        (Eq. 2, Lb = number of word boundaries)
    Dp = Db * (1 + beta * s)          (Eq. 3, seconds)
Pause positions are chosen at random (NOT learned). Pitch/energy are not varied;
only timing is. One synthetic voice (LJSpeech). --mode words gives the old
word-by-word behaviour.

Run with the TTS virtualenv:
  ~/venv_tts/bin/python synth_dys.py --manifest manifest.tsv --out synth/exp2 --n_utts 27000
Quick CPU test:  add --test
Utterance i depends only on (seed, i), so the output for n_utts=8000 is exactly
the first 8000 lines of the output for n_utts=27000.
Writes <out>/manifest.tsv (same columns as manifest.tsv) and <out>/meta.tsv.
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL = "tts_models/en/ljspeech/fast_pitch"
GRID = [1.0, 1.2, 1.4, 1.7, 2.0, 2.3]      # duration scales
DISCRETE_S = [0.0, 0.33, 0.66, 1.0]
OUT_SR = 16000
FADE = int(0.01 * OUT_SR)


def fade(w):
    n = min(FADE, len(w) // 2)
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        w = w.copy()
        w[:n] *= r
        w[-n:] *= r[::-1]
    return w


class Voice:
    def __init__(self, device):
        from TTS.api import TTS
        self.tts = TTS(MODEL, progress_bar=False)
        if device == "cuda":
            self.tts.to("cuda")
        self.model = self.tts.synthesizer.tts_model
        self.sr = self.tts.synthesizer.output_sample_rate
        self.cache = {}

    def seg(self, text, ls):
        key = (text, ls)
        if key not in self.cache:
            import librosa
            if len(self.cache) > 4000:
                self.cache.clear()
            self.model.length_scale = ls
            wav = np.asarray(self.tts.tts(text=text, split_sentences=False), dtype=np.float32)
            trimmed, _ = librosa.effects.trim(wav, top_db=35)
            if len(trimmed) > 400:
                wav = trimmed
            wav = resample_poly(wav, OUT_SR, self.sr).astype(np.float32)
            self.cache[key] = fade(wav)
        return self.cache[key]


def make_utt(i, a, voice, pool_multi, pool_single):
    rng = random.Random(f"{a.seed}-{i}")
    if pool_single and rng.random() < a.frac_single:
        spk, text = rng.choice(pool_single)
    else:
        spk, text = rng.choice(pool_multi)
    words = text.split()
    Lb = len(words) - 1
    if a.severity == "zero":
        s = 0.0
    elif a.severity == "discrete":
        s = rng.choice(DISCRETE_S)
    else:
        s = rng.random()
    if a.pause == "none" or Lb < 1:
        P, Dp = 0, 0.0
    elif a.pause == "binary":
        P, Dp = int(round(0.5 * Lb)), a.db * 1.5      # fixed, ignores severity
    else:
        P, Dp = int(math.floor(a.alpha * s * Lb)), a.db * (1 + a.beta * s)
    slots = set(rng.sample(range(Lb), min(P, Lb))) if Lb >= 1 else set()

    if a.mode == "words":
        segs = [(w, j) for j, w in enumerate(words)]
    else:
        segs, cur = [], []
        for j, w in enumerate(words):
            cur.append(w)
            if j in slots or j == len(words) - 1:
                segs.append((" ".join(cur), j))
                cur = []

    gap = np.zeros(int(0.04 * OUT_SR), np.float32)
    parts = []
    for k, (txt, j) in enumerate(segs):
        ls = (1.0 + a.stretch * s) * math.exp(rng.gauss(0, 0.15 * s))
        ls = min(max(ls, GRID[0]), GRID[-1])
        ls = min(GRID, key=lambda g: abs(g - ls))
        parts.append(voice.seg(txt, ls))
        if k < len(segs) - 1:
            parts.append(gap)
            if j in slots:
                parts.append(np.zeros(int(Dp * rng.uniform(0.8, 1.2) * OUT_SR), np.float32))
    wav = np.concatenate(parts)
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak > 0:
        wav = 0.8 * wav / peak
    return spk, text, s, len(slots), wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_utts", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["segments", "words"], default="segments")
    ap.add_argument("--severity", choices=["continuous", "discrete", "zero"], default="continuous")
    ap.add_argument("--pause", choices=["continuous", "binary", "none"], default="continuous")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--db", type=float, default=0.15, help="baseline pause, seconds")
    ap.add_argument("--stretch", type=float, default=1.0, help="how much severity slows speech")
    ap.add_argument("--frac_single", type=float, default=0.3)
    ap.add_argument("--max_sec", type=float, default=11.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    pool_multi = [(r["speaker"], r["text"]) for r in rows if len(r["text"].split()) >= 2]
    pool_single = [(r["speaker"], r["text"]) for r in rows
                   if r["group"] == "dysarthric" and len(r["text"].split()) == 1]
    print(f"prompt pools: {len(pool_multi)} multi-word, {len(pool_single)} single-word (dysarthric)")
    n = 4 if a.test else a.n_utts
    os.makedirs(a.out, exist_ok=True)
    voice = Voice("cpu" if (a.cpu or a.test) else "cuda")
    if a.test:
        x, y = voice.seg("hello", 1.0), voice.seg("hello", 2.0)
        print(f"length check: {len(x) / OUT_SR:.2f}s at scale 1.0 vs {len(y) / OUT_SR:.2f}s at 2.0 (expect about 2x)")

    man = open(os.path.join(a.out, "manifest.tsv"), "w")
    meta = open(os.path.join(a.out, "meta.tsv"), "w")
    man.write("utt_id\tspeaker\tgroup\twav\tduration\ttext\n")
    meta.write("utt_id\tseverity\tn_pauses\n")
    done, i, t0 = 0, 0, time.time()
    while done < n:
        spk, text, s, npz, wav = make_utt(i, a, voice, pool_multi, pool_single)
        i += 1
        dur = len(wav) / OUT_SR
        if dur > a.max_sec or dur < 0.3:
            continue
        utt = f"SYN{done:06d}_{spk}"
        path = os.path.abspath(os.path.join(a.out, utt + ".wav"))
        sf.write(path, wav, OUT_SR, subtype="PCM_16")
        man.write(f"{utt}\t{spk}\tsynthetic\t{path}\t{dur:.3f}\t{text}\n")
        meta.write(f"{utt}\t{s:.3f}\t{npz}\n")
        if a.test:
            print(f"{utt}: s={s:.2f} pauses={npz} dur={dur:.2f}s text='{text}'")
        done += 1
        if done % 500 == 0:
            print(f"{done}/{n} done, {time.time() - t0:.0f}s", flush=True)
    man.close()
    meta.close()
    print(f"wrote {done} utterances to {a.out}")


if __name__ == "__main__":
    main()
