"""Leave-one-speaker-out ASR: wav2vec2-base + CTC head (character level).

Baseline fold:
  python train_w2v.py --manifest manifest.tsv --test_spk F04 --out runs/w2v_base_F04
With synthetic data added to the TRAINING set (synth manifest needs a 'wav' column):
  python train_w2v.py ... --synth_manifest synth.tsv --out runs/w2v_exp1_F04
Quick check that everything runs:  add --smoke
Writes <out>/ref.txt, <out>/hyp.txt and <out>/result.json (same format as train_asr.py).
"""
import argparse
import copy
import csv
import json
import os
import random
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC

VOC = ["_"] + list("abcdefghijklmnopqrstuvwxyz' ")
C2I = {c: i for i, c in enumerate(VOC)}
MAX_TRAIN_SEC = 12.0   # longer training utterances are skipped (memory)
BATCH_SEC = 50.0       # total seconds of audio per batch


def read_rows(manifest):
    return [(r["utt_id"], r["speaker"], r["group"], r["wav"], r["text"])
            for r in csv.DictReader(open(manifest), delimiter="\t")]


def load(rows, train):
    data, skipped = [], 0
    for utt, spk, grp, wav, text in rows:
        try:
            y, sr = sf.read(wav)
        except Exception:
            skipped += 1
            continue
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        if train and len(y) > MAX_TRAIN_SEC * 16000:
            skipped += 1
            continue
        y = (y - y.mean()) / (y.std() + 1e-7)
        ids = [C2I[c] for c in text if c in C2I]
        if len(ids) == 0 or len(y) < 4000:
            skipped += 1
            continue
        data.append((torch.from_numpy(y), torch.tensor(ids), utt, " ".join(text.split())))
    return data, skipped


def batches(data, shuffle):
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    groups, cur, longest = [], [], 0
    for i in order:
        n = len(data[i][0])
        if cur and max(longest, n) * (len(cur) + 1) > BATCH_SEC * 16000:
            groups.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        groups.append(cur)
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        b = [data[i] for i in g]
        T = max(len(z[0]) for z in b)
        x = torch.zeros(len(b), T)
        for k, z in enumerate(b):
            x[k, :len(z[0])] = z[0]
        xl = torch.tensor([len(z[0]) for z in b])
        y = torch.cat([z[1] for z in b])
        yl = torch.tensor([len(z[1]) for z in b])
        yield x, xl, y, yl, b


def edit_distance(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(prev + (r[i - 1] != h[j - 1]), d[j] + 1, d[j - 1] + 1)
            prev = cur
    return d[len(h)]


def forward(model, x, xl, dev):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        logits = model(x.to(dev)).logits
    lp = logits.float().log_softmax(-1)
    ol = model._get_feat_extract_output_lengths(xl).to(dev)
    return lp, ol


def ctc_loss(lp, ol, y, yl, dev):
    return F.ctc_loss(lp.transpose(0, 1), y.to(dev), ol, yl.to(dev),
                      blank=0, reduction="mean", zero_infinity=True)


@torch.no_grad()
def evaluate(model, data, dev):
    model.eval()
    tot, n, errs, words, outs = 0.0, 0, 0, 0, []
    for x, xl, y, yl, b in batches(data, False):
        lp, ol = forward(model, x, xl, dev)
        tot += ctc_loss(lp, ol, y, yl, dev).item() * len(b)
        n += len(b)
        best = lp.argmax(-1).cpu().numpy()
        for k, item in enumerate(b):
            prev, s = 0, []
            for t in best[k][:int(ol[k])]:
                if t != prev and t != 0:
                    s.append(VOC[t])
                prev = t
            hyp = "".join(s).split()
            ref = item[3].split()
            errs += edit_distance(ref, hyp)
            words += len(ref)
            outs.append((item[2], ref, hyp))
    return tot / max(n, 1), 100.0 * errs / max(words, 1), outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--test_spk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", default="models/wav2vec2-base")
    ap.add_argument("--synth_manifest")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no_controls", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = read_rows(a.manifest)
    test = [r for r in rows if r[1] == a.test_spk]
    train = [r for r in rows if r[1] != a.test_spk
             and not (a.no_controls and r[2] == "control")]
    random.shuffle(train)
    nval = max(1, len(train) // 20)
    val, train = train[:nval], train[nval:]
    synth = read_rows(a.synth_manifest) if a.synth_manifest else []
    synth = [r for r in synth if r[1] != a.test_spk]   # never train on held-out speaker
    if a.smoke:
        train, val, test, synth = train[:48], val[:16], test[:16], synth[:16]
        a.epochs = 2

    t0 = time.time()
    tr, s1 = load(train + synth, True)
    va, s2 = load(val, False)
    te, s3 = load(test, False)
    print(f"fold {a.test_spk}: train {len(tr)} (synthetic {len(synth)}), val {len(va)}, "
          f"test {len(te)}, skipped train/val/test {s1}/{s2}/{s3}, "
          f"loaded in {time.time() - t0:.0f}s, device {dev}", flush=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        a.model_dir, vocab_size=len(VOC), pad_token_id=0,
        mask_time_prob=0.05, layerdrop=0.05, ctc_zero_infinity=True,
    ).to(dev)
    model.freeze_feature_encoder()
    head = [p for n, p in model.named_parameters() if n.startswith("lm_head")]
    body = [p for n, p in model.named_parameters()
            if not n.startswith("lm_head") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": 5e-5}, {"params": head, "lr": 1e-3}],
                            weight_decay=0.0)
    steps_per_ep = sum(1 for _ in batches(tr, False))
    total = steps_per_ep * a.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))

    best_loss, best_state, step = 1e9, None, 0
    for ep in range(a.epochs):
        model.train()
        te0, run, cnt = time.time(), 0.0, 0
        for x, xl, y, yl, b in batches(tr, True):
            lp, ol = forward(model, x, xl, dev)
            loss = ctc_loss(lp, ol, y, yl, dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            run += loss.item() * len(b)
            cnt += len(b)
        vl, vw, _ = evaluate(model, va, dev)
        print(f"epoch {ep + 1}: train loss {run / cnt:.3f}  val loss {vl:.3f}  "
              f"val WER {vw:.1f}  ({time.time() - te0:.0f}s)", flush=True)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, wer, outs = evaluate(model, te, dev)
    print(f"TEST WER {a.test_spk}: {wer:.1f}", flush=True)
    with open(os.path.join(a.out, "ref.txt"), "w") as fr, \
            open(os.path.join(a.out, "hyp.txt"), "w") as fh:
        for utt, ref, hyp in outs:
            fr.write(f"{utt} {' '.join(ref)}\n")
            fh.write(f"{utt} {' '.join(hyp)}\n")
    json.dump({"test_spk": a.test_spk, "wer": wer, "n_train": len(tr),
               "n_synth": len(synth), "epochs": a.epochs, "seed": a.seed,
               "asr": "wav2vec2-base+CTC"},
              open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
