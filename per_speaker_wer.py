"""Per-speaker WER table from finished leave-one-speaker-out runs.

    python per_speaker_wer.py runs/w2v_base
    python per_speaker_wer.py runs/w2v_base runs/w2v_exp1 runs/w2v_exp2   (compare several)

For each prefix it looks for <prefix>_<SPK>/ref.txt and hyp.txt.
Prints WER per speaker, the mean over speakers, and the pooled WER (all words together),
plus WER split into single-word and multi-word utterances.
"""
import os
import sys

SPK = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]


def edit_distance(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(prev + (r[i - 1] != h[j - 1]), d[j] + 1, d[j - 1] + 1)
            prev = cur
    return d[len(h)]


def read(path):
    out = {}
    for line in open(path):
        parts = line.rstrip("\n").split(" ", 1)
        out[parts[0]] = parts[1].split() if len(parts) > 1 else []
    return out


def score(prefix, spk):
    d = f"{prefix}_{spk}"
    rp, hp = os.path.join(d, "ref.txt"), os.path.join(d, "hyp.txt")
    if not (os.path.exists(rp) and os.path.exists(hp)):
        return None
    ref, hyp = read(rp), read(hp)
    e = w = e1 = w1 = em = wm = 0
    for u, r in ref.items():
        err = edit_distance(r, hyp.get(u, []))
        e += err
        w += len(r)
        if len(r) == 1:
            e1 += err
            w1 += 1
        else:
            em += err
            wm += len(r)
    return dict(e=e, w=w, e1=e1, w1=w1, em=em, wm=wm, n=len(ref))


def pct(e, w):
    return 100.0 * e / w if w else float("nan")


def main():
    prefixes = sys.argv[1:]
    if not prefixes:
        print(__doc__)
        return
    res = {p: {s: score(p, s) for s in SPK} for p in prefixes}
    print(f"{'spk':6}{'utts':>6}{'words':>7}" + "".join(f"{os.path.basename(p):>16}" for p in prefixes))
    for s in SPK:
        first = next((res[p][s] for p in prefixes if res[p][s]), None)
        if not first:
            continue
        row = f"{s:6}{first['n']:6d}{first['w']:7d}"
        for p in prefixes:
            r = res[p][s]
            row += f"{pct(r['e'], r['w']):16.1f}" if r else f"{'-':>16}"
        print(row)
    print("-" * (19 + 16 * len(prefixes)))
    done = [s for s in SPK if all(res[p][s] for p in prefixes)]
    print(f"speakers finished in ALL columns: {len(done)} of {len(SPK)} -> {' '.join(done)}")
    if not done:
        return
    for label, f in (("mean over speakers", lambda r: pct(r["e"], r["w"])),):
        row = f"{label:19}"
        for p in prefixes:
            row += f"{sum(f(res[p][s]) for s in done) / len(done):16.1f}"
        print(row)
    for label, ke, kw in (("pooled (all words)", "e", "w"),
                          ("single-word utts", "e1", "w1"),
                          ("multi-word utts", "em", "wm")):
        row = f"{label:19}"
        for p in prefixes:
            row += f"{pct(sum(res[p][s][ke] for s in done), sum(res[p][s][kw] for s in done)):16.1f}"
        print(row)
    if len(prefixes) > 1:
        base = prefixes[0]
        b = sum(pct(res[base][s]["e"], res[base][s]["w"]) for s in done) / len(done)
        for p in prefixes[1:]:
            m = sum(pct(res[p][s]["e"], res[p][s]["w"]) for s in done) / len(done)
            print(f"relative improvement of {os.path.basename(p)} over {os.path.basename(base)} "
                  f"(mean over speakers): {100 * (b - m) / b:.1f}%")


if __name__ == "__main__":
    main()
