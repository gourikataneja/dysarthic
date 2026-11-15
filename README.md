# Severity-controlled synthetic speech for dysarthric ASR data augmentation

Code, logs and per-speaker results for the paper *Severity-Controlled Synthetic Speech for Dysarthric ASR Data Augmentation: An Empirical Study with a Pretrained TTS Model*.

A **pretrained, unmodified FastPitch** model (Coqui TTS, LJSpeech voice) synthesises speech from transcripts. A scalar `s` in [0, 1] sets the speaking rate (via the model's duration scale) and the number and duration of inserted pauses. The synthetic utterances are added to the real training data of a **wav2vec 2.0-base + CTC** recogniser, and word error rate (WER) is measured on held-out dysarthric speakers (leave-one-speaker-out) on **TORGO** (8 speakers) and **UASpeech** (14 speakers). No dysarthric speech is used to train the synthesiser. Pause positions are chosen at random; nothing in the generator is learned.

## Main findings

1. **Synthetic augmentation lowers WER on both corpora.**
2. **A healthy-voice synthetic control (`s = 0`, no pauses) of the same size gives most of the gain** on both corpora. On TORGO and UASpeech, dysarthric-style synthesis is not significantly better than the control.
3. **No evidence that continuous severity or severity-scaled pauses beat discrete severity or binary pauses** (TORGO ablation, differences within seed-to-seed noise).
4. All comparisons use few speakers (8 and 14) and one or two seeds, so small differences between synthetic variants cannot be resolved here.

## Results

All numbers are test WER (%), mean over held-out dysarthric speakers. "Better" counts speakers where the first system has lower WER. p-values are paired t-tests over speakers (Wilcoxon in brackets where reported). No correction for multiple comparisons.

### UASpeech (14 speakers, 12 epochs, seed 0)

8,000 synthetic utterances for Ctrl and Exp. 1 (all single-word prompts, so no pauses are inserted; Exp. 1 differs from Ctrl only in speaking rate).

| System | Mean WER |
|---|---|
| Base (real data only) | 59.6 |
| Ctrl (healthy-voice synthetic) | 58.0 |
| Exp. 1 (severity-controlled synthetic) | 57.8 |

| Comparison | Δ WER | Better / worse / tied | p |
|---|---|---|---|
| Exp. 1 vs Base | −1.8 | 12 / 2 / 0 | 0.005 (0.007) |
| Ctrl vs Base | −1.6 | 10 / 4 / 0 | 0.042 (0.020) |
| Exp. 1 vs Ctrl | −0.2 | 6 / 7 / 1 | 0.81 (0.65) |

About 90% of the Exp. 1 gain is reproduced by the healthy-voice control.

| Speaker | Base | Ctrl | Exp. 1 |
|---|---|---|---|
| F02 | 88.0 | 88.4 | 86.6 |
| F03 | 92.5 | 90.1 | 91.3 |
| F04 | 51.6 | 43.4 | 49.9 |
| M01 | 91.2 | 91.0 | 91.6 |
| M04 | 97.7 | 97.1 | 97.1 |
| M05 | 61.7 | 61.0 | 57.4 |
| M07 | 72.0 | 72.1 | 67.1 |
| M08 | 22.1 | 20.5 | 18.4 |
| M09 | 24.6 | 24.9 | 26.4 |
| M10 | 8.1 | 7.5 | 7.9 |
| M11 | 58.8 | 52.0 | 56.1 |
| M12 | 93.6 | 94.8 | 91.6 |
| M14 | 18.5 | 17.1 | 18.2 |
| M16 | 54.3 | 52.0 | 49.4 |
| **Mean** | **59.6** | **58.0** | **57.8** |

Speakers with near-total failure (M04, M12, M01, F03; WER above 90%) change little.

**Earlier 4-epoch schedule.** UASpeech was first trained for 4 epochs, at which validation WER was still falling. Mean WER there was Base 70.5, Ctrl 68.9, Exp. 1 68.2 (Exp. 1 vs Base p = 0.007; Exp. 1 vs Ctrl p = 0.27). The 12-epoch results above are the primary ones. The 4-epoch per-speaker values are in `results/ua_4epoch_wer.csv`.

### TORGO (8 dysarthric speakers, 15 epochs)

8,000 synthetic utterances for Ctrl, Exp. 1 and the ablations; 27,000 for Exp. 2 (the first 8,000 of which are Exp. 1).

| System | Seed 0 | Seed 1 | Mean of 2 seeds |
|---|---|---|---|
| Base | 50.4 | 51.2 | 50.8 |
| Ctrl (healthy-voice) | 44.8 | 45.7 | 45.3 |
| Exp. 1 | 43.3 | 45.0 | 44.1 |
| Exp. 2 (27k synthetic) | 39.3 | not run | – |
| Discrete severity | 42.4 | not run | – |
| Binary pause | 41.9 | not run | – |

| Comparison | Δ WER | Better | p |
|---|---|---|---|
| Exp. 1 vs Base, seed 0 | −7.1 | 8/8 | 0.008 (0.008) |
| Ctrl vs Base, seed 0 | −5.6 | 8/8 | 0.007 (0.008) |
| Exp. 1 vs Ctrl, seed 0 | −1.5 | 5/8 | 0.22 (0.38) |
| Exp. 2 vs Base, seed 0 | −11.1 | 8/8 | 0.003 (0.008) |
| Exp. 2 vs Ctrl, seed 0 (not volume-matched) | −5.5 | 8/8 | 0.022 (0.008) |
| Exp. 2 vs Exp. 1, seed 0 | −4.1 | 8/8 | 0.028 (0.008) |
| Exp. 1 vs Base, seed 1 | −6.2 | 8/8 | 0.002 (0.008) |
| Ctrl vs Base, seed 1 | −5.5 | 8/8 | <0.001 (0.008) |
| Exp. 1 vs Ctrl, seed 1 | −0.8 | 6/8 | 0.40 (0.38) |
| Exp. 1 vs Base, mean of 2 seeds | −6.6 | 8/8 | 0.001 (0.008) |
| Ctrl vs Base, mean of 2 seeds | −5.5 | 8/8 | 0.001 (0.008) |
| Exp. 1 vs Ctrl, mean of 2 seeds | −1.1 | 5/8 | 0.12 (0.15) |

With n = 8 the smallest attainable Wilcoxon p-value is 0.0078. Exp. 2 differs from Exp. 1 only in the amount of synthetic data, so its gain reflects data volume; no volume-matched control was run.

**Seed noise.** Per-speaker WER shifts by 2.4 to 3.7 points on average between seeds (up to 8.4 points, e.g. M02 Exp. 1: 62.4 vs 70.8), and the mean WER shifts by 0.8 to 1.6 points. The ablation differences below are of that size or smaller.

| Speaker | Base s0 | Ctrl s0 | Exp. 1 s0 | Exp. 2 s0 | Discrete s0 | Binary s0 | Base s1 | Ctrl s1 | Exp. 1 s1 |
|---|---|---|---|---|---|---|---|---|---|
| F01 | 59.7 | 58.6 | 57.6 | 45.3 | 51.8 | 51.1 | 56.5 | 54.7 | 54.0 |
| F03 | 36.9 | 30.8 | 31.2 | 29.2 | 34.9 | 32.0 | 43.6 | 34.6 | 29.9 |
| F04 | 9.1 | 6.4 | 6.0 | 5.3 | 6.0 | 5.7 | 9.4 | 5.6 | 7.1 |
| M01 | 68.6 | 59.6 | 53.6 | 52.5 | 58.3 | 59.9 | 67.9 | 64.0 | 60.6 |
| M02 | 74.1 | 68.9 | 62.4 | 55.8 | 57.9 | 57.7 | 74.9 | 68.1 | 70.8 |
| M03 | 7.1 | 4.5 | 4.0 | 3.8 | 3.9 | 4.0 | 9.9 | 4.2 | 3.5 |
| M04 | 79.4 | 75.4 | 77.1 | 74.2 | 72.8 | 69.1 | 81.6 | 74.3 | 74.0 |
| M05 | 68.3 | 54.1 | 54.8 | 48.0 | 53.4 | 55.9 | 65.5 | 60.2 | 59.7 |
| **Mean** | **50.4** | **44.8** | **43.3** | **39.3** | **42.4** | **41.9** | **51.2** | **45.7** | **45.0** |

### Ablation (TORGO, seed 0, 8,000 synthetic utterances)

| System | Mean WER | p vs Exp. 1 | p vs Ctrl |
|---|---|---|---|
| Ctrl | 44.8 | – | – |
| Exp. 1 (continuous severity + continuous pauses) | 43.3 | – | 0.22 |
| Discrete severity | 42.4 | 0.50 | 0.18 |
| Binary pause | 41.9 | 0.42 | 0.14 |

Discrete is better than Exp. 1 for 5 speakers, tied for 1 and worse for 2; Binary is better for 4, tied for 1 and worse for 3. The ablations are not significantly different from Exp. 1 or from Ctrl, and the experiment is underpowered (8 speakers, one seed). Discrete severity changes both speaking rate and pauses; binary pause also changes the average amount of pausing (see "How synthesis works").

## How synthesis works

`synth_dys.py` draws an utterance index and seed, picks a prompt from the training transcripts, and draws `s`:

- continuous (default): `s ~ U(0, 1)`; discrete: `s` from {0, 0.33, 0.66, 1.0}; control: `s = 0`.
- pause count `P = floor(alpha * s * Lb)` with `Lb` the number of word boundaries, pause length `Dp = Db * (1 + beta * s)` with `alpha = 1.0`, `beta = 2.0`, `Db = 0.15 s`; positions are random. Single-word prompts get no pauses.
- the sentence is cut into phrases only at the pause positions; each phrase is synthesised as one piece with duration scale `(1 + s) * exp(N(0, (0.15 s)^2))`, clipped to [1.0, 2.3] and snapped to {1.0, 1.2, 1.4, 1.7, 2.0, 2.3}. Pitch and energy are not varied; one voice is used.
- phrases are joined with a 40 ms gap plus the pause silence (times a random factor in [0.8, 1.2]); utterances shorter than 0.3 s or longer than 11.5 s are dropped.
- binary pause (ablation): `P = round(0.5 * Lb)` pauses of `1.5 * Db`, independent of `s` (Python rounding, so two-word prompts get none).

Prompts: for TORGO 70% multi-word and 30% single-word dysarthric transcripts; for UASpeech only single-word dysarthric transcripts (all 8,000 synthetic utterances are single words and contain no pauses). Leave-one-speaker-out: synthetic utterances whose prompt came from the held-out speaker are removed. Identical sentences or words contributed by other speakers remain, which is substantial for UASpeech, where all speakers share a vocabulary; this applies equally to every condition.

## Training and evaluation

`train_w2v.py`: wav2vec 2.0-base with a character CTC head, fine-tuned per fold. The checkpoint with the **lowest validation loss** (on real validation utterances from the training speakers) is used for testing; the learning rate has a linear warm-up and linear decay to zero. Seed 0 unless stated.

## Repository layout

```
results/
  per_speaker_wer.csv      one row per (run, speaker); run names: base, ctl_zero (Ctrl), exp1, exp2,
                           abl_discrete, abl_binary, *_s1 (seed 1, TORGO), ua_base / ua_ctrl / ua_exp1 (UASpeech, 12 epochs)
  ua_4epoch_wer.csv        earlier 4-epoch UASpeech values (14 speakers)
  result_json/             one JSON per run (test WER, n_train, n_synth, epochs, seed)
  logs/                    training logs (per-epoch loss and validation WER, test WER)
  synthesis_logs/          logs of the synthesis runs
  synthetic_manifests/     manifests of the synthetic utterances (audio is regenerated by synth_dys.py)
synth_dys.py               synthesiser
train_w2v.py               recogniser training and testing
prep_torgo.py, prep_uaspeech.py   manifest preparation
run_queue*.sh              job queues used for the experiments
```

## Reproducing

Synthesis (with the Coqui TTS virtualenv), e.g. the TORGO Exp. 2 set (Exp. 1 is its first 8,000 utterances):

```
~/venv_tts/bin/python synth_dys.py --manifest manifest.tsv --out synth/exp2 --n_utts 27000
# control: add --severity zero; ablations: --severity discrete  or  --pause binary
```

Training one UASpeech fold:

```
python -u train_w2v.py --epochs 12 --manifest manifest_ua.tsv --test_spk M09 --seed 0 --out runs/w2v_ua_base_M09
```

TODO: add the exact TORGO training command, the UASpeech synthesis command (the prompt-pool settings used for the single-word-only prompts), and the Python and package versions.

## Notes and caveats

- UASpeech was run for 14 of the dysarthric speakers: F02, F03, F04, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16. TODO: state why the remaining speaker(s) were excluded.
- One Ctrl run (UASpeech F02, 4 epochs) stayed at 100% WER on its first attempt and was rerun once with identical settings; the collapsed run is kept as `runs_ep4/w2v_ua_ctrl_F02_collapsed`.
- Only seed 0 exists for UASpeech, Exp. 2 and the ablations. TORGO Base, Ctrl and Exp. 1 have a second seed.
- Absolute WER is not comparable across TORGO (sentence-level) and UASpeech (isolated words).

## License and citation

TODO: add the license, author names and citation.
