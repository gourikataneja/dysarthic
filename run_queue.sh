#!/bin/bash
# Queue: wait for the main synthesis -> synthesise the healthy-voice control -> Exp1 folds -> control folds.
# Safe to re-run: folds that already have runs/<name>_<SPK>/result.json are skipped.
cd ~/dysarthria || exit 1
echo "$(date) queue started, waiting for synthesis (tmux session 'synth')"
while tmux has-session -t synth 2>/dev/null; do sleep 30; done

if ! tail -n 1 synth_exp2.log | grep -q "wrote 27000"; then
  echo "$(date) STOP: main synthesis did not end with 'wrote 27000' (see synth_exp2.log)"
  exit 1
fi
head -n 8001 synth/exp2/manifest.tsv > synth/exp1_manifest.tsv
echo "$(date) exp1 manifest: $(wc -l < synth/exp1_manifest.tsv) lines"

if ! tail -n 1 synth_ctl_zero.log 2>/dev/null | grep -q "wrote 8000"; then
  echo "$(date) synthesising healthy-voice control (severity zero, no pauses)"
  ~/venv_tts/bin/python synth_dys.py --manifest manifest.tsv --out synth/ctl_zero \
      --n_utts 8000 --severity zero --pause none > synth_ctl_zero.log 2>&1
  if ! tail -n 1 synth_ctl_zero.log | grep -q "wrote 8000"; then
    echo "$(date) STOP: control synthesis failed (see synth_ctl_zero.log)"
    exit 1
  fi
fi

run_loop() {
  name=$1
  manifest=$2
  for s in F01 F03 F04 M01 M02 M03 M04 M05; do
    if [ -f runs/${name}_$s/result.json ]; then
      echo "$(date) skip ${name}_$s (done)"
      continue
    fi
    echo "$(date) start ${name}_$s"
    python -u train_w2v.py --manifest manifest.tsv --synth_manifest $manifest \
        --test_spk $s --out runs/${name}_$s > runs/${name}_$s.log 2>&1
    echo "$(date) end ${name}_$s: $(grep 'TEST WER' runs/${name}_$s.log)"
  done
}

run_loop w2v_exp1 synth/exp1_manifest.tsv
run_loop w2v_ctl_zero synth/ctl_zero/manifest.tsv
echo "$(date) queue finished"
