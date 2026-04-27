#!/bin/bash
# run_multiseed_eval.sh — multi-seed eval_sweep --tasks 1 over JEPA ckpts.
#
# Settles paper-rigor: probe head init was unseeded, so single-seed Pearson
# can swing ±0.2. Multi-seed mean ± std gives publishable numbers.
#
# Per ckpt × dataset, runs N seeds (default 3) and writes one JSON per
# (seed, ckpt, dataset) combination into eval_results/multiseed/.
#
# Aggregation → see aggregate_multiseed.py (run separately).
#
# Usage:
#   ./run_multiseed_eval.sh                 # default 3 seeds, all tracked ckpts
#   SEEDS="0 1 2 3 4" ./run_multiseed_eval.sh
# =============================================================================
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

OUT=eval_results/multiseed
mkdir -p "$OUT" logs/multiseed

SEEDS=${SEEDS:-"0 1 2"}
H5_TEST=/home/lthil/.stable_worldmodel/scenario4_test_lewm.h5
H5_TH=/home/lthil/.stable_worldmodel/scenario4_test_hard_lewm.h5

# (name, path) pairs — JEPA family
CKPTS=(
  "E1:/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H16_S1_P4/lewm_s4_sensor_w1_H16_S1_P4_epoch_10_object.ckpt"
  "E2:/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H32_S1_P4/lewm_s4_sensor_w1_H32_S1_P4_epoch_10_object.ckpt"
  "E3:/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H16_S5_P4/lewm_s4_sensor_w1_H16_S5_P4_epoch_10_object.ckpt"
  "E5:/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H8_S1_P4/lewm_s4_sensor_w1_H8_S1_P4_epoch_10_object.ckpt"
  "E6probe:/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H16_S10_P4/lewm_s4_sensor_w1_H16_S10_P4_epoch_1_object.ckpt"
)

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

START_TS=$(ts)
echo "[$START_TS] multi-seed eval starting; seeds=[$SEEDS], ckpts=${#CKPTS[@]}"

for ckpt_pair in "${CKPTS[@]}"; do
    ckpt_name=${ckpt_pair%%:*}
    ckpt_path=${ckpt_pair#*:}

    if [ ! -f "$ckpt_path" ]; then
        echo "[$(ts)] SKIP $ckpt_name — ckpt not found: $ckpt_path"
        continue
    fi

    for seed in $SEEDS; do
        for split in test test_hard; do
            tag="${ckpt_name}_${split}_seed${seed}"
            outdir="$OUT/$tag"
            logf="logs/multiseed/${tag}.log"

            if [ -f "$outdir/${ckpt_name}_${split}_s${seed}.json" ]; then
                echo "[$(ts)] CACHED $tag — skipping"
                continue
            fi

            mkdir -p "$outdir"
            h5_path=$([ "$split" = "test" ] && echo "$H5_TEST" || echo "$H5_TH")
            run_label="${ckpt_name}_${split}_s${seed}"

            echo "[$(ts)] RUN $tag (h5=$(basename $h5_path))"
            python eval_sweep.py --tasks 1 --no_parallel --task1_seq_lens 1 \
                --hdf5 "$h5_path" \
                --out_dir "$outdir" \
                --seed "$seed" \
                "${run_label}:$ckpt_path" > "$logf" 2>&1

            if [ $? -eq 0 ] && [ -f "$outdir/${run_label}.json" ]; then
                echo "[$(ts)] OK $tag"
            else
                echo "[$(ts)] FAIL $tag (see $logf)"
            fi
        done
    done
done

echo "[$(ts)] multi-seed eval complete (started $START_TS)"
echo "Aggregating..."
python aggregate_multiseed.py 2>&1 | tail -20
