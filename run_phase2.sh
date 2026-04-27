#!/bin/bash
# run_phase2.sh — phase-2 dragon work, auto-starts when overnight_chain ends.
#
# What this does (after current overnight chain finishes):
#   1. Forecasting (eval_sweep task 3) on E2, E6probe, E7probe at horizon=200
#   2. E2 trajectory: eval_sweep --tasks 1 on E2_epoch_5_ckpt with 3 seeds × 2 splits
#   3. Re-aggregate multiseed SUMMARY + regen figures
#
# Cap forecast_horizon=200 (per Lucas: don't run rollouts to 20k timesteps).
#
# Logs: logs/phase2_*.log
# =============================================================================
set -u
cd "$(dirname "$0")"

LOG=logs/phase2_$(date +%Y%m%d_%H%M).log
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

{
echo "[$(ts)] phase2 launched; waiting for overnight_chain to end first..."
while tmux has-session -t overnight_chain 2>/dev/null; do sleep 60; done
echo "[$(ts)] overnight_chain ended; proceeding with phase 2"

source .venv/bin/activate
sleep 30   # GPU cooldown

E2_CKPT=/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H32_S1_P4/lewm_s4_sensor_w1_H32_S1_P4_epoch_10_object.ckpt
E2_EP5=/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H32_S1_P4/lewm_s4_sensor_w1_H32_S1_P4_epoch_5_object.ckpt
E6_CKPT=/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H16_S10_P4/lewm_s4_sensor_w1_H16_S10_P4_epoch_1_object.ckpt
E7_CKPT=/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H16_S20_P4/lewm_s4_sensor_w1_H16_S20_P4_epoch_1_object.ckpt

H5_TEST=/home/lthil/.stable_worldmodel/scenario4_test_lewm.h5
H5_TH=/home/lthil/.stable_worldmodel/scenario4_test_hard_lewm.h5

# ── PART 1: Forecasting (task 3) on best ckpts ────────────────────────────
echo "[$(ts)] PART 1: forecasting (task 3) at horizon=200"
mkdir -p eval_results/forecast
for cfg in "E2:$E2_CKPT" "E6probe:$E6_CKPT" "E7probe:$E7_CKPT"; do
    name=${cfg%%:*}; path=${cfg#*:}
    if [ ! -f "$path" ]; then echo "  SKIP $name (no ckpt)"; continue; fi
    for split in test test_hard; do
        h5=$([ "$split" = test ] && echo "$H5_TEST" || echo "$H5_TH")
        outdir="eval_results/forecast/${name}_${split}"
        mkdir -p "$outdir"
        echo "[$(ts)] forecast $name on $split (horizon=200)"
        python eval_sweep.py --tasks 1 3 --no_parallel \
            --task1_seq_lens 1 \
            --forecast_horizon 200 \
            --hdf5 "$h5" \
            --out_dir "$outdir" \
            --seed 0 \
            "${name}_fc:$path" 2>&1 | tail -3
    done
done

# ── PART 2: E2 trajectory (epoch 5 vs epoch 10) — multi-seed task 1 ──────────
echo "[$(ts)] PART 2: E2 trajectory (epoch 5 vs 10)"
mkdir -p eval_results/multiseed
for seed in 0 2 3; do
    for split in test test_hard; do
        h5=$([ "$split" = test ] && echo "$H5_TEST" || echo "$H5_TH")
        for ep_pair in "E2ep5:$E2_EP5" "E2ep10:$E2_CKPT"; do
            ep_name=${ep_pair%%:*}; ep_path=${ep_pair#*:}
            outdir="eval_results/multiseed/${ep_name}_${split}_seed${seed}"
            if [ -f "$outdir/${ep_name}_${split}_s${seed}.json" ]; then
                echo "  [$(ts)] CACHED ${ep_name}_${split}_seed${seed}"
                continue
            fi
            mkdir -p "$outdir"
            echo "[$(ts)] $ep_name × $split × seed=$seed"
            python eval_sweep.py --tasks 1 --no_parallel --task1_seq_lens 1 \
                --hdf5 "$h5" \
                --out_dir "$outdir" \
                --seed "$seed" \
                "${ep_name}_${split}_s${seed}:$ep_path" 2>&1 | tail -2
        done
    done
done

# ── PART 3: re-aggregate + regen figures ──────────────────────────────────
echo "[$(ts)] PART 3: aggregate + regen figures"
python aggregate_multiseed.py | tail -20
python make_paper_figures.py | tail -8

# ── PART 4: counterfactual fidelity (online-simulator demo) ──────────────
echo "[$(ts)] PART 4: counterfactual fidelity — multi-ckpt × all 7 actions × 3 branch positions"
E1_CKPT=/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H16_S1_P4/lewm_s4_sensor_w1_H16_S1_P4_epoch_10_object.ckpt
E5_CKPT=/home/lthil/.stable_worldmodel/scenario4_sensor_w1_H8_S1_P4/lewm_s4_sensor_w1_H8_S1_P4_epoch_10_object.ckpt
PYTHONPATH=/home/lthil/thesis/rl_opendeck_simulator:$PYTHONPATH \
    python counterfactual_fidelity.py \
        --ckpts E1:$E1_CKPT E2:$E2_CKPT E5:$E5_CKPT E6probe:$E6_CKPT \
        --sim_h5 /home/lthil/.stable_worldmodel/scenario4_train_sensors.h5 \
        --eval_h5 /home/lthil/.stable_worldmodel/scenario4_test_lewm.h5 \
        --out_dir eval_results/counterfactual \
        --n_episodes 30 --horizon 100 \
        --actions 0 1 2 3 4 5 6 \
        --branch_fracs 0.25 0.5 0.75 2>&1 | tail -25

# ── PART 5: push to GitHub ───────────────────────────────────────────────
echo "[$(ts)] PART 5: commit + push"
git add eval_results/ figures/ make_paper_figures.py counterfactual_fidelity.py 2>&1 | tail -3
git -c user.email=dragon@le-wm -c user.name=dragon-agent \
    commit -m "phase2: forecasting at horizon=200 + E2 trajectory + counterfactual + figs" 2>&1 | tail -3
git push origin feature/option-b-sensor-native 2>&1 | tail -3
./aggregate_results.sh > /dev/null
./publish_tracking.sh 2>&1 | tail -3

echo "[$(ts)] PHASE 2 COMPLETE"
} 2>&1 | tee "$LOG"
