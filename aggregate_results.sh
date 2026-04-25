#!/bin/bash
# aggregate_results.sh — produce RESULTS.md, a one-stop cross-architecture
# summary of the dragon (JEPA) + orailixtower (AR-LSTM) campaigns.
#
# Reads:
#   coordination/dragon/results.md
#   coordination/orailixtower/results.md
#   coordination/dragon/current_state.md
#   coordination/orailixtower/current_state.md
#   hi_probe_metrics.csv  (dragon-side per-epoch probe data)
#
# Writes:
#   RESULTS.md — single file, idempotent.
#
# Safe to run anytime; idempotent.
# =============================================================================
set -u
cd "$(dirname "$0")"

OUT=RESULTS.md
TS=$(date -u +'%Y-%m-%d %H:%M UTC')

{
cat <<EOF
# Scenario-4 TurboSens — cross-architecture results

_Last regenerated: ${TS}_

Auto-aggregated from \`coordination/{dragon,orailixtower}/\` and
\`hi_probe_metrics.csv\`. **Do not edit by hand** — run \`./aggregate_results.sh\`.

---

## What's running right now

### Dragon (JEPA, RTX 5090)

EOF

if [ -f coordination/dragon/current_state.md ]; then
    sed -n '/^## Running now/,/^## /p' coordination/dragon/current_state.md \
      | sed '$d' | tail -n +2
else
    echo "_(coordination/dragon/current_state.md not found)_"
fi

cat <<EOF

### OrailixTower (AR-LSTM, RTX A6000)

EOF

if [ -f coordination/orailixtower/current_state.md ]; then
    sed -n '/^## Running now/,/^## /p' coordination/orailixtower/current_state.md \
      | sed '$d' | tail -n +2
else
    echo "_(coordination/orailixtower/current_state.md not yet pulled)_"
fi

cat <<EOF

---

## Per-architecture results

EOF

if [ -f coordination/dragon/results.md ]; then
    echo "### JEPA (dragon)"
    echo
    tail -n +2 coordination/dragon/results.md
    echo
fi

if [ -f coordination/orailixtower/results.md ]; then
    echo "### AR-LSTM (orailixtower)"
    echo
    tail -n +2 coordination/orailixtower/results.md
    echo
fi

cat <<EOF

---

## Dragon's per-epoch hi_probe trajectory (raw, from \`hi_probe_metrics.csv\`)

EOF

if [ -f hi_probe_metrics.csv ]; then
    # Mean across components per epoch — the headline trajectory
    echo "**HI mean across 10 components, per epoch:**"
    echo
    echo "| epoch | global_step | seq_len | R²    | RMSE     | Pearson-r |"
    echo "|-------|-------------|---------|-------|----------|-----------|"
    awk -F',' 'NR>1 && $4=="hi" && $5=="MEAN" {
        printf "| %s | %s | %s | %.4f | %.6f | %.4f |\n", $1, $2, $3, $6, $7, $8
    }' hi_probe_metrics.csv | sort -u
    echo
    echo "**Action-RUL probe per epoch:**"
    echo
    echo "| epoch | global_step | seq_len | R²    | RMSE   | Pearson-r |"
    echo "|-------|-------------|---------|-------|--------|-----------|"
    awk -F',' 'NR>1 && $4=="rul" && $5=="steps-to-action" {
        printf "| %s | %s | %s | %.4f | %.4f | %.4f |\n", $1, $2, $3, $6, $7, $8
    }' hi_probe_metrics.csv | sort -u
    echo
else
    echo "_(no hi_probe_metrics.csv yet)_"
fi

cat <<EOF

---

## Coordination logs

### Dragon log
\`\`\`
EOF
[ -f coordination/dragon/log.md ] && tail -10 coordination/dragon/log.md || echo "(none)"
cat <<EOF
\`\`\`

### OrailixTower log
\`\`\`
EOF
[ -f coordination/orailixtower/log.md ] && tail -10 coordination/orailixtower/log.md || echo "(none)"
cat <<EOF
\`\`\`

---

_To refresh: \`./aggregate_results.sh\`. Hooked into the \`peer_sync\` daemon
so it regenerates after every pull tick (~every 10 min)._
EOF
} > "$OUT"

echo "[$(date -u +%H:%MZ)] $OUT regenerated ($(wc -l < "$OUT") lines, $(wc -c < "$OUT") bytes)"
