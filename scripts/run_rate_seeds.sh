#!/usr/bin/env bash
# Training seeds for the rate-matched 2.5 Hz cell.
#
#   docker exec -d sim /scripts/run_rate_seeds.sh
#
# The control-rate null currently rests on one seed per arm: a policy trained on
# 2.5 Hz demonstrations and evaluated at action_repeat=4 reaches 60.7%, against
# 60.0% for the 10 Hz policy evaluated at action_repeat=1. A +0.7-point gap with
# a 95% episode-level interval of [-7.2, +8.5] is not evidence of equivalence,
# and an episode-level interval says nothing about training noise anyway.
#
# The 10 Hz arm already has four seeds (/checkpoints/bc_seeds/fs4_seed{1,2,3} and
# the seed-0 run in /checkpoints/s5d_fs4), reading 60.0, 56.3, 56.7, 55.3. Note
# that seed 0 is the best of those four, so the published comparison pairs the
# luckiest 10 Hz seed against the only 2.5 Hz seed we have. This script brings
# the 2.5 Hz arm to the same four.
#
# SEED COUNT IS FIXED HERE, BEFORE THE RUNS, AND WE REPORT WHATEVER COMES OUT:
#   control rate (7x7, 2.5 Hz arm) : 4 seeds   (1 exists, +3 here)
# Adding seeds until the gap moves would be the selection effect the paper is
# about, so the count is committed in advance and matches the 10 Hz arm.
set -uo pipefail
LOG=/logs/sweep/rate_seeds.log
STATUS=/logs/sweep/rate_seeds_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; EPISODES=300; GOAL_SEED=1000
DATASET=/demos/stage11_dagger_agg2_2p5hz.npz
OUT=/checkpoints/rate_seeds

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

[[ -f "$DATASET" ]] || { warn "missing $DATASET"; exit 1; }

say "training seeds 1-3 on the 2.5 Hz demonstrations"
for sd in 1 2 3; do
  d=/checkpoints/s5r_fs4_seed${sd}
  [[ -f "$d/bc_best.pt" ]] && { echo "  (exists) seed=$sd"; continue; }
  echo "  training seed=$sd"
  python3 -m bc.train --config /configs/bc_s5r_fs4_seed${sd}.yaml \
      --dataset "$DATASET" --seed "$sd" 2>&1 | tail -1
  [[ -f "$d/bc_best.pt" ]] || { warn "training seed=$sd"; exit 1; }
done

cool_gate
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet"; exit 1; }

say "evaluating at action_repeat=4 (2.5 Hz), the rate these were trained for"
for sd in 1 2 3; do
  out="$OUT/ratematched_fs4_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) $(basename "$out")"; continue; }
  say "  seed=$sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config "/configs/bc_s5r_fs4_seed${sd}.yaml" \
      --checkpoint "/checkpoints/s5r_fs4_seed${sd}/bc_best.pt" \
      --episodes "$EPISODES" --action-repeat 4 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result|WARNING' || true
  [[ -f "$out" ]] || warn "no output: seed=$sd"
done

/scripts/launch_sim_fleet.sh --stop; sleep 4
say "RATE SEEDS COMPLETE"

python3 - <<'PY' || true
import json, math, os
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)

arms = {
    "2.5 Hz (trained 2.5 Hz, ar=4)": [
        "/checkpoints/rate_matched/ratematched_fs4_2p5hz.json",
        "/checkpoints/rate_seeds/ratematched_fs4_seed1.json",
        "/checkpoints/rate_seeds/ratematched_fs4_seed2.json",
        "/checkpoints/rate_seeds/ratematched_fs4_seed3.json",
    ],
    "10 Hz  (trained 10 Hz,  ar=1)": [
        "/checkpoints/substrate_s5_dagger/bc_fs4_ar1.json",
        "/checkpoints/bc_seeds/fs4_seed1.json",
        "/checkpoints/bc_seeds/fs4_seed2.json",
        "/checkpoints/bc_seeds/fs4_seed3.json",
    ],
}
for name, files in arms.items():
    print(f"\n{name}")
    rates = []
    for sd, f in enumerate(files):
        if not os.path.exists(f):
            print(f"  seed {sd}: MISSING {f}"); continue
        d = json.load(open(f)); k, n = d["successes"], d["episodes"]
        lo, hi = wilson(k, n); rates.append(100 * k / n)
        print(f"  seed {sd}: {k:3d}/{n} = {100*k/n:5.1f}%  [{lo:4.1f},{hi:4.1f}]")
    if rates:
        m = sum(rates) / len(rates)
        sd_ = (sum((r - m) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5 if len(rates) > 1 else float("nan")
        print(f"  mean {m:5.1f}%  sd {sd_:4.1f}  (n={len(rates)} seeds)")
PY
