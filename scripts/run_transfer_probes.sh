#!/usr/bin/env bash
# Three evaluation-only probes of the transfer failure. No training.
#
#   docker exec -d sim /scripts/run_transfer_probes.sh
#
# Waits for run_both_arenas.sh -- one sim instance.
#
# A. Time limit. Doubling max_episode_steps 450 -> 900 in the unseen arena. If
#    success jumps, failures were detours and getting lost rather than
#    collisions, and the arena is less "harder" than it looks. A home run at
#    900 is the control: if home also jumps, the limit was binding everywhere.
#
# B. The wrapper, away from home. The hand-written controllers are worth 8.3
#    points at home. Their thresholds were tuned on the 7x7 arena, so this asks
#    whether hand engineering transfers better or worse than the learned policy.
#
# C. DAgger, away from home. We have 0-round (s5_fs4) and 1-round (s5d_fs4)
#    policies. The project adopted DAgger to cure overfitting; this is the
#    direct test of whether it bought any transfer. NOTE: only one seed of the
#    0-round policy exists, so this probe is a single cell against three.
#
# References: unseen BC+DAgger (450 steps, no overrides) = 17.0% over 3 seeds;
# home = 61.3%.
set -uo pipefail
LOG=/logs/sweep/transfer_probes.log
STATUS=/logs/sweep/transfer_probes_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/transfer_probes
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

for _ in $(seq 1 480); do
  pgrep -f 'run_both_arenas' >/dev/null 2>&1 || break
  sleep 30
done
pgrep -f 'run_both_arenas' >/dev/null 2>&1 && { echo "ABORT: both_arenas still running" >&2; exit 1; }
say "both_arenas clear; starting transfer probes"

cfg ()  { [[ $1 == 0 ]] && echo /configs/bc_s5d_fs4.yaml || echo /configs/bc_s5d_fs4_seed$1.yaml; }
ck ()   { [[ $1 == 0 ]] && echo /checkpoints/s5d_fs4/bc_best.pt || echo /checkpoints/s5d_fs4_seed$1/bc_best.pt; }
run ()  { # out, config, checkpoint, stage-flags...
  local out=$1; shift
  [[ -f "$out" ]] && { echo "  (exists) $(basename "$out")"; return; }
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --episodes "$EPISODES" \
      --action-repeat 1 --quiet --out "$out" "$@" 2>&1 | grep -E 'rewound|Result' || true
}

# ── A + B + C in the unseen arena ───────────────────────────────────────────
/scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4
/scripts/launch_sim_fleet.sh -n 1 -s 6 --fast --goal-seed "$GOAL_SEED" || exit 1

say "A. unseen, 900-step limit"
for sd in 0 1 2; do
  say "  long seed $sd"
  run "$OUT/long_unseen_seed${sd}.json" --config /configs/bc_s5d_fs4_long_seed${sd}.yaml \
      --checkpoint "$(ck $sd)" --no-overrides
done

say "B. unseen, WITH the hand-written wrapper"
for sd in 0 1 2; do
  say "  wrapper seed $sd"
  run "$OUT/wrapper_unseen_seed${sd}.json" --config "$(cfg $sd)" --checkpoint "$(ck $sd)"
done

say "C. unseen, 0 DAgger rounds (single seed available)"
run "$OUT/nodagger_unseen_seed0.json" --config /configs/bc_s5_fs4.yaml \
    --checkpoint /checkpoints/s5_fs4/bc_best.pt --no-overrides

/scripts/launch_sim_fleet.sh --stop; sleep 4

# ── control: does the 900-step limit also help at home? ─────────────────────
say "A-control. home, 900-step limit, seed 0"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1
run "$OUT/long_home_seed0.json" --config /configs/bc_s5d_fs4_long_seed0.yaml \
    --checkpoint "$(ck 0)" --no-overrides
/scripts/launch_sim_fleet.sh --stop; sleep 4

say "TRANSFER PROBES COMPLETE"
