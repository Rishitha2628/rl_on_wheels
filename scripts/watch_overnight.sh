#!/usr/bin/env bash
# Stall detector for the overnight queue.
#
# Checks that WORK IS ADVANCING, not merely that a process exists. An AIRL seed
# once sat blocked for 39 hours with four sims burning CPU, no errors logged and
# the watcher reporting healthy, because the watcher only checked liveness.
# Here the signal is the episode counter in the fleet log plus the count of
# result files; if neither moves for STALL_MIN minutes, say so loudly.
set -uo pipefail
STALL_MIN=${STALL_MIN:-45}
OUT=/logs/sweep/overnight_watch.log
exec > >(stdbuf -oL tee -a "$OUT") 2>&1

progress () {
  local eps files
  eps=$(grep -c handle_reset /logs/fleet/env0.log 2>/dev/null || echo 0)
  files=$(ls /checkpoints/substrate/*.json /checkpoints/substrate_s5/*.json \
             /checkpoints/substrate_s5_dagger/*.json /checkpoints/floor/*.json \
             /checkpoints/floor_s5/*.json /checkpoints/tune_s5/*/*.json \
             /checkpoints/sweep_s5/*/*.json /demos/dagger_s5_iter2.npz 2>/dev/null | wc -l)
  local steps=0
  pe=$(grep -l per_episode /checkpoints/substrate*/*.json /checkpoints/floor*/*.json \
       /checkpoints/rate_matched/*.json 2>/dev/null | wc -l)
  files=$(( ${files:-0} + ${pe:-0} ))
  d=$(grep -c handle_reset /logs/dagger_sim.log 2>/dev/null | head -1)
  eps=$(( ${eps:-0} + ${d:-0} ))
  for f in /logs/sweep/td3_s5_seed*.log /logs/sweep/airl_s5*.log /logs/sweep/td3_s5_shipped*.log; do
    [[ -f "$f" ]] && steps=$((steps + $(wc -c < "$f")))
  done
  echo "${eps}:${files}:${steps}"
}

last=$(progress); last_change=$SECONDS
while true; do
  sleep 300
  if ! pgrep -f '[r]un_overnight.sh|[r]un_stage5_all.sh|[r]un_stage5_fixup.sh|[r]un_airl_s5.sh|[r]un_rate_matched.sh|[r]un_paired.sh' > /dev/null; then
    echo "$(date '+%a %H:%M')  queue process gone - finished or died. Last: $last"
    exit 0
  fi
  now=$(progress)
  if [[ "$now" != "$last" ]]; then
    last=$now; last_change=$SECONDS
    echo "$(date '+%a %H:%M')  advancing (episodes:results:logbytes = $now)"
  else
    mins=$(( (SECONDS - last_change) / 60 ))
    if (( mins >= STALL_MIN )); then
      echo "$(date '+%a %H:%M')  *** STALLED ${mins} min, no progress (= $now) ***"
      notify-send --urgency=critical --app-name="rl_on_wheels" \
        "Overnight queue stalled" "${mins} min without progress" 2>/dev/null || true
      last_change=$SECONDS
    fi
  fi
done
