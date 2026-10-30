#!/usr/bin/env bash
# Continuous thermal record. Runs on the HOST (nvidia-smi and the thermal zones
# are not visible inside the container). One line a minute, plus a loud note
# whenever CPU crosses the warn line.
set -uo pipefail
CSV=${CSV:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/thermal/thermal.csv"}
WARN=${WARN:-96}
[[ -f "$CSV" ]] || echo "iso_time,cpu_c,gpu_c,gpu_util,gpu_w,load1" > "$CSV"
while true; do
  cpu=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)
  cpu=$(( ${cpu:-0} / 1000 ))
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  g=$(echo "$gpu" | cut -d, -f1 | tr -d ' '); u=$(echo "$gpu" | cut -d, -f2 | tr -d ' '); w=NA
  l=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date -Is),$cpu,${g:-NA},${u:-NA},${w:-NA},$l" >> "$CSV"
  if (( cpu >= WARN )); then
    echo "$(date '+%a %H:%M')  CPU ${cpu}C >= ${WARN}C (gpu ${g:-NA}C, load $l)" \
      >> "${CSV%.csv}_warn.log"
  fi
  sleep 60
done
