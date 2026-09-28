#!/usr/bin/env bash
# Add a swap file. Run with sudo:
#
#     sudo scripts/add_swap.sh
#
# Why: the machine has 15 GB RAM, zero swap, and roughly 1 GB free while the
# sweep runs. With no swap there is nothing to absorb a transient spike, so a
# momentary shortfall stalls the simulator instead of paging. That is what
# killed seed 0 -- all four workers timed out on /step at once because the sim
# could not respond within the service timeout.
#
# Swappiness is set to 10, not the default 60. This swap is a safety net, not
# a place to actively page from: at 60 the kernel would push parts of the
# running simulator out to disk and make the throughput problem worse. At 10 it
# stays essentially unused until memory is genuinely short, which is the point.

set -euo pipefail

SWAPFILE=/swapfile
SIZE_GB=8

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

if swapon --show 2>/dev/null | grep -q "$SWAPFILE"; then
  echo "[swap] $SWAPFILE is already active:"
  swapon --show
  exit 0
fi

avail_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if (( avail_gb < SIZE_GB + 10 )); then
  echo "[swap] Only ${avail_gb}G free on /; want ${SIZE_GB}G swap plus 10G headroom." >&2
  echo "[swap] Lower SIZE_GB in this script or free some disk first." >&2
  exit 1
fi

if [[ -e "$SWAPFILE" ]]; then
  echo "[swap] $SWAPFILE already exists but is not active; reusing it."
else
  echo "[swap] allocating ${SIZE_GB}G at $SWAPFILE ..."
  fallocate -l "${SIZE_GB}G" "$SWAPFILE" 2>/dev/null \
    || dd if=/dev/zero of="$SWAPFILE" bs=1M count=$((SIZE_GB * 1024)) status=progress
fi

chmod 600 "$SWAPFILE"
mkswap "$SWAPFILE" >/dev/null
swapon "$SWAPFILE"

# Safety net, not an active paging tier.
sysctl -w vm.swappiness=10 >/dev/null
echo "vm.swappiness=10" > /etc/sysctl.d/99-rl-swappiness.conf

echo
echo "[swap] active:"
swapon --show
free -h | head -2

echo
echo "[swap] This lasts until reboot. To make it permanent, append to /etc/fstab:"
echo "    $SWAPFILE none swap sw 0 0"
echo "[swap] To remove later:  sudo swapoff $SWAPFILE && sudo rm $SWAPFILE"
