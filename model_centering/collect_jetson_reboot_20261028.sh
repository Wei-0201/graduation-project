#!/usr/bin/env bash
# Read-only snapshot: does not open camera/serial or start the car.
set -u
cd "$(dirname "$0")" || exit 1
report="jetson_reboot_report_20261028.txt"
exec > >(tee "$report") 2>&1
section() { printf '\n=== %s ===\n' "$1"; }
section 'Time / boot / board'
date -Is
uptime
uname -a
cat /proc/device-tree/model 2>/dev/null; printf '\n'
cat /etc/nv_tegra_release 2>/dev/null
last -x -n 12 2>/dev/null
section 'Memory and storage'
free -h
df -h / /dev/shm
section 'Processes (largest resident memory first)'
ps -eo pid,etime,pcpu,pmem,rss,stat,args --sort=-rss | head -n 18
section 'Temperatures (millidegrees C)'
for zone in /sys/class/thermal/thermal_zone*; do
    [ -d "$zone" ] || continue
    printf '%s: ' "$(cat "$zone/type" 2>/dev/null)"
    cat "$zone/temp" 2>/dev/null
done
section 'Power mode (query only)'
if command -v nvpmodel >/dev/null; then timeout 5 nvpmodel -q; fi
section 'Previous boot kernel (may be unavailable without persistent journal)'
timeout 12 journalctl -k -b -1 --no-pager -n 180
section 'Current boot kernel'
timeout 12 journalctl -k -b 0 --no-pager -n 180
section 'Pstore crash records'
for item in /sys/fs/pstore/*; do
    [ -f "$item" ] || continue
    printf '\n%s\n' "$item"
    head -c 32000 "$item"
done
section 'Recent car log'
latest=$(find . -maxdepth 1 -name 'car_model_*.log' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -n "$latest" ]; then tail -n 100 "$latest"; fi
section 'USB camera formats (query only)'
if command -v v4l2-ctl >/dev/null; then
    timeout 5 v4l2-ctl --list-devices
    timeout 5 v4l2-ctl -d /dev/video0 --list-formats-ext
fi
section 'Finished'
printf 'Saved beside this script: %s\n' "$report"
