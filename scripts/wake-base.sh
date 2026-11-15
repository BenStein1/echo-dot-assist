#!/bin/sh
# Reusable private Echo mic recordings; no training or installation.
# --connect HOST uses the existing signed, temporary ADB access helper.
set -eu
cd "$(dirname "$0")/.."
if [ "${1:-}" = --connect ]; then
    [ "$#" -ge 2 ] || { echo 'usage: scripts/wake-base.sh [--connect HOST] [phrase]' >&2; exit 1; }
    host=$2
    shift 2
    sh scripts/adb-wifi.sh "$host"
    exec python3 tools/wake-base.py --serial "$host:5555" "$@"
fi
exec python3 tools/wake-base.py "$@"
