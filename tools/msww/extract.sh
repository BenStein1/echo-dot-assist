#!/bin/sh
# extract.sh SET_DIR RAW OUT.feat : run the stock engine (qemu) over RAW and dump the Dense:20 input/output per cycle.
# Needs: build/donut/pryon_extract (make DEVICE=donut build/donut/pryon_extract), qemu-arm, bwrap, the unpacked firmware.
R=$(cd "$(dirname "$0")/../.." && pwd)
SET=$1; RAW=$2; OUT=$3
rm -f "$OUT"
exec timeout -s KILL ${TMO:-900} bwrap --ro-bind / / --bind /tmp /tmp --bind "$R/build" "$R/build" --dev /dev --proc /proc --unshare-pid --die-with-parent \
  /usr/bin/qemu-arm -L "$R/firmware/donut/rootfs" -E LD_LIBRARY_PATH=/system/lib:/system/vendor/lib -E PRYON_EXTRACT_SPEED=${SPEED:-4} \
  "$R/build/donut/pryon_extract" -w "$SET/pryon.manifest" "$RAW" "$OUT"
