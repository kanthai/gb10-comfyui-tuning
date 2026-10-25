#!/bin/sh
# Keep the torch.compile (inductor/autotune) cache of DGX Monarch actors on disk.
# Monarch sets the actors' TMPDIR to $XDG_RUNTIME_DIR/dgx-monarch (tmpfs), so the cache
# (torchinductor_$USER) is lost on reboot. Point it at ~/.cache instead. Runs before dgxm-worker.
set -eu
R="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/dgx-monarch"
D="$HOME/.cache/torchinductor-monarch"
L="$R/torchinductor_$(id -un)"
mkdir -p "$D" && chmod 700 "$D"
[ -d "$R" ] || install -d -m 700 "$R"      # same owner/mode Monarch's _private_directory expects
if [ -d "$L" ] && [ ! -L "$L" ]; then      # carry over a cache built before this service existed
    cp -a "$L/." "$D/" && rm -rf "$L"
fi
ln -sfn "$D" "$L"
echo "inductor cache: $L -> $D ($(du -sh "$D" | cut -f1))"
