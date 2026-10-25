# 7. Persistent torch.compile cache for Monarch actors

## Problem

Monarch actors run with `TMPDIR=$XDG_RUNTIME_DIR/dgx-monarch` — a tmpfs under `/run/user/<uid>`.
PyTorch inductor and Triton autotune caches default to `$TMPDIR/torchinductor_<user>`, so every
reboot (or stop of the user's systemd session) threw away all compiled kernels, and the first image
after that paid a full `max-autotune-no-cudagraphs` compile again.

| First Krea image after a mesh recycle | Time |
|---|---|
| uncompiled | 22.3 s |
| compiled, cold cache | 32.7 s |
| compiled, warm disk cache | **20.4 s** |

## Fix

A oneshot user service that runs **before** the Monarch worker and replaces the cache directory
with a symlink to disk (`systemd/inductor-cache-link.sh`, `systemd/inductor-cache-link.service`):

```
$XDG_RUNTIME_DIR/dgx-monarch/torchinductor_<user>  ->  ~/.cache/torchinductor-monarch
```

- creates `$XDG_RUNTIME_DIR/dgx-monarch` with mode 0700 if missing (Monarch validates that the
  directory itself is owned by the user and private; it does not check what is inside)
- migrates an existing tmpfs cache into the disk directory once
- `Before=dgxm-worker.service`, enabled in the user's `default.target`

Install as the ComfyUI/Monarch user:

```bash
install -Dm755 systemd/inductor-cache-link.sh ~/.local/bin/inductor-cache-link.sh
install -Dm644 systemd/inductor-cache-link.service ~/.config/systemd/user/inductor-cache-link.service
systemctl --user daemon-reload && systemctl --user enable --now inductor-cache-link.service
```

Inductor keys its cache on source, torch/Triton version and device, so stale entries are ignored
after upgrades (they just take disk space; delete the directory after a torch upgrade).
