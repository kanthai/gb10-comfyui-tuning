# Tools

| File | What it does |
|---|---|
| `mma_peak2.cu` | mma.sync peak microbenchmark (BF16/FP8/INT8/NVFP4); build NVFP4 with `-gencode arch=compute_121a,code=sm_121a` |
| `tri_int8.py` | Triton INT8 GEMM config sweep on H3-like shapes |
| `test_v4.py` | checks that an INT8 SwiGLU linear compiles and equals eager (`ck_int8_patch_<ver>.py` next to it) |
| `ws_timing.py` | per-node timings of a ComfyUI prompt via the websocket |
| `vae_bench.py` | H3 VAE decode timing across tile sizes (uses `first_stage_model`, never `VAE.decode`) |
| `copy_bench.py` | GPU->host copy into fresh vs pre-touched vs pinned memory |
| `d2h_matrix.py` | full pageable-memory matrix (fresh / zero_ / read-touch / MADV_POPULATE_WRITE / cudaHostRegister / pinned, both directions, fault counts) |
| `frame_md5.py` | hash decoded video frames and audio of an MP4 (PyAV) for exactness checks |
| `krea_int8_from_bf16.py` | BF16 Krea 2 DiT -> INT8 tensorwise ConvRot-256 drop-in |
| `qerr.py` | per-layer NVFP4 error, RTN vs learned-rounding checkpoint |
| `clock_bench.sh` / `clock_sustain.sh` | burst and 6-minute sustained clock-cap comparisons with a thermal watchdog |
