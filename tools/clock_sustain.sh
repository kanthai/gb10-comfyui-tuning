#!/bin/bash
# Sustained GB10 clock test: 6 min each at lgc 2000 / lgc 2200 / uncapped, per-minute throughput, throttle reasons, temps. Ends capped at 2000.
cd /tmp; SOC_LIMIT=98; GPU_LIMIT=90; DUR=360
soc() { m=0; for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = acpitz ] && { t=$(( $(cat $z/temp) / 1000 )); [ $t -gt $m ] && m=$t; }; done; echo $m; }
cap() { nvidia-smi -lgc 0,2000 >/dev/null; }
trap 'cap; echo "[restore] capped 2000"' EXIT INT TERM
cat > /tmp/sustain_load.py <<'PY'
import torch, time, sys
dur = float(sys.argv[1]); dev = "cuda"
shapes = [(8192, 6144, 6144), (8192, 16384, 6144), (8192, 6144, 16384)]
w8 = [(torch.randint(-127,127,(m,k),device=dev,dtype=torch.int8), torch.randint(-127,127,(n,k),device=dev,dtype=torch.int8).t()) for m,n,k in shapes]
bf = [(torch.randn(m, k, device=dev, dtype=torch.bfloat16), torch.randn(n, k, device=dev, dtype=torch.bfloat16)) for m, n, k in shapes]
fl = sum(2*m*n*k for m,n,k in shapes); t0 = time.time(); work = 0.0; minute = 0; mstart = t0
while time.time() - t0 < dur:
    for a, b in w8: torch._int_mm(a, b)
    for x, w in bf: torch.nn.functional.linear(x, w)
    torch.cuda.synchronize(); work += 2 * fl
    if time.time() - mstart >= 60:
        print(f"   minute {minute+1}: {work/(time.time()-mstart)/1e12:6.1f} T/s mixed INT8+BF16", flush=True); minute += 1; mstart = time.time(); work = 0.0
print(f"   TOTAL work done in {dur:.0f} s (relative): {minute} full minutes logged", flush=True)
PY
for MODE in 2000 2200 uncapped; do
  until [ "$(soc)" -lt 58 ]; do sleep 5; done
  if [ $MODE = uncapped ]; then nvidia-smi -rgc >/dev/null; else nvidia-smi -lgc 0,$MODE >/dev/null; fi
  echo "== $MODE (start SoC $(soc) C)"
  /opt/comfyui-venv/bin/python -I /tmp/sustain_load.py $DUR 2>&1 | grep -v -i "warn\|ck-int8" & BP=$!
  : > /tmp/cs_$MODE.csv; AB=0
  while kill -0 $BP 2>/dev/null; do
    r=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.active --format=csv,noheader,nounits | tr -d ' ')
    s=$(soc); echo "$r,$s" >> /tmp/cs_$MODE.csv; g=${r%%,*}
    if [ "$s" -ge $SOC_LIMIT ] || [ "$g" -ge $GPU_LIMIT ]; then echo "!! SoC $s C / GPU $g C: aborting + capping"; cap; pkill -f sustain_load.py; AB=1; fi
    sleep 1
  done; wait
  python3 - "$MODE" <<'PY'
import csv, sys, collections
r = [x for x in csv.reader(open(f"/tmp/cs_{sys.argv[1]}.csv")) if len(x) == 5]
g = [float(x[0]) for x in r]; p = [float(x[1]) for x in r]; c = [float(x[2]) for x in r]; s = [float(x[4]) for x in r]
reasons = collections.Counter(x[3] for x in r)
last = r[-120:]
print(f"   SoC {s[0]:.0f}->{max(s):.0f} C (last 2 min avg {sum(float(x[4]) for x in last)/len(last):.0f}) | GPU max {max(g):.0f} C | power avg {sum(p)/len(p):.1f} W | clock avg {sum(c)/len(c):.0f} MHz, last 2 min {sum(float(x[2]) for x in last)/len(last):.0f} | throttle reasons {dict(reasons.most_common(3))}")
PY
  [ $AB = 1 ] && break
done
