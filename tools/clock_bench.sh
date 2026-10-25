#!/bin/bash
# GB10 clock-cap A/B: sustained GEMM at lgc 0,2000 vs uncapped (rgc), SoC/GPU watchdog. Ends with the cap state given in $1 (keep|remove).
cd /tmp
SOC_LIMIT=98; GPU_LIMIT=90
soc() { m=0; for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = acpitz ] && { t=$(( $(cat $z/temp) / 1000 )); [ $t -gt $m ] && m=$t; }; done; echo $m; }
cap() { nvidia-smi -lgc 0,2000 >/dev/null; }
restore() { cap; echo "[restore] cap re-applied (lgc 0,2000)"; }
trap restore INT TERM
cat > /tmp/gemm_load_gb.py <<'PY'
import torch, time, sys
dur = float(sys.argv[1]); dev = "cuda"
shapes = [(8192, 6144, 6144), (8192, 16384, 6144), (8192, 6144, 16384)]
ts = [(torch.randn(m, k, device=dev, dtype=torch.bfloat16), torch.randn(n, k, device=dev, dtype=torch.bfloat16)) for m, n, k in shapes]
w8 = [(torch.randint(-127,127,(m,k),device=dev,dtype=torch.int8), torch.randint(-127,127,(n,k),device=dev,dtype=torch.int8).t()) for m,n,k in shapes]
fl = sum(2*m*n*k for m,n,k in shapes)
torch.cuda.synchronize(); t0 = time.time(); bf = []; i8 = []
while time.time() - t0 < dur:
    s = time.time()
    for x, w in ts: torch.nn.functional.linear(x, w)
    torch.cuda.synchronize(); bf.append((time.time()-t0, fl/(time.time()-s)/1e12))
    s = time.time()
    for a, b in w8: torch._int_mm(a, b)
    torch.cuda.synchronize(); i8.append((time.time()-t0, fl/(time.time()-s)/1e12))
last = lambda v: sum(x for t,x in v if t > dur-20)/max(1,sum(1 for t,x in v if t > dur-20))
print(f"   BF16 GEMM last 20 s {last(bf):.1f} TFLOPS | INT8 GEMM last 20 s {last(i8):.1f} TOPS")
PY
for MODE in capped uncapped; do
  until [ "$(soc)" -lt 60 ]; do sleep 5; done
  if [ $MODE = capped ]; then cap; else nvidia-smi -rgc >/dev/null; fi
  echo "== $MODE (start SoC $(soc) C, GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C)"
  /opt/comfyui-venv/bin/python -I /tmp/gemm_load_gb.py 60 > /tmp/cb_$MODE.txt 2>&1 & BP=$!
  : > /tmp/cb_tele_$MODE.csv; AB=0
  while kill -0 $BP 2>/dev/null; do
    r=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits | tr -d ' ')
    s=$(soc); f=$(cat /sys/class/hwmon/hwmon*/fan1_input 2>/dev/null | head -1)
    echo "$r,$s,${f:-0}" >> /tmp/cb_tele_$MODE.csv
    g=${r%%,*}
    if [ "$s" -ge $SOC_LIMIT ] || [ "$g" -ge $GPU_LIMIT ]; then echo "!! SoC $s C / GPU $g C: aborting + capping"; cap; kill $BP; AB=1; fi
    sleep 0.5
  done
  grep -v -i "warn\|ck-int8" /tmp/cb_$MODE.txt
  python3 -c "
import csv
r=[[float(v) for v in x] for x in csv.reader(open('/tmp/cb_tele_$MODE.csv')) if x and len(x)==5]
s=[x for x in r if x[1]>20]
print(f'   GPU {r[0][0]:.0f}->{max(x[0] for x in r):.0f} C | SoC {r[0][3]:.0f}->{max(x[3] for x in r):.0f} C | power avg {sum(x[1] for x in s)/len(s):.1f} W | SM clock avg {sum(x[2] for x in s)/len(s):.0f} MHz (min {min(x[2] for x in s):.0f})')"
  [ $AB = 1 ] && break
done
if [ "${1:-keep}" = remove ]; then nvidia-smi -rgc >/dev/null; echo "[end] cap removed (rgc)"; else restore; fi
