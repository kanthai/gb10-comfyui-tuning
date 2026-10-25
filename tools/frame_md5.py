import av, hashlib, sys
for f in sys.argv[1:]:
    c = av.open(f); hv = hashlib.md5(); ha = hashlib.md5(); n = 0
    for fr in c.decode(video=0): hv.update(fr.to_ndarray(format="rgb24").tobytes()); n += 1
    c = av.open(f)
    for fr in c.decode(audio=0): ha.update(fr.to_ndarray().tobytes())
    print(f.split("/")[-1], "frames", n, "video", hv.hexdigest()[:16], "audio", ha.hexdigest()[:16])
