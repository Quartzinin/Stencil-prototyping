#!/usr/bin/python3
filenames = input()
(inpf,outf) = filenames.split()
with open(inpf,'r') as f:
    with open(outf,'w') as w:
        numbers=list(map(float,f))[-30:]
        mean=int(sum(numbers)/len(numbers))
        text=str(mean)+"\n"
        w.write(text)

