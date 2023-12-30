import os
import sys
import numpy as np

steps = 200
dirname = f"pgd/whisper-small-35-{steps}-100/235"
tot = np.zeros(steps)
f = open(os.path.join(dirname, "log_wer.txt"), "r")
curr = 0
for i in range(len(f)):
    txt = f.read(i)
    if "ID" not in txt:
        tot[curr % steps] += float(txt)
tot = tot/(len(f)/201)
np.savetxt(os.path.join(dirname, "wer.npy"), tot)
