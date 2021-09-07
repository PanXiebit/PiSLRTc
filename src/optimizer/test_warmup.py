
import torch
import matplotlib.pyplot as plt
import math


warmup_steps = 10000
def slt_lr_schedule(step, d_model=512):
    arg1 = 1/ math.sqrt(step+1)
    arg2 = step * (warmup_steps ** -1.5)
    new_lr = 1 / math.sqrt(d_model) * min(arg1, arg2)
    return new_lr

lrs = []
for step in list(range(20000)):
    lrs.append(slt_lr_schedule(step))
print(max(lrs))
plt.plot(list(range(20000)), lrs)
plt.show()