"""Sanity check: run the vectorized env with throttle-biased random actions."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from polydrive.env import VecGame

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
env = VecGame(n, browsers=2)
obs = env.reset()
rng = np.random.default_rng(0)
t0 = time.time()
steps = 0
ends = []
while time.time() - t0 < 20:
    a = rng.choice(7, size=n, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    obs, r, d, infos = env.step(a)
    steps += 1
    ends += infos
dt = time.time() - t0
print(f"{steps*n/dt:.0f} env steps/s = {steps*n*0.05/dt:.1f}x real time; episodes {len(ends)}")
for info in ends[:8]:
    print(info)
print("obs", obs.shape, obs.dtype, obs.mean())
env.close()
