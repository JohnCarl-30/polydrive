"""Sanity check: run the vectorized env with throttle-biased random actions."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from polydrive.env import VecGame

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
env = VecGame(n)
obs = env.reset()
rng = np.random.default_rng(0)
t0 = time.time()
steps = 0
while time.time() - t0 < 25:
    a = rng.choice(7, size=n, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    obs, r, d, infos = env.step(a)
    steps += 1
    for info in infos:
        print(info)
    if steps % 40 == 0:
        e = env.envs[0]
        print(f"step {steps} rate {steps/(time.time()-t0):.1f}/s prog {e.ep.progress:.1f} max {e.ep.max_progress:.1f} r {r}")
print("obs", obs.shape, obs.dtype, obs.mean())
env.close()
