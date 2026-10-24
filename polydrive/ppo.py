"""PPO from pixels on the real game. No human runs, no ghosts: just the track
and the agent's own crashes.

    uv run python -m polydrive.ppo --envs 6 --minutes 120
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env import ACTIONS, H, STACK, W, VecGame


class Policy(nn.Module):
    def __init__(self, n_actions=len(ACTIONS)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(STACK, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.body(torch.zeros(1, STACK, H, W)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, 512), nn.ReLU())
        self.pi = nn.Linear(512, n_actions)
        self.v = nn.Linear(512, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        with torch.no_grad():  # lean toward throttle actions so early exploration moves forward
            self.pi.bias[:3] = 1.0

    def forward(self, obs):
        h = self.fc(self.body(obs.float() / 255.0))
        return self.pi(h), self.v(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--steps", type=int, default=256, help="rollout length per env")
    ap.add_argument("--run", default="runs/summer1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = Policy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-4, eps=1e-5)
    state = {"updates": 0, "env_steps": 0, "best_frac": 0.0, "best_finish": None}
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]

    gamma, lam, clip, ent_coef, vf_coef = 0.99, 0.95, 0.1, 0.01, 0.5
    epochs, minibatch = 4, 256
    N, T = args.envs, args.steps

    env = VecGame(N)
    obs = env.reset()
    log_path = run / "episodes.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["wall_s", "env_steps", "end", "time", "frac", "checkpoints", "return"])

    buf_obs = torch.zeros((T, N, STACK, H, W), dtype=torch.uint8)
    buf_act = torch.zeros((T, N), dtype=torch.long)
    buf_logp = torch.zeros((T, N))
    buf_val = torch.zeros((T, N))
    buf_rew = torch.zeros((T, N))
    buf_done = torch.zeros((T, N))

    t_start = time.time()
    recent = []
    while time.time() - t_start < args.minutes * 60:
        lag = 0.0
        for t in range(T):
            o = torch.from_numpy(obs)
            with torch.no_grad():
                logits, v = net(o.to(dev))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            buf_obs[t] = o
            buf_act[t] = a.cpu()
            buf_logp[t] = dist.log_prob(a).cpu()
            buf_val[t] = v.cpu()
            t0 = time.perf_counter()
            obs, r, d, infos = env.step(a.cpu().numpy())
            lag = max(lag, time.perf_counter() - t0 - env.dt)
            buf_rew[t] = torch.from_numpy(r)
            buf_done[t] = torch.from_numpy(d)
            for info in infos:
                recent.append(info)
                writer.writerow([round(time.time() - t_start, 1), state["env_steps"] + t * N, info["end"],
                                 round(info["time"], 3), round(info["frac"], 4), info["checkpoints"],
                                 round(info["return"], 3)])
                if info["frac"] > state["best_frac"]:
                    state["best_frac"] = info["frac"]
                    torch.save(net.state_dict(), run / "best_progress.pt")
                if info["end"] == "finish" and (state["best_finish"] is None or info["time"] < state["best_finish"]):
                    state["best_finish"] = info["time"]
                    torch.save(net.state_dict(), run / "best_finish.pt")
                    print(f"*** new best finish {info['time']:.3f}s", flush=True)
        log.flush()
        state["env_steps"] += T * N

        # Freeze physics while learning so cars don't drive on stale keys.
        env.pause(True)
        with torch.no_grad():
            _, last_v = net(torch.from_numpy(obs).to(dev))
            last_v = last_v.cpu()
        adv = torch.zeros((T, N))
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + gamma * nv * nonterm - buf_val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_val

        b_obs = buf_obs.reshape(T * N, STACK, H, W).to(dev)
        b_act, b_logp = buf_act.reshape(-1).to(dev), buf_logp.reshape(-1).to(dev)
        b_adv, b_ret = adv.reshape(-1).to(dev), ret.reshape(-1).to(dev)
        stats = []
        for _ in range(epochs):
            idx = torch.randperm(T * N, device=dev)
            for s in range(0, T * N, minibatch):
                mb = idx[s:s + minibatch]
                logits, v = net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                a_mb = b_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - clip, 1 + clip) * a_mb).mean()
                vl = 0.5 * (v - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + vf_coef * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
        state["updates"] += 1
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "state": state}, run / "last.pt")
        (run / "state.json").write_text(json.dumps(state, indent=1))
        env.pause(False)

        pg_m, vl_m, ent_m = np.mean(stats, axis=0)
        done_eps = recent[-50:]
        fr = np.mean([e["frac"] for e in done_eps]) if done_eps else 0.0
        ends = {}
        for e in done_eps:
            ends[e["end"]] = ends.get(e["end"], 0) + 1
        print(f"[{(time.time()-t_start)/60:5.1f}m] upd {state['updates']} steps {state['env_steps']} "
              f"frac(avg50) {fr:.3f} best {state['best_frac']:.3f} finish {state['best_finish']} "
              f"ends {ends} ent {ent_m:.2f} vl {vl_m:.3f} lag {lag*1000:.0f}ms", flush=True)

    env.close()


if __name__ == "__main__":
    main()
