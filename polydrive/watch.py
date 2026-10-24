"""Run a trained policy in a visible window and report how far it gets.

    uv run python -m polydrive.watch runs/summer1-fast/best_progress.pt --runs 3
    uv run python -m polydrive.watch runs/summer1-fast/best_finish.pt --greedy --video out/ --sub 3

Plays at real-time pace (game time is still virtual, so results match training).
--fast skips the pacing; --sub renders extra frames per decision for smoother video.
"""
import argparse
import json
from pathlib import Path

import torch

from .env import VecGame
from .ppo import Policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--video", help="directory to save a video of the window")
    ap.add_argument("--sub", type=int, default=1, help="render ticks per 50 ms decision")
    ap.add_argument("--fast", action="store_true", help="don't pace to real time")
    ap.add_argument("--trace", help="write car positions per run to this JSON file")
    args = ap.parse_args()

    net = Policy()
    sd = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(sd["net"] if "net" in sd else sd)
    net.eval()

    env = VecGame(1, browsers=1, headless=False, sub=args.sub, realtime=not args.fast, video_dir=args.video)
    game = env.envs[0]
    traces = []
    obs = env.reset()
    trace = []
    while len(traces) < args.runs:
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs))
        a = logits.argmax(-1) if args.greedy else torch.distributions.Categorical(logits=logits).sample()
        obs, _, _, infos = env.step(a.numpy())
        if not infos and game.ep.steps:
            trace.append(game.ep.progress)
        for info in infos:
            print(f"run {len(traces)+1}: {info['end']} at {info['time']:.3f}s, "
                  f"{100*info['frac']:.1f}% of track, {info['checkpoints']} checkpoints", flush=True)
            traces.append({**info, "progress": trace})
            trace = []
    if args.trace:
        Path(args.trace).write_text(json.dumps(traces))
    env.close()


if __name__ == "__main__":
    main()
