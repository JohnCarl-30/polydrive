"""Run a trained policy in one window and report how far it gets.

    uv run python -m polydrive.watch runs/summer1/best_progress.pt --runs 3
    uv run python -m polydrive.watch runs/summer1/best_finish.pt --video out/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

from .env import ACTIONS, CHROME_ARGS, GameEnv
from .ppo import Policy
from .track import Centerline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--video", help="directory to save a video of the window")
    ap.add_argument("--trace", help="write car positions per run to this JSON file")
    args = ap.parse_args()

    net = Policy()
    sd = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(sd["net"] if "net" in sd else sd)
    net.eval()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, args=CHROME_ARGS)
        env = GameEnv(browser, Centerline(), video_dir=args.video)
        traces = []
        for k in range(args.runs):
            obs = env.restart()
            trace = []
            deadline = time.perf_counter()
            while True:
                with torch.no_grad():
                    logits, _ = net(torch.from_numpy(obs)[None])
                a = int(logits.argmax()) if args.greedy else int(torch.distributions.Categorical(logits=logits).sample())
                env.set_keys(ACTIONS[a])
                deadline += 0.05
                time.sleep(max(0.0, deadline - time.perf_counter()))
                car = env.observe()
                obs = env.frames.copy()
                r, done, info = env.reward(car)
                if car is not None:
                    trace.append(car["pos"])
                if done:
                    print(f"run {k+1}: {info[0]} at {info[1]:.3f}s, "
                          f"{100*env.ep.max_progress/env.line.total:.1f}% of track, "
                          f"{env.ep.checkpoint} checkpoints", flush=True)
                    traces.append({"end": info[0], "time": info[1], "pos": trace})
                    break
        env.set_keys(())
        time.sleep(1.0)
        if args.trace:
            Path(args.trace).write_text(json.dumps(traces))
        env.ctx.close()  # flushes the video file
        browser.close()


if __name__ == "__main__":
    main()
