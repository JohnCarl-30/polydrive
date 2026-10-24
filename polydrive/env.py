"""Real-time vectorized Polytrack environment.

Each env is its own Chromium window running the real game. The policy only
gets pixels: a stack of small grayscale frames grabbed from the game canvas.
Car state from the physics worker is used for reward and episode ends only.
"""
import base64
import time

import numpy as np

from .game import open_game
from .track import Centerline

W, H, STACK = 96, 64, 4
# throttle x steering, plus brake. Reflex-style: it just presses keys.
ACTIONS = [
    ("w",), ("w", "a"), ("w", "d"),
    (), ("a",), ("d",),
    ("s",),
]

CHROME_ARGS = [
    "--use-angle=metal",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--mute-audio",
]


class Episode:
    def __init__(self):
        self.max_progress = 0.0
        self.progress = 0.0
        self.history = []
        self.steps = 0
        self.checkpoint = 0
        self.ret = 0.0


class GameEnv:
    def __init__(self, browser, line, video_dir=None):
        opts = {"viewport": {"width": 480, "height": 270}}
        if video_dir:  # 2x pixel density so the saved video isn't tiny
            opts.update(device_scale_factor=2, record_video_dir=video_dir,
                        record_video_size={"width": 960, "height": 540})
        self.ctx = browser.new_context(**opts)
        self.page = open_game(self.ctx)
        self.line = line
        self.keys = set()
        self.frames = np.zeros((STACK, H, W), np.uint8)
        self.ep = Episode()
        self.runs = self.page.evaluate("() => window.__pd.runs || 0")

    def set_keys(self, want):
        want = set(want)
        for k in self.keys - want:
            self.page.keyboard.up(k)
        for k in want - self.keys:
            self.page.keyboard.down(k)
        self.keys = want

    def observe(self):
        img, car, runs = self.page.evaluate(f"() => [window.__pd.grab({W}, {H}), window.__pd.car, window.__pd.runs || 0]")
        frame = np.frombuffer(base64.b64decode(img), np.uint8).reshape(H, W)
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = frame
        return car if runs == self.runs else None

    def restart(self):
        self.set_keys(())
        before, started = self.page.evaluate("() => [window.__pd.runs || 0, window.__pd.car !== null]")
        if started:  # T is a no-op while the car is still waiting at the start
            self.page.keyboard.press("t")
            self.page.wait_for_function(f"() => (window.__pd.runs || 0) > {before}", timeout=10000)
            before += 1
        self.runs = before
        self.ep = Episode()
        self.observe()
        self.frames[:] = self.frames[-1]
        return self.frames.copy()

    def reward(self, car):
        """Reward and done flag for the latest car state."""
        ep = self.ep
        ep.steps += 1
        if car is None:  # not started yet (first key press starts the clock)
            if ep.steps > 100:
                return -1.0, True, ("nostart", 0.0)
            return -0.005, False, None
        prog, lateral, height = self.line.project(car["pos"], ep.progress)
        ep.progress = prog
        r = -0.005
        if prog > ep.max_progress:
            r += 0.05 * (prog - ep.max_progress)
            ep.max_progress = prog
        ep.history.append(ep.max_progress)
        if car["nextCheckpointIndex"] > ep.checkpoint:
            r += 3.0 * (car["nextCheckpointIndex"] - ep.checkpoint)
            ep.checkpoint = car["nextCheckpointIndex"]
        t = car["frames"] / 1000.0
        if car["finishFrames"] is not None:
            ft = car["finishFrames"] / 1000.0
            return r + 20.0 + 0.5 * max(0.0, 60.0 - ft), True, ("finish", ft)
        if lateral > 22.0 or height < -8.0:
            return r - 3.0, True, ("off", t)
        if len(ep.history) > 60 and ep.max_progress - ep.history[-61] < 3.0:
            return r - 1.0, True, ("stall", t)
        if t > 75.0:
            return r, True, ("timeout", t)
        return r, False, None


class VecGame:
    def __init__(self, n, hz=20.0, headless=False):
        from playwright.sync_api import sync_playwright

        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(headless=headless, args=CHROME_ARGS)
        line = Centerline()
        self.envs = [GameEnv(self.browser, line) for _ in range(n)]
        self.dt = 1.0 / hz
        self.n = n
        self.deadline = None

    def reset(self):
        obs = np.stack([e.restart() for e in self.envs])
        self.deadline = time.perf_counter() + self.dt
        return obs

    def step(self, actions):
        for e, a in zip(self.envs, actions):
            e.set_keys(ACTIONS[int(a)])
        now = time.perf_counter()
        if self.deadline > now:
            time.sleep(self.deadline - now)
        self.deadline = max(self.deadline + self.dt, time.perf_counter())
        obs = np.zeros((self.n, STACK, H, W), np.uint8)
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, np.float32)
        infos = []
        for i, e in enumerate(self.envs):
            car = e.observe()
            r, d, info = e.reward(car)
            e.ep.ret += r
            rew[i], done[i] = r, d
            if d:
                infos.append({"end": info[0], "time": info[1], "progress": e.ep.max_progress,
                              "frac": e.ep.max_progress / e.line.total, "checkpoints": e.ep.checkpoint,
                              "return": e.ep.ret})
                obs[i] = e.restart()
            else:
                obs[i] = e.frames
        return obs, rew, done, infos

    def pause(self, paused):
        for e in self.envs:
            if paused:
                e.set_keys(())
            e.page.evaluate(f"() => window.__pd.pause({'true' if paused else 'false'})")
        if not paused:
            self.deadline = time.perf_counter() + self.dt

    def close(self):
        self.browser.close()
        self.pw.stop()
