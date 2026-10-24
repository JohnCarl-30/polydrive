"""Vectorized Polytrack environment on a virtual clock.

Each env is the real game in a headless Chromium page. Game time only moves
when the env steps it (see game.py), so every page runs as fast as it can
simulate and render, and nothing happens while the learner is updating.

The policy only gets pixels: a stack of small grayscale frames grabbed from
the game canvas. Car state from the physics worker is used for reward and
episode ends only.
"""
import asyncio
import base64
import time

import numpy as np

from .game import open_game
from .track import Centerline

W, H, STACK = 96, 64, 4
STEP_MS = 50  # 20 decisions per second of game time
# throttle x steering, plus brake. Reflex-style: it just presses keys.
ACTIONS = [
    ("KeyW",), ("KeyW", "KeyA"), ("KeyW", "KeyD"),
    (), ("KeyA",), ("KeyD",),
    ("KeyS",),
]

CHROME_ARGS = [
    "--use-angle=metal",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--mute-audio",
]

# Key presses are dispatched inside the page (one round trip per step).
_ACT = "([keys, ms, w, h, sub]) => window.__pd.act(keys).then(() => window.__pd.step(ms, w, h, sub))"


class Episode:
    def __init__(self):
        self.max_progress = 0.0
        self.progress = 0.0
        self.history = []
        self.steps = 0
        self.checkpoint = 0
        self.ret = 0.0


class GameEnv:
    def __init__(self, page, line, sub=1):
        self.page = page
        self.line = line
        self.sub = sub
        self.frames = np.zeros((STACK, H, W), np.uint8)
        self.ep = Episode()
        self.runs = 0

    @classmethod
    async def create(cls, context, line, sub=1):
        page = await open_game(context, virtual=True)
        await page.evaluate("() => { window.__pd.autoTick = false; }")
        env = cls(page, line, sub)
        env.runs = await page.evaluate("() => window.__pd.runs")
        return env

    def _push(self, img):
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = np.frombuffer(base64.b64decode(img), np.uint8).reshape(H, W)

    async def act(self, action):
        """Hold the keys for one step; return the car state (None before the start)."""
        img, car, runs = await self.page.evaluate(_ACT, [list(ACTIONS[action]), STEP_MS, W, H, self.sub])
        self._push(img)
        return car if runs == self.runs else None

    async def restart(self):
        started = await self.page.evaluate("() => window.__pd.act([]).then(() => window.__pd.car !== null)")
        if started:  # T is a no-op while the car is still waiting at the start
            self.runs = await self.page.evaluate("() => window.__pd.press('KeyT').then(() => window.__pd.runs)")
        self.ep = Episode()
        img, _, _ = await self.page.evaluate("([w, h]) => window.__pd.step(0, w, h, 1)", [W, H])
        self._push(img)
        self.frames[:] = self.frames[-1]
        return self.frames.copy()

    def reward(self, car):
        """Reward, done flag and end info for the latest car state."""
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
    """N games across a few browser instances (each has its own GPU process)."""

    def __init__(self, n, browsers=2, headless=True, sub=1, realtime=False, video_dir=None):
        from playwright.async_api import async_playwright

        self.loop = asyncio.new_event_loop()
        self.n = n
        self.realtime = realtime
        self._next = None

        async def setup():
            self.pw = await async_playwright().start()
            self.browsers = [await self.pw.chromium.launch(headless=headless, args=CHROME_ARGS)
                             for _ in range(min(browsers, n))]
            line = Centerline()
            opts = {"viewport": {"width": 192, "height": 108}}
            if video_dir:  # a viewable size for recordings
                opts = {"viewport": {"width": 960, "height": 540}, "record_video_dir": video_dir,
                        "record_video_size": {"width": 960, "height": 540}}
            elif not headless:
                opts = {"viewport": {"width": 480, "height": 270}}
            self.contexts = [await self.browsers[i % len(self.browsers)].new_context(**opts) for i in range(n)]
            return await asyncio.gather(*[GameEnv.create(c, line, sub) for c in self.contexts])

        self.envs = self.loop.run_until_complete(setup())

    def _all(self, coros):
        async def gather():
            return await asyncio.gather(*coros)
        return self.loop.run_until_complete(gather())

    def reset(self):
        self._next = time.perf_counter()
        return np.stack(self._all([e.restart() for e in self.envs]))

    def step(self, actions):
        async def one(e, a):
            car = await e.act(int(a))
            r, d, info = e.reward(car)
            e.ep.ret += r
            if d:
                info = {"end": info[0], "time": info[1], "progress": e.ep.max_progress,
                        "frac": e.ep.max_progress / e.line.total, "checkpoints": e.ep.checkpoint,
                        "return": e.ep.ret}
                return await e.restart(), r, d, info
            return e.frames.copy(), r, d, None

        if self.realtime:  # pace to the wall clock for watching
            self._next += STEP_MS / 1000
            time.sleep(max(0.0, self._next - time.perf_counter()))
        out = self._all([one(e, a) for e, a in zip(self.envs, actions)])
        obs = np.stack([o[0] for o in out])
        rew = np.array([o[1] for o in out], np.float32)
        done = np.array([o[2] for o in out], np.float32)
        return obs, rew, done, [o[3] for o in out if o[3] is not None]

    def close(self):
        async def shutdown():
            for c in self.contexts:
                await c.close()  # flushes recorded video
            for b in self.browsers:
                await b.close()
            await self.pw.stop()
        self.loop.run_until_complete(shutdown())
