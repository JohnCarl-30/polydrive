# polydrive

A pixels-only agent that learns to drive [PolyTrack](https://www.kodub.com/apps/polytrack) (Summer 1) from scratch, inspired by aayu's Reflex-1.

**The rule:** the agent never sees a human run. No ghosts, no leaderboard. It gets the track and its own crashes.

## How it works

- **The real game, faster than real time.** Each environment is the live game (v0.6.3) in a headless Chromium page. The game runs on a virtual clock: the physics worker's real-time loop is replaced so it only advances when the trainer says "50 ms", and the page's `performance.now` and `requestAnimationFrame` only move when the trainer ticks them. Six games across two browser processes run at about 29× real time on an M4 (580 steps/s, versus about 85 when they ran on the wall clock).
- **It presses keys.** Each step dispatches W/A/S/D key events into the page, so the game's own input handling runs.
- **Pixels in, keys out.** The policy sees a stack of 4 grayscale 96×64 frames grabbed from the game canvas and picks one of 7 key combos at 20 Hz. A small CNN is trained with PPO on the M4 GPU (MPS).
- **Reward comes from the track, not from people.** A hook reads the car state that the physics worker sends to the page (position, speed, checkpoints, finish time). Progress is measured along a centerline traced from the track's own part layout (`polydrive/track.py`). This is only used for reward and episode ends. The policy never sees it.
- **Clean room.** Requests to Kodub's server (`vps.kodub.com`) are blocked, so no ghosts or leaderboards are loaded and no times get uploaded.
- **Nothing moves while learning.** Game time is virtual, so the cars wait during each PPO update.

Reward per 50 ms step: +0.05 per metre of new progress, +3 per checkpoint, +20 (plus a time bonus) for finishing, −3 for leaving the track, −1 for stalling, and −0.005 per step.

## Run

```bash
uv sync
uv run playwright install chromium

# train headless (6 games across 2 browser processes)
uv run python -m polydrive.ppo --envs 6 --browsers 2 --minutes 90 --run runs/summer1-fast

# watch a checkpoint drive in a window at real-time pace; optionally save a video
uv run python -m polydrive.watch runs/summer1-fast/best_progress.pt --runs 3
uv run python -m polydrive.watch runs/summer1-fast/best_finish.pt --greedy --sub 3 --video out/
```

Training logs every episode to `runs/<name>/episodes.csv` and saves `last.pt`, `best_progress.pt` and `best_finish.pt`.

## Files

| file | what |
|---|---|
| `polydrive/hook.js` | injected into the page: virtual clock, key events, car state decoding, frame grabs |
| `polydrive/game.py` | opens the game, blocks the server, patches the worker (track layout, manual stepping) |
| `polydrive/track.py` | Summer 1 centerline and progress projection |
| `polydrive/env.py` | async vectorized env on the virtual clock, reward, episode ends |
| `polydrive/ppo.py` | CNN policy and PPO training loop |
| `polydrive/watch.py` | run a checkpoint, print results, record video |

The human world record on Summer 1 is 22.123s (Monstrosity).
