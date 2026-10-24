# polydrive

A pixels-only agent that learns to drive [PolyTrack](https://www.kodub.com/apps/polytrack) (Summer 1) from scratch, inspired by aayu's Reflex-1.

**The rule:** the agent never sees a human run. No ghosts, no leaderboard. It gets the track and its own crashes.

## How it works

- **The real game, in real time.** Each environment is a Chromium window running the live game (v0.6.3). Playwright presses W/A/S/D like a player would.
- **Pixels in, keys out.** The policy sees a stack of 4 grayscale 96×64 frames grabbed from the game canvas and picks one of 7 key combos at 20 Hz. A small CNN is trained with PPO on the M4 GPU (MPS).
- **Reward comes from the track, not from people.** A hook reads the car state that the physics worker sends to the page (position, speed, checkpoints, finish time). Progress is measured along a centerline traced from the track's own part layout (`polydrive/track.py`). This is only used for reward and episode ends. The policy never sees it.
- **Clean room.** Requests to Kodub's server (`vps.kodub.com`) are blocked, so no ghosts or leaderboards are loaded and no times get uploaded.
- **Physics pauses while learning.** The trainer pauses every car through the worker's own `PauseCar` message during each PPO update, so the cars don't keep driving on stale keys.

Reward per 50 ms step: +0.05 per metre of new progress, +3 per checkpoint, +20 (plus a time bonus) for finishing, −3 for leaving the track, −1 for stalling, and −0.005 per step.

## Run

```bash
uv sync
uv run playwright install chromium

# train (opens N game windows)
uv run python -m polydrive.ppo --envs 6 --minutes 100

# watch a checkpoint drive; optionally save a video
uv run python -m polydrive.watch runs/summer1/best_progress.pt --runs 3
uv run python -m polydrive.watch runs/summer1/best_finish.pt --greedy --video out/
```

Training logs every episode to `runs/<name>/episodes.csv` and saves `last.pt`, `best_progress.pt` and `best_finish.pt`.

## Files

| file | what |
|---|---|
| `polydrive/hook.js` | injected into the page: decodes car state, grabs frames, pauses physics |
| `polydrive/game.py` | opens the game, blocks the server, patches the worker to expose the track layout |
| `polydrive/track.py` | Summer 1 centerline and progress projection |
| `polydrive/env.py` | real-time vectorized env, reward, episode ends |
| `polydrive/ppo.py` | CNN policy and PPO training loop |
| `polydrive/watch.py` | run a checkpoint, print results, record video |

The human world record on Summer 1 is 22.123s (Monstrosity).
