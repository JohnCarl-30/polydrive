"""Launch Polytrack in Chromium with the hook installed.

Kodub's server (vps.kodub.com) is blocked: no ghosts, no leaderboard, no
uploads. The agent only ever sees the track and its own runs.
"""
from pathlib import Path

GAME = "https://app-polytrack.kodub.com/0.6.3/"
HOOK = (Path(__file__).parent / "hook.js").read_text()

# Inside the worker, after the track is parsed for a new car, post the placed
# parts back to the page: [x, y, z, partId, rotation, axis, color, checkpointOrder, startOrder].
_PARSE = 'const i=Ko.fromSaveString(t.trackData);if(null==i)throw new Error("Failed to load track");'
_EXPOSE = 'postMessage({messageType:99,parts:(()=>{const a=[];i.forEachPart((...x)=>a.push(x));return a})()});'


def _patch_worker(route):
    resp = route.fetch()
    body = resp.text()
    assert body.count(_PARSE) == 1, "simulation worker changed; update _PARSE"
    route.fulfill(response=resp, body=body.replace(_PARSE, _PARSE + _EXPOSE))


def open_game(context):
    """Open the game in a new page of a browser context (its own window)."""
    page = context.new_page()
    page.route("**/vps.kodub.com/**", lambda r: r.abort())
    page.route("**/simulation_worker.bundle.js", _patch_worker)
    page.add_init_script(HOOK)
    page.goto(GAME)
    page.wait_for_function("() => window.__pd && window.__pd.layout && window.__pd.car === null", timeout=60000)
    return page
