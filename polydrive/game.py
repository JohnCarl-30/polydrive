"""Launch Polytrack in Chromium with the hook installed.

Kodub's server (vps.kodub.com) is blocked: no ghosts, no leaderboard, no
uploads. The agent only ever sees the track and its own runs.

In virtual mode the game runs on a clock the env controls: the physics
worker only advances when told to (message 98), and the page only renders
when the env ticks it. Game time no longer depends on wall time, so each
window runs as fast as it can simulate and render.
"""
from pathlib import Path

GAME = "https://app-polytrack.kodub.com/0.6.3/"
HOOK = (Path(__file__).parent / "hook.js").read_text()

# After the track is parsed for a new car, post the placed parts back to the page:
# [x, y, z, partId, rotation, axis, color, checkpointOrder, startOrder].
_PARSE = 'const i=Ko.fromSaveString(t.trackData);if(null==i)throw new Error("Failed to load track");'
_EXPOSE = 'postMessage({messageType:99,parts:(()=>{const a=[];i.forEachPart((...x)=>a.push(x));return a})()});'

# The worker's real-time loop `l` is scheduled off rAF/setInterval. In virtual
# mode we keep a handle to it instead and call it on demand.
_SCHEDULE = ('if(r)if(self.requestAnimationFrame){let d=performance.now();function u(){l(),self.requestAnimationFrame(u),'
             'd=performance.now()}u(),setInterval((()=>{performance.now()-d>100&&l()}),1e3/60)}else setInterval(l,1e3/60);'
             'else setInterval(h)')
_MANUAL = 'if(r)self.__pdStep=l;else setInterval(h)'
_VCLOCK = (
    'self.__vt=0;performance.now=()=>self.__vt;'
    'self.addEventListener("message",e=>{const d=e.data;if(d&&d.messageType===98){'
    'self.__vt+=d.ms;self.__pdStep&&self.__pdStep();postMessage({messageType:97});e.stopImmediatePropagation();}});\n'
)


def _patcher(virtual):
    async def patch(route):
        resp = await route.fetch()
        body = await resp.text()
        assert body.count(_PARSE) == 1, "simulation worker changed; update _PARSE"
        body = body.replace(_PARSE, _PARSE + _EXPOSE)
        if virtual:
            assert body.count(_SCHEDULE) == 1, "simulation worker changed; update _SCHEDULE"
            body = _VCLOCK + body.replace(_SCHEDULE, _MANUAL)
        await route.fulfill(response=resp, body=body)
    return patch


async def open_game(context, virtual=False):
    """Open the game in a new page of a (Playwright async) browser context."""
    page = await context.new_page()
    await page.route("**/vps.kodub.com/**", lambda r: r.abort())
    await page.route("**/simulation_worker.bundle.js", _patcher(virtual))
    if virtual:
        await page.add_init_script("window.__PD_VIRTUAL = true;")
    await page.add_init_script(HOOK)
    await page.goto(GAME)
    await page.wait_for_function("() => window.__pd && window.__pd.layout && window.__pd.car === null", timeout=60000)
    return page
