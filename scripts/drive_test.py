"""Hold throttle, steer, restart; print car state and frame-grab timing."""
import base64
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

GAME = "https://app-polytrack.kodub.com/0.6.3/"
HOOK = (Path(__file__).parent.parent / "polydrive" / "hook.js").read_text()
OUT = Path("/private/tmp/claude-501/-Users-dyeyyyc/bdfafb69-cd58-4fa9-a950-ee0213fd8f40/scratchpad/ex")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    page = browser.new_page(viewport={"width": 960, "height": 540})
    page.route("**/vps.kodub.com/**", lambda r: r.abort())
    page.add_init_script(HOOK)
    page.goto(GAME)
    page.wait_for_function("() => window.__pd && window.__pd.track", timeout=30000)
    time.sleep(2)
    print("track len", page.evaluate("() => window.__pd.track.length"))

    def show(tag):
        c = page.evaluate("() => window.__pd.car")
        if c:
            print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items() if k != "quat"},
                  [round(x, 2) for x in c["quat"]])
        else:
            print(tag, None)

    page.keyboard.down("w")
    t0 = time.time()
    while time.time() - t0 < 4:
        show(f"{time.time()-t0:4.1f}")
        time.sleep(0.5)
    page.keyboard.down("d")
    time.sleep(1.5)
    show("after right")
    page.keyboard.up("d")
    page.keyboard.up("w")

    ts = []
    for i in range(50):
        a = time.perf_counter()
        b = page.evaluate("() => window.__pd.grab(96, 64)")
        ts.append(time.perf_counter() - a)
    print("grab ms median", 1000 * np.median(ts))
    img = np.frombuffer(base64.b64decode(b), np.uint8).reshape(64, 96)
    Image.fromarray(img).resize((384, 256), Image.NEAREST).save(OUT / "grab.png")

    page.keyboard.press("t")
    time.sleep(1)
    show("after T")
    page.screenshot(path=OUT / "after_t.png")
    browser.close()
