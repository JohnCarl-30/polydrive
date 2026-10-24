// Injected before the game loads. Exposes window.__pd for the Python env:
// latest decoded car state, a small grayscale frame grab, and (in virtual
// mode) a step() that advances physics and rendering by a fixed amount of
// game time, as fast as the machine allows.
(() => {
  const pd = (window.__pd = { car: null, runs: 0, controls: 0 });
  const virtual = !!window.__PD_VIRTUAL;

  // Keep the WebGL drawing buffer so we can read the canvas between frames.
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    if (type === "webgl" || type === "webgl2") {
      attrs = Object.assign({}, attrs, { preserveDrawingBuffer: true });
    }
    return getContext.call(this, type, attrs);
  };

  // Virtual clock: performance.now and requestAnimationFrame only move when
  // tick() is called. A real-time pump drives them while the game loads.
  if (virtual) {
    let vt = performance.now();
    performance.now = () => vt;
    let queue = [];
    let nextId = 1;
    window.requestAnimationFrame = (cb) => { queue.push([nextId, cb]); return nextId++; };
    window.cancelAnimationFrame = (id) => { queue = queue.filter(([i]) => i !== id); };
    pd.tick = (ms) => {
      vt += ms;
      const q = queue;
      queue = [];
      for (const [, cb] of q) {
        try { cb(vt); } catch (err) { console.error(err); }
      }
    };
    pd.autoTick = true;
    setInterval(() => { if (pd.autoTick) pd.tick(16); }, 16);
  }

  // Decode the head of a CarState buffer (layout from the game's own decoder).
  // Worker buffers start with a 4-byte car id.
  function decode(buf) {
    const b = new Uint8Array(buf, 4);
    const v = new DataView(buf, 4);
    let i = 0;
    const frames = b[0] | (b[1] << 8) | (b[2] << 16);
    i = 3;
    const speedKmh = v.getFloat32(i, true);
    i += 4;
    const flags = b[i];
    i += 1;
    const finished = !!(flags & 2);
    let finishFrames = null;
    if (finished) {
      finishFrames = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16);
      i += 3;
    }
    const nextCheckpointIndex = v.getUint16(i, true);
    i += 2;
    const p = [v.getFloat32(i, true), v.getFloat32(i + 4, true), v.getFloat32(i + 8, true)];
    i += 12;
    const q = [v.getFloat32(i, true), v.getFloat32(i + 4, true), v.getFloat32(i + 8, true), v.getFloat32(i + 12, true)];
    i += 16;
    const nImpulses = b[i];
    i += 1;
    let impulse = 0;
    for (let k = 0; k < nImpulses; k++) impulse = Math.max(impulse, v.getFloat32(i + 4 * k, true));
    const contacts = ((flags >> 3) & 1) + ((flags >> 4) & 1) + ((flags >> 5) & 1) + ((flags >> 6) & 1);
    return {
      frames, speedKmh, hasStarted: !!(flags & 1), finishFrames, nextCheckpointIndex,
      pos: p, quat: q, impulse, contacts,
    };
  }

  const RealWorker = window.Worker;
  window.Worker = function (url, opts) {
    const w = new RealWorker(url, opts);
    if (!String(url).includes("simulation_worker")) return w;
    const post = w.postMessage.bind(w);
    w.addEventListener("message", (e) => {
      const d = e.data;
      if (!d) return;
      // 99 and 97 are ours (injected by game.py): track layout, step finished.
      if (d.messageType === 99) { pd.layout = d.parts; e.stopImmediatePropagation(); return; }
      if (d.messageType === 97) { const r = pd._stepDone; pd._stepDone = null; r && r(); e.stopImmediatePropagation(); return; }
      // 10 = UpdateResult; the player car is the only car we create.
      if (d.messageType === 10 && d.carStateBuffers && d.carStateBuffers.length) {
        const last = d.carStateBuffers[d.carStateBuffers.length - 1];
        pd.carId = new DataView(last).getUint32(0, true);
        pd.car = decode(last);
      }
    });
    w.postMessage = (msg, ...rest) => {
      if (msg && msg.messageType === 0 && msg.isRealtime) {
        pd.worker = post;
        // 7 = PauseCar. Freezes physics while the learner updates (real-time mode).
        pd.pause = (paused) => { if (pd.car !== null) post({ messageType: 7, carId: pd.carId, isPaused: paused }); };
      }
      // 3 = CreateCar (a new run), 6 = ControlCar (key state changed).
      if (msg && msg.messageType === 3) { pd.car = null; pd.runs += 1; }
      if (msg && msg.messageType === 6) pd.controls += 1;
      return post(msg, ...rest);
    };
    return w;
  };
  window.Worker.prototype = RealWorker.prototype;

  // Downscaled grayscale frame as base64, w*h bytes.
  const small = document.createElement("canvas");
  const ctx = small.getContext("2d", { willReadFrequently: true });
  pd.grab = (w, h) => {
    small.width = w;
    small.height = h;
    ctx.drawImage(document.getElementById("screen"), 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const out = new Uint8Array(w * h);
    for (let k = 0, j = 0; k < out.length; k++, j += 4) {
      out[k] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
    }
    let s = "";
    for (let k = 0; k < out.length; k += 8192) s += String.fromCharCode.apply(null, out.subarray(k, k + 8192));
    return btoa(s);
  };

  // Virtual mode: advance game time by ms in `sub` slices. Each slice runs the
  // physics frames synchronously in the worker, then one render tick. Then
  // grab. Resolves to [frame, car, runs].
  const physics = (ms) => new Promise((resolve) => {
    pd._stepDone = resolve;
    pd.worker({ messageType: 98, ms });
  });
  // Key presses as page events, so the game's own input handling runs.
  const held = new Set();
  const key = (type, code) => document.body.dispatchEvent(
    new KeyboardEvent(type, { code, key: code.replace("Key", "").toLowerCase(), bubbles: true }));
  pd.act = async (codes) => {
    const want = new Set(codes);
    for (const c of [...held]) if (!want.has(c)) { key("keyup", c); held.delete(c); }
    for (const c of want) if (!held.has(c)) { key("keydown", c); held.add(c); }
  };
  pd.press = async (code) => {
    const before = pd.runs;
    key("keydown", code);
    key("keyup", code);
    for (let k = 0; k < 200 && pd.runs === before; k++) await new Promise((r) => setTimeout(r, 1));
  };

  pd.step = async (ms, w, h, sub) => {
    for (let k = 0; k < sub; k++) {
      await physics(ms / sub);
      pd.tick(ms / sub);
    }
    return [pd.grab(w, h), pd.car, pd.runs];
  };
})();
