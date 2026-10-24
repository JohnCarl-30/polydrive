// Injected before the game loads. Exposes window.__pd for the Python env:
// latest decoded car state, a small grayscale frame grab, and the track data
// sent to the physics worker.
(() => {
  const pd = (window.__pd = { car: null, track: null, workers: [] });

  // Keep the WebGL drawing buffer so we can read the canvas between frames.
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    if (type === "webgl" || type === "webgl2") {
      attrs = Object.assign({}, attrs, { preserveDrawingBuffer: true });
    }
    return getContext.call(this, type, attrs);
  };

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
    if (String(url).includes("simulation_worker")) {
      pd.workers.push(w);
      w.addEventListener("message", (e) => {
        const d = e.data;
        // messageType 99 is ours: placed track parts, injected into the worker by env.py.
        if (d && d.messageType === 99) { pd.layout = d.parts; e.stopImmediatePropagation(); return; }
        // messageType 10 = UpdateResult; the player car is the only car we create.
        if (d && d.messageType === 10 && d.carStateBuffers && d.carStateBuffers.length) {
          const last = d.carStateBuffers[d.carStateBuffers.length - 1];
          pd.carId = new DataView(last).getUint32(0, true);
          pd.car = decode(last);
        }
      });
      const post = w.postMessage.bind(w);
      // messageType 7 = PauseCar. Freezes physics while the learner updates.
      pd.pause = (paused) => { if (pd.car !== null) post({ messageType: 7, carId: pd.carId, isPaused: paused }); };
      w.postMessage = (msg, ...rest) => {
        // messageType 0 = Init (track physics parts), 3 = CreateCar (new run).
        if (msg && msg.messageType === 0) {
          pd.parts = msg.trackParts.map((t) => ({
            id: t.id, detector: t.detector, n: t.vertices.length,
            min: [0, 1, 2].map((a) => { let m = Infinity; for (let k = a; k < t.vertices.length; k += 3) m = Math.min(m, t.vertices[k]); return m; }),
            max: [0, 1, 2].map((a) => { let m = -Infinity; for (let k = a; k < t.vertices.length; k += 3) m = Math.max(m, t.vertices[k]); return m; }),
          }));
        }
        if (msg && msg.messageType === 3) {
          pd.track = msg.trackData;
          pd.car = null;
          pd.runs = (pd.runs || 0) + 1;
        }
        return post(msg, ...rest);
      };
    }
    return w;
  };
  window.Worker.prototype = RealWorker.prototype;

  // Downscaled grayscale frame as base64, w*h bytes.
  const small = document.createElement("canvas");
  pd.grab = (w, h) => {
    const src = document.getElementById("screen");
    small.width = w;
    small.height = h;
    const ctx = small.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(src, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const out = new Uint8Array(w * h);
    for (let k = 0, j = 0; k < out.length; k++, j += 4) {
      out[k] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
    }
    let s = "";
    for (let k = 0; k < out.length; k += 8192) s += String.fromCharCode.apply(null, out.subarray(k, k + 8192));
    return btoa(s);
  };
})();
