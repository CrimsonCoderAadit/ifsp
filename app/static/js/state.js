// Tiny store: one immutable state object, shallow patches, synchronous subscribers.
export function createStore(initial) {
  let state = initial;
  const subscribers = new Set();
  return {
    get: () => state,
    set(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      state = {...state, ...next};
      for (const fn of subscribers) fn(state);
    },
    subscribe(fn) {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
  };
}

export const store = createStore({
  started: false,
  connection: 'idle',          // idle | connecting | connected | reconnecting | offline
  mode: 'engineer',            // rider | engineer
  theme: 'dark',               // dark | light | hc
  target: null,                // target_bus from state messages
  partial: '',
  transcript: null,            // {text, latency_ms, confidence}
  intent: null,
  lastSpoken: null,            // {id, text, priority}
  speaking: null,              // speak message currently playing
  risk: null,                  // {text, direction, priority, id}
  scheduler: null,
  overlay: null,
  metrics: null,
  pipeline: null,
  health: null,
  recording: false,
  micLevel: 0,
  boardMode: false,
  paused: false,
  source: null,                // {kind, label}
  selectedTrack: null,
});

// Run render only when the selected slice changes (by reference or primitive value).
export function watch(selector, render) {
  let previous;
  let first = true;
  const run = state => {
    const value = selector(state);
    if (!first && value === previous) return;
    first = false;
    previous = value;
    render(value, state);
  };
  store.subscribe(run);
  run(store.get());
}

// Reallocating a canvas backing store every tick stalls frames on weak GPUs; resize only on change.
export function fitCanvas(canvas, width, height) {
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

export const storage = {
  get(key, fallback) {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* private mode or blocked storage */ }
  },
};

// Client-side events that are not server messages (mic start/stop, typed commands, sources).
const bus = new EventTarget();
export const events = {
  emit: (name, detail) => bus.dispatchEvent(new CustomEvent(name, {detail})),
  on: (name, fn) => bus.addEventListener(name, event => fn(event.detail)),
};
