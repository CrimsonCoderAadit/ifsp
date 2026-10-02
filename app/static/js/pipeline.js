// Pipeline tracer: Mic → STT → Intent → Detect → OCR → Normalise → GTFS match → Scheduler → TTS.
import {events, store} from './state.js';
import {on} from './ws.js';

const NODES = [
  ['mic', 'Mic'], ['stt', 'STT'], ['intent', 'Intent'], ['detect', 'Detect'], ['ocr', 'OCR'],
  ['normalise', 'Normalise'], ['gtfs', 'GTFS match'], ['scheduler', 'Scheduler'], ['tts', 'TTS'],
];
const PRIORITY = ['CRITICAL', 'HIGH', 'BUS', 'INFO', 'CHAT'];
const STALE_MS = 8000;
const elements = {};
let lastFrame = null;
let lastDecision = '';

const formatMs = ms => (ms == null || Number.isNaN(ms) ? '' : `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`);

function setNode(name, {value, ms = null, state = 'lit', detail = ''}) {
  const node = elements[name];
  if (!node) return;
  node.dataset.state = state;
  node.dataset.updated = String(Date.now());
  node.querySelector('.node-value').textContent = value;
  node.querySelector('.node-ms').textContent = [formatMs(ms), detail].filter(Boolean).join(' · ');
  if (state === 'lit' && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    node.classList.remove('pulse');
    void node.offsetWidth;
    node.classList.add('pulse');
  }
}

function winningAttempt(box) {
  const attempts = box?.ocr_attempts || [];
  return attempts.find(attempt => attempt.route) || attempts[attempts.length - 1] || null;
}

function onOverlay(overlay) {
  if (!overlay.frame_id || overlay.frame_id === lastFrame) return;
  lastFrame = overlay.frame_id;
  const detector = store.get().metrics?.latency_ms?.detector?.last;
  const count = overlay.boxes?.length || 0;
  const board = store.get().boardMode;
  setNode('detect', {value: `${count} ${count === 1 ? 'box' : 'boxes'}`, ms: board ? null : detector,
                     detail: board ? 'YOLO bypassed (board mode)' : ''});
  if (!count) {
    for (const name of ['ocr', 'normalise', 'gtfs']) setNode(name, {value: '—', state: 'idle'});
    return;
  }
  const selected = store.get().selectedTrack;
  const box = overlay.boxes.find(item => item.track_id === selected) || overlay.boxes[0];
  const attempt = winningAttempt(box);
  if (!attempt) {
    setNode('ocr', {value: 'waiting for read', state: 'pending'});
    return;
  }
  const raw = attempt.text.join(' ') || '(no text)';
  setNode('ocr', {value: `“${raw}”`, ms: attempt.latency_ms,
                  detail: `${box.ocr_attempts.length} ${box.ocr_attempts.length === 1 ? 'try' : 'tries'} · ${attempt.crop}`,
                  state: attempt.text.length ? 'lit' : 'fail'});
  const candidate = attempt.candidates?.[0];
  setNode('normalise', {value: candidate || '—', state: candidate ? 'lit' : 'fail'});
  if (attempt.route) {
    const confirmed = box.confirmed_route === attempt.route;
    setNode('gtfs', {value: `${attempt.route} · ${attempt.method}`, detail: confirmed ? 'confirmed' : 'voting', state: 'lit'});
  } else {
    setNode('gtfs', {value: candidate ? `✕ ${candidate} not in GTFS` : '✕ no route', state: 'fail'});
  }
}

function onState(message) {
  const log = message.scheduler?.log || [];
  const row = [...log].reverse().find(entry => entry.decision !== 'cleared_cooldowns');
  if (!row) return;
  const key = `${row.decision}|${row.item_id}|${row.count}`;
  if (key === lastDecision) return;
  lastDecision = key;
  const priority = row.priority === '' ? '' : PRIORITY[row.priority] || row.priority;
  const count = row.count > 1 ? ` ×${row.count}` : '';
  setNode('scheduler', {value: `${row.decision.replace(/_/g, ' ')}${count}${priority ? ` · ${priority}` : ''}`,
                        ms: row.decision === 'played' ? Number(row.age_ms) : null,
                        detail: row.decision === 'played' ? 'event → audio' : '',
                        state: /pre-empted|dropped|error/.test(row.decision) ? 'fail'
                          : /suppressed|deduped|flushed/.test(row.decision) ? 'done' : 'lit'});
}

function staleSweep() {
  const now = Date.now();
  for (const node of Object.values(elements)) {
    if (node.dataset.state === 'lit' && now - Number(node.dataset.updated || 0) > STALE_MS) node.dataset.state = 'done';
  }
}

export function initPipeline() {
  const list = document.getElementById('tracer');
  NODES.forEach(([name, label], index) => {
    const item = document.createElement('li');
    item.className = 'node';
    item.dataset.node = name;
    item.dataset.state = 'idle';
    item.innerHTML = `<span class="node-step">${index + 1}</span><span class="node-name"></span>
      <span class="node-value">—</span><span class="node-ms"></span>`;
    item.querySelector('.node-name').textContent = label;
    list.append(item);
    elements[name] = item;
  });
  events.on('mic-start', () => {
    setNode('mic', {value: 'listening…', state: 'pending'});
    setNode('stt', {value: 'waiting', state: 'pending'});
    setNode('intent', {value: 'waiting', state: 'pending'});
  });
  events.on('mic-end', ({seconds}) => setNode('mic', {value: `${seconds.toFixed(1)} s audio`}));
  events.on('typed', ({text}) => {
    setNode('mic', {value: 'typed', state: 'done'});
    setNode('stt', {value: `“${text}” (typed)`, state: 'done'});
  });
  on('transcript', message => setNode('stt', {value: `“${message.text || '…'}”`, ms: message.latency_ms,
                                             detail: `conf ${(message.confidence ?? 0).toFixed(2)}`,
                                             state: message.text ? 'lit' : 'fail'}));
  on('intent', message => setNode('intent', {value: `${message.intent}${message.target ? ` (${message.target})` : ''}`,
                                            state: message.intent === 'UNKNOWN' ? 'fail' : 'lit'}));
  on('overlay', onOverlay);
  on('state', onState);
  on('speak', message => setNode('tts', {value: `“${message.text}”`, ms: store.get().metrics?.latency_ms?.tts?.last,
                                        detail: `synth${message.interrupt ? ' · interrupting' : ''}`}));
  setInterval(staleSweep, 1000);
}
