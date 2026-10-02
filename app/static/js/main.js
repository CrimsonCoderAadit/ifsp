// Entry point: wires server messages into the store and DOM controls into actions.
import {enableAudio, onSpeechEnd, playReady, speak, stopAudio} from './audio.js';
import {initLatency, loadHealth} from './latency.js';
import {initLedBoard} from './ledboard.js';
import {micAvailable, releaseMic, startMic, stopMic} from './mic.js';
import {initOverlay, readColors as readOverlayColors} from './overlay.js';
import {initPipeline} from './pipeline.js';
import {fireRisk, initRiskPad} from './riskpad.js';
import {fillCheatSheet, initShortcuts} from './shortcuts.js';
import {events, storage, store, watch} from './state.js';
import {initTimeline, readColors as readTimelineColors} from './timeline.js';
import {cameraAvailable, initVideo, loadImage, loadSample, loadVideo, runImageApi, setBoardMode, startWebcam, togglePause} from './video.js';
import {connect, getHistory, on, onHistory, retryNow, send} from './ws.js';

const $ = id => document.getElementById(id);
const PRIORITY = ['CRITICAL', 'HIGH', 'BUS', 'INFO', 'CHAT'];
const RISK_PRIORITY = {CRITICAL: 0, HIGH: 1};
// Mirrors SchedulerConfig cooldowns: a repeat inside this window is suppressed and never spoken.
const RISK_COOLDOWN_MS = {CRITICAL: 2000, HIGH: 4000};
const PENDING_RISK_MS = 2500;
const recentRisks = new Map();
const NARROW = matchMedia('(max-width: 767px)');
let stillFile = null;
let riskTimer = null;

/* ---------- toasts ---------- */
function toast(message, {kind = 'info', action = null, timeout = 6000} = {}) {
  const item = document.createElement('div');
  item.className = 'toast';
  item.dataset.kind = kind;
  item.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  const text = document.createElement('p');
  text.textContent = message;
  item.append(text);
  if (action) {
    const button = document.createElement('button');
    button.className = 'button';
    button.textContent = action.label;
    button.addEventListener('click', () => { item.remove(); action.fn(); });
    item.append(button);
  }
  const close = document.createElement('button');
  close.className = 'button icon-button';
  close.setAttribute('aria-label', 'Dismiss notification');
  close.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  close.addEventListener('click', () => item.remove());
  item.append(close);
  $('toasts').append(item);
  if (timeout) setTimeout(() => item.remove(), timeout);
  while ($('toasts').children.length > 4) $('toasts').firstElementChild.remove();
}

const guard = fn => async (...args) => {
  try { await fn(...args); } catch (error) { toast(error.message || String(error), {kind: 'error'}); }
};

/* ---------- theme and mode ---------- */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $('theme').value = theme;
  store.set({theme});
  storage.set('theme', theme);
  readOverlayColors();
  readTimelineColors();
}

function toggleContrast() {
  const current = store.get().theme;
  if (current === 'hc') applyTheme(storage.get('themeBeforeHc', 'dark'));
  else { storage.set('themeBeforeHc', current); applyTheme('hc'); }
}

function setMode(mode) {
  document.body.dataset.mode = mode;
  for (const button of document.querySelectorAll('[data-mode-button]')) {
    button.setAttribute('aria-pressed', String(button.dataset.modeButton === mode));
  }
  store.set({mode});
}

/* ---------- commands ---------- */
function command(text, {typed = true} = {}) {
  const value = text.trim();
  if (!value) return;
  if (typed) events.emit('typed', {text: value});
  if (!send({type: 'text_command', text: value})) toast('Not connected. Commands will work once the connection returns.', {kind: 'error'});
}

/* ---------- risk takeover ---------- */
// Same wording as core.events.risk_to_phrase, so the banner does not change when audio arrives.
export function riskPhrase(level, object, rawDirection) {
  const direction = {front: 'ahead', back: 'behind'}[rawDirection] || rawDirection;
  const where = ['right', 'left', 'ahead', 'behind'].includes(direction) ? direction : 'ahead';
  const name = {car: 'Car', motorcycle: 'Motorcycle', motorbike: 'Motorcycle', bus: 'Bus', truck: 'Truck',
                bicycle: 'Bicycle', person: 'Person'}[object.toLowerCase()] || 'Obstacle';
  if (level === 'CRITICAL') return name === 'Motorcycle' ? `${name} ${where}!` : `Stop. ${name} ${where}.`;
  if (level === 'HIGH') return `${name === 'Person' ? 'Person' : name === 'Obstacle' ? 'Obstacle' : 'Vehicle'} approaching ${where}.`;
  return `${name} ${where}.`;
}

export function riskDirection(text) {
  const lower = text.toLowerCase();
  if (/\bleft\b/.test(lower)) return 'left';
  if (/\bright\b/.test(lower)) return 'right';
  if (/\b(behind|back)\b/.test(lower)) return 'behind';
  return 'ahead';
}

function announceRisk(priority, text) {
  $('riskLive').textContent = '';
  setTimeout(() => { $('riskLive').textContent = `${PRIORITY[priority]} warning: ${text}`; }, 30);
}

function onRiskSent({level, object, direction}) {
  if (!(level in RISK_PRIORITY)) return;
  const identity = `${object}:${direction}`;
  const previous = recentRisks.get(identity);
  const escalates = previous && RISK_PRIORITY[level] < RISK_PRIORITY[previous.level];
  if (previous && !escalates && Date.now() - previous.at < RISK_COOLDOWN_MS[level]) return;
  recentRisks.set(identity, {at: Date.now(), level});
  const current = store.get().risk;
  if (current && RISK_PRIORITY[level] >= current.priority && !current.pending) return;
  const text = riskPhrase(level, object, direction);
  store.set({risk: {id: null, pending: true, text, priority: RISK_PRIORITY[level],
                    direction: riskDirection(text)}});
  announceRisk(RISK_PRIORITY[level], text);
  clearTimeout(riskTimer);
  riskTimer = setTimeout(() => { if (store.get().risk?.pending) clearRisk(); }, PENDING_RISK_MS);
}

function showRisk(message) {
  const current = store.get().risk;
  const adopted = current?.pending && current.text === message.text;
  store.set({risk: {id: message.id, pending: false, text: message.text, priority: message.priority,
                    direction: riskDirection(message.text)}});
  if (!adopted) announceRisk(message.priority, message.text);
  clearTimeout(riskTimer);
  riskTimer = setTimeout(() => clearRisk(message.id), 15000);
}

function clearRisk(id) {
  if (!store.get().risk || (id && store.get().risk.id !== id)) return;
  clearTimeout(riskTimer);
  store.set({risk: null});
}

function renderRisk(risk) {
  const banner = $('riskBanner');
  banner.hidden = !risk;
  document.body.dataset.risk = risk ? risk.direction : '';
  if (!risk) return;
  banner.dataset.level = PRIORITY[risk.priority].toLowerCase();
  banner.dataset.direction = risk.direction;
  $('riskLevelLabel').textContent = PRIORITY[risk.priority];
  $('riskText').textContent = risk.text;
  $('riskArrowLabel').textContent = {left: 'Left', right: 'Right', ahead: 'Ahead', behind: 'Behind'}[risk.direction];
}

/* ---------- server messages ---------- */
function wireMessages() {
  on('ready', () => loadHealth());
  on('partial_transcript', message => store.set({partial: message.text || ''}));
  on('transcript', message => store.set({transcript: message, partial: ''}));
  on('intent', message => store.set({intent: message, target: message.state?.target_bus || null}));
  on('state', message => store.set({target: message.target_bus || null, scheduler: message.scheduler}));
  on('overlay', message => store.set({overlay: message, ...(message.state ? {target: message.state.target_bus || null} : {})}));
  on('metrics', message => store.set({metrics: message, pipeline: message.pipeline}));
  on('speak', message => {
    if (message.priority <= 1) showRisk(message);
    else if (store.get().risk && !store.get().risk.pending) clearRisk();
    speak(message).catch(error => toast(error.message, {kind: 'error'}));
  });
  on('stop_audio', () => stopAudio());
  on('error', message => toast(message.message, {kind: 'error'}));
  on('offline', () => toast('Lost connection to the assistant server.', {kind: 'error', timeout: 0,
                                                                         action: {label: 'Retry', fn: retryNow}}));
  onSpeechEnd(message => clearRisk(message.id));
  events.on('risk-sent', onRiskSent);
}

/* ---------- rider panels ---------- */
function wireRider() {
  watch(state => state.connection, connection => {
    const pill = $('connection');
    pill.dataset.state = connection;
    pill.querySelector('.pill-text').textContent = {idle: 'Not started', connecting: 'Connecting…', connected: 'Connected',
                                                    reconnecting: 'Reconnecting…', offline: 'Offline'}[connection];
  });
  watch(state => state.lastSpoken, spoken => {
    const label = !spoken ? '' : spoken.priority == null ? 'SYSTEM' : PRIORITY[spoken.priority];
    $('lastSpoken').textContent = spoken?.text || '—';
    $('lastSpokenPriority').textContent = label;
    $('lastSpoken').dataset.priority = label.toLowerCase();
  });
  watch(state => state.partial, partial => { $('partial').textContent = partial || (store.get().recording ? 'Listening…' : ''); });
  watch(state => state.transcript, transcript => {
    $('transcript').textContent = transcript ? `Heard: “${transcript.text || '(nothing)'}”` : '';
  });
  watch(state => state.intent, intent => {
    $('intentLine').textContent = intent ? `${intent.intent}${intent.target ? ` ${intent.target}` : ''}${intent.acknowledgement ? ` — ${intent.acknowledgement}` : ''}` : '';
  });
  watch(state => state.recording, recording => {
    const button = $('ptt');
    button.setAttribute('aria-pressed', String(recording));
    button.querySelector('.ptt-label').textContent = recording ? 'Listening… release to send' : 'Hold to talk';
    if (recording) $('partial').textContent = 'Listening…';
  });
  watch(state => state.micLevel, value => {
    $('micMeter').style.setProperty('--level', value.toFixed(3));
  });
  watch(state => state.risk, renderRisk);

  const ptt = $('ptt');
  const talkStart = guard(startMic);
  ptt.addEventListener('pointerdown', event => {
    event.preventDefault();
    ptt.setPointerCapture(event.pointerId);
    talkStart();
  });
  ptt.addEventListener('pointerup', stopMic);
  ptt.addEventListener('pointercancel', stopMic);
  ptt.addEventListener('click', event => {
    if (event.detail === 0) store.get().recording ? stopMic() : talkStart();
  });
  for (const button of document.querySelectorAll('[data-command]')) {
    button.addEventListener('click', guard(async () => { await enableAudio(); command(button.dataset.command); }));
  }
  $('commandForm').addEventListener('submit', guard(async event => {
    event.preventDefault();
    await enableAudio();
    command($('commandInput').value);
  }));
  if (!micAvailable()) {
    ptt.disabled = true;
    ptt.querySelector('.ptt-label').textContent = 'Mic needs HTTPS';
  }
}

/* ---------- engineer panels ---------- */
function wireSources() {
  const panels = document.querySelectorAll('[data-source-panel]');
  for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener('change', () => {
      for (const panel of panels) panel.hidden = panel.dataset.sourcePanel !== radio.value;
    });
  }
  $('webcamStart').addEventListener('click', guard(startWebcam));
  $('riderCamera').addEventListener('click', guard(startWebcam));
  if (!cameraAvailable()) {
    for (const button of [$('webcamStart'), $('riderCamera')]) {
      button.disabled = true;
      button.textContent = 'Camera needs HTTPS';
    }
  }
  $('videoFile').addEventListener('change', guard(async event => {
    const file = event.target.files[0];
    if (file) await loadVideo(URL.createObjectURL(file), file.name, {revoke: true});
  }));
  $('imageFile').addEventListener('change', event => {
    stillFile = event.target.files[0] || null;
    if (stillFile) loadImage(URL.createObjectURL(stillFile), stillFile.name, {revoke: true});
    $('imageApi').disabled = !stillFile;
  });
  $('imageApi').addEventListener('click', guard(async () => {
    if (!stillFile) return;
    $('imageApi').disabled = true;
    try {
      const result = await runImageApi(stillFile, $('commandInput').value);
      const link = $('annotated');
      link.href = `data:image/jpeg;base64,${result.annotated_jpeg_b64}`;
      link.hidden = false;
      toast(`Image API: ${result.overlay.status || 'done'}${result.overlay.matched_route ? ` · ${result.overlay.matched_route}` : ''}`);
    } finally {
      $('imageApi').disabled = false;
    }
  }));
  $('boardMode').addEventListener('change', event => setBoardMode(event.target.checked));
  watch(state => state.boardMode, board => { $('boardMode').checked = board; });
  $('pauseFrames').addEventListener('click', togglePause);
  watch(state => state.paused, paused => {
    $('pauseFrames').setAttribute('aria-pressed', String(paused));
    $('pauseFrames').querySelector('span').textContent = paused ? 'Resume frames' : 'Pause frames';
  });
  watch(state => state.source, source => { $('sourceLabel').textContent = source ? `Source: ${source.label}` : 'No source selected'; });
  loadSamples();
}

async function loadSamples() {
  const gallery = $('samples');
  try {
    const items = await (await fetch('/static/samples/index.json')).json();
    for (const item of items) {
      const button = document.createElement('button');
      button.className = 'sample';
      button.type = 'button';
      const preview = document.createElement(item.kind === 'video' ? 'video' : 'img');
      preview.src = `/static/samples/${item.file}`;
      preview.setAttribute('aria-hidden', 'true');
      if (item.kind === 'video') { preview.muted = true; preview.preload = 'metadata'; } else { preview.alt = ''; }
      const label = document.createElement('span');
      label.textContent = `${item.kind === 'video' ? 'Clip: ' : ''}${item.label}`;
      button.append(preview, label);
      button.addEventListener('click', guard(async () => { await enableAudio(); await loadSample(item); }));
      gallery.append(button);
    }
  } catch {
    gallery.textContent = 'Samples unavailable. Run scripts/prepare_web_assets.py.';
  }
}

function renderInspector() {
  if (!$('ocrDrawer').open) return;
  const overlay = store.get().overlay;
  const select = $('trackSelect');
  const boxes = overlay?.boxes || [];
  const tracks = boxes.map(box => String(box.track_id));
  if ([...select.options].map(option => option.value).join() !== tracks.join()) {
    const previous = select.value;
    select.replaceChildren(...tracks.map(id => new Option(`Track ${id}`, id)));
    if (tracks.includes(previous)) select.value = previous;
  }
  const selected = store.get().selectedTrack;
  if (selected != null && tracks.includes(String(selected))) select.value = String(selected);
  const box = boxes.find(item => String(item.track_id) === select.value);
  const body = $('ocrRows');
  const attempts = box?.ocr_attempts || [];
  const winner = attempts.findIndex(attempt => attempt.route);
  body.replaceChildren(...attempts.map((attempt, index) => {
    const row = document.createElement('tr');
    if (index === winner) row.className = 'winner';
    const cells = [index === winner ? `✓ ${attempt.crop}` : attempt.crop, attempt.variant,
                   attempt.latency_ms.toFixed(1), attempt.text.join(' ') || '—', attempt.route || '—', attempt.method];
    for (const value of cells) row.insertCell().textContent = value;
    return row;
  }));
  $('ocrEmpty').hidden = attempts.length > 0;
}

function renderSchedulerLog() {
  if (!$('logDrawer').open) return;
  const log = store.get().scheduler?.log || [];
  const time = seconds => new Date(seconds * 1000).toLocaleTimeString();
  $('schedulerLog').replaceChildren(...[...log].reverse().map(row => {
    const item = document.createElement('li');
    item.dataset.decision = row.decision;
    const when = row.count > 1 ? `${time(row.first_timestamp)}–${time(row.last_timestamp)}` : time(row.timestamp);
    const priority = row.priority === '' ? '' : ` [${PRIORITY[row.priority]}]`;
    item.textContent = `${when}  ${row.decision}${row.count > 1 ? ` ×${row.count}` : ''}${priority}  ${row.text || row.detail || ''}`;
    return item;
  }));
}

function renderWsLog() {
  if (!$('logDrawer').open || $('wsPause').checked) return;
  const filter = $('wsFilter').value;
  const history = getHistory();
  const types = [...new Set(history.map(entry => entry.type))].sort();
  const options = ['', ...types];
  if ([...$('wsFilter').options].map(option => option.value).join() !== options.join()) {
    $('wsFilter').replaceChildren(...options.map(type => new Option(type || 'All types', type)));
    $('wsFilter').value = options.includes(filter) ? filter : '';
  }
  const shown = history.filter(entry => !filter || entry.type === filter);
  $('wsCount').textContent = `${shown.length} of ${history.length} messages`;
  $('wsLog').textContent = shown.slice(-200).reverse().map(entry =>
    `${new Date(entry.at).toLocaleTimeString()} ${entry.direction === 'in' ? '←' : '→'} ${JSON.stringify(entry.message)}`).join('\n');
}

function wireDrawers() {
  watch(state => state.overlay, renderInspector);
  watch(state => state.selectedTrack, renderInspector);
  $('ocrDrawer').addEventListener('toggle', renderInspector);
  $('trackSelect').addEventListener('change', event => store.set({selectedTrack: Number(event.target.value)}));
  watch(state => state.scheduler, renderSchedulerLog);
  $('logDrawer').addEventListener('toggle', () => { renderSchedulerLog(); renderWsLog(); });
  let pending = false;
  onHistory(() => {
    if (pending) return;
    pending = true;
    setTimeout(() => { pending = false; renderWsLog(); }, 500);
  });
  $('wsFilter').addEventListener('change', renderWsLog);
  $('wsPause').addEventListener('change', renderWsLog);
}

/* ---------- start ---------- */
async function start() {
  const button = $('startButton');
  button.disabled = true;
  try {
    await enableAudio();
    connect();
    store.set({started: true});
    $('startOverlay').hidden = true;
    document.body.classList.remove('locked');
    for (const element of document.querySelectorAll('[data-behind-start]')) {
      element.inert = false;
      element.removeAttribute('data-behind-start');
    }
    $('ptt').focus();
    store.set({lastSpoken: {id: 'ready', text: 'Ready.', priority: null}});
    await playReady().catch(() => toast('Could not play the “Ready.” clip.', {kind: 'error'}));
  } catch (error) {
    button.disabled = false;
    toast(error.message, {kind: 'error'});
  }
}

function init() {
  for (const element of document.body.children) {
    if (['startOverlay', 'toasts'].includes(element.id) || element.matches('[aria-live], svg, dialog')) continue;
    element.inert = true;
    element.dataset.behindStart = '';
  }
  applyTheme(storage.get('theme', 'dark'));
  setMode(NARROW.matches ? 'rider' : 'engineer');
  if (!window.isSecureContext) $('secureNote').hidden = false;
  for (const button of document.querySelectorAll('[data-mode-button]')) {
    button.addEventListener('click', () => setMode(button.dataset.modeButton));
  }
  $('theme').addEventListener('change', event => applyTheme(event.target.value));
  $('helpButton').addEventListener('click', () => $('helpDialog').showModal());
  $('helpClose').addEventListener('click', () => $('helpDialog').close());
  fillCheatSheet($('shortcutTable'));
  $('startButton').addEventListener('click', start);
  $('startOverlay').addEventListener('keydown', event => {
    if (event.key !== 'Tab') return;
    event.preventDefault();
    $('startButton').focus();
  });
  initVideo();
  initOverlay();
  initPipeline();
  initTimeline();
  initLatency();
  initRiskPad();
  initLedBoard();
  wireMessages();
  wireRider();
  wireSources();
  wireDrawers();
  initShortcuts({
    talkStart: guard(startMic),
    talkEnd: stopMic,
    repeat: () => command('repeat'),
    stop: () => command('stop'),
    ahead: () => command('what is ahead'),
    cancel: () => command('cancel bus'),
    mode: () => setMode(store.get().mode === 'rider' ? 'engineer' : 'rider'),
    contrast: toggleContrast,
    riskMedium: () => fireRisk('MEDIUM'),
    riskHigh: () => fireRisk('HIGH'),
    riskCritical: () => fireRisk('CRITICAL'),
    pause: togglePause,
    help: () => $('helpDialog').showModal(),
  }, {blocked: () => !store.get().started || $('helpDialog').open});
  window.addEventListener('beforeunload', releaseMic);
  $('startButton').focus();
}

init();
