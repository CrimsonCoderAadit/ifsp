// WebSocket client with exponential-backoff reconnect and a raw message history.
import {store} from './state.js';

const handlers = new Map();
const history = [];
const historyListeners = new Set();
const BACKOFF_MS = [500, 1000, 2000, 4000, 8000];
const MAX_RETRIES = 6;
let socket = null;
let retries = 0;
let retryTimer = null;
let wanted = false;

export function on(type, fn) {
  if (!handlers.has(type)) handlers.set(type, new Set());
  handlers.get(type).add(fn);
}

function emit(type, message) {
  for (const fn of handlers.get(type) || []) fn(message);
  for (const fn of handlers.get('*') || []) fn(message);
}

function record(direction, message) {
  const summary = {...message};
  if (summary.audio_b64_wav) summary.audio_b64_wav = `<${Math.round(summary.audio_b64_wav.length * 3 / 4)} bytes WAV>`;
  if (summary.jpeg_b64) summary.jpeg_b64 = `<${summary.jpeg_b64.length} chars JPEG>`;
  history.push({at: Date.now(), direction, type: message.type || 'binary', message: summary});
  if (history.length > 200) history.shift();
  for (const fn of historyListeners) fn(history);
}

export function onHistory(fn) { historyListeners.add(fn); }
export function getHistory() { return history; }

export function isOpen() { return socket?.readyState === WebSocket.OPEN; }

export function send(message) {
  if (!isOpen()) return false;
  socket.send(JSON.stringify(message));
  record('out', message);
  return true;
}

export function sendBinary(data, label = 'binary') {
  if (!isOpen()) return false;
  socket.send(data);
  if (label !== 'pcm') record('out', {type: label, bytes: data.byteLength});
  return true;
}

export function connect() {
  wanted = true;
  clearTimeout(retryTimer);
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  store.set({connection: retries ? 'reconnecting' : 'connecting'});
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  const ws = new WebSocket(url);
  socket = ws;
  ws.onopen = () => {
    retries = 0;
    store.set({connection: 'connected'});
    emit('open', {type: 'open'});
  };
  ws.onmessage = ({data}) => {
    let message;
    try { message = JSON.parse(data); } catch { return; }
    record('in', message);
    emit(message.type, message);
  };
  ws.onclose = () => {
    if (socket !== ws) return;
    socket = null;
    emit('close', {type: 'close'});
    if (!wanted) { store.set({connection: 'offline'}); return; }
    if (retries >= MAX_RETRIES) {
      store.set({connection: 'offline'});
      emit('offline', {type: 'offline'});
      return;
    }
    store.set({connection: 'reconnecting'});
    retryTimer = setTimeout(connect, BACKOFF_MS[Math.min(retries, BACKOFF_MS.length - 1)]);
    retries += 1;
  };
}

export function retryNow() {
  retries = 0;
  connect();
}
