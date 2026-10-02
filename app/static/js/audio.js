// Plays server-synthesised WAVs. The Python scheduler owns priority; this only obeys it.
import {store} from './state.js';
import {send} from './ws.js';

let context = null;
let current = null;
let generation = 0;
const endListeners = new Set();

export function onSpeechEnd(fn) { endListeners.add(fn); }

export async function enableAudio() {
  context ||= new AudioContext();
  await context.resume();
  return context;
}

export function audioContext() { return context; }

function finished(entry, interrupted) {
  send({type: 'playback_done', id: entry.id});
  if (store.get().speaking?.id === entry.id) store.set({speaking: null});
  for (const fn of endListeners) fn(entry.message, interrupted);
}

export function stopAudio() {
  generation += 1;
  if (!current) return;
  const old = current;
  current = null;
  try { old.source.stop(); } catch { /* already stopped */ }
  finished(old, true);
}

function decode(base64) {
  const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
  return context.decodeAudioData(bytes.buffer);
}

export async function speak(message) {
  if (message.interrupt) stopAudio();
  const mine = generation;
  try {
    if (!context || context.state !== 'running') throw new Error('Audio is not enabled yet.');
    const buffer = await decode(message.audio_b64_wav);
    if (mine !== generation) { send({type: 'playback_done', id: message.id}); return; }
    if (current) stopAudio();
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const entry = {source, id: message.id, message};
    current = entry;
    source.onended = () => {
      if (current !== entry) return;
      current = null;
      finished(entry, false);
    };
    source.start();
    store.set({speaking: {...message, duration: buffer.duration}, lastSpoken: {id: message.id, text: message.text, priority: message.priority}});
  } catch (error) {
    send({type: 'playback_done', id: message.id});
    throw error;
  }
}

export async function playReady() {
  const response = await fetch('/static/audio/ready.wav');
  const buffer = await context.decodeAudioData(await response.arrayBuffer());
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
}
