// Push-to-talk capture: AudioWorklet -> 16 kHz PCM16 packets over the WebSocket.
import {enableAudio} from './audio.js';
import {events, store} from './state.js';
import {isOpen, send, sendBinary} from './ws.js';

const MAX_SECONDS = 29;
let stream = null;
let worklet = null;
let wanted = false;
let starting = false;
let finishing = false;
let timer = null;
let bytes = 0;
let startedAt = 0;

function level(buffer) {
  const samples = new Int16Array(buffer);
  if (!samples.length) return 0;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  const rms = Math.sqrt(sum / samples.length) / 32768;
  const db = 20 * Math.log10(rms || 1e-6);
  return Math.max(0, Math.min(1, (db + 60) / 60));
}

async function ensureWorklet() {
  if (worklet) return;
  store.set({partial: 'Waiting for microphone permission…'});
  const context = await enableAudio();
  stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true}, video: false});
  await context.audioWorklet.addModule('/static/pcm-worklet.js');
  const source = context.createMediaStreamSource(stream);
  worklet = new AudioWorkletNode(context, 'pcm16');
  source.connect(worklet);
  worklet.connect(context.destination);
  worklet.port.onmessage = ({data}) => {
    if (data.type === 'pcm') {
      bytes += data.buffer.byteLength;
      store.set({micLevel: level(data.buffer)});
      sendBinary(data.buffer, 'pcm');
    }
    if (data.type === 'stopped') {
      send({type: 'audio_end'});
      events.emit('mic-end', {seconds: bytes / 32000, ms: performance.now() - startedAt});
      finishing = false;
      store.set({micLevel: 0});
    }
  };
}

export function micAvailable() {
  return window.isSecureContext && !!navigator.mediaDevices?.getUserMedia && 'AudioWorkletNode' in window;
}

export async function startMic() {
  if (finishing) throw new Error('Wait for the previous recording to finish.');
  wanted = true;
  if (store.get().recording || starting) return;
  starting = true;
  try {
    if (!micAvailable()) throw new Error('Microphone needs localhost or HTTPS.');
    if (!isOpen()) throw new Error('Not connected to the assistant.');
    await ensureWorklet();
    if (!wanted) return;
    bytes = 0;
    startedAt = performance.now();
    send({type: 'audio_start'});
    worklet.port.postMessage('start');
    store.set({recording: true, partial: ''});
    events.emit('mic-start', {});
    timer = setTimeout(stopMic, MAX_SECONDS * 1000);
  } catch (error) {
    wanted = false;
    store.set({partial: ''});
    throw error;
  } finally {
    starting = false;
  }
}

export function stopMic() {
  wanted = false;
  clearTimeout(timer);
  if (!store.get().recording) return;
  finishing = true;
  worklet.port.postMessage('stop');
  store.set({recording: false});
}

export function releaseMic() {
  stopMic();
  stream?.getTracks().forEach(track => track.stop());
}
