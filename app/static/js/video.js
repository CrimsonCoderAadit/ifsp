// Camera / file / still / sample sources and the 5 fps JPEG frame loop (waits for frame_ack).
import {enableAudio} from './audio.js';
import {events, store} from './state.js';
import {isOpen, on, send, sendBinary} from './ws.js';

const FRAME_INTERVAL_MS = 200;
const MAX_WIDTH = 1280;
const capture = document.createElement('canvas');
let video;
let still;
let mediaStream = null;
let objectURL = null;
let frameBusy = false;
let frameId = 0;
let media = null;

export function currentMedia() { return media; }

export function cameraAvailable() {
  return window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;
}

function clearSource() {
  mediaStream?.getTracks().forEach(track => track.stop());
  mediaStream = null;
  video.pause();
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
  still.removeAttribute('src');
  media = null;
  store.set({overlay: null, selectedTrack: null});
}

function show(element) {
  video.hidden = element !== video;
  still.hidden = element !== still;
  media = element;
}

export function setBoardMode(board) {
  store.set({boardMode: board});
  send({type: 'source_mode', mode: board ? 'board' : 'bus'});
}

export async function startWebcam() {
  if (!cameraAvailable()) throw new Error('Camera needs localhost or HTTPS.');
  await enableAudio();
  clearSource();
  mediaStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment'}, audio: false});
  video.srcObject = mediaStream;
  video.loop = false;
  show(video);
  await video.play();
  store.set({source: {kind: 'webcam', label: 'Webcam'}, paused: false});
}

export async function loadVideo(url, label, {loop = false, revoke = false} = {}) {
  await enableAudio();
  clearSource();
  if (revoke) objectURL = url;
  video.src = url;
  video.loop = loop;
  show(video);
  await video.play();
  store.set({source: {kind: 'video', label}, paused: false});
}

export function loadImage(url, label, {revoke = false} = {}) {
  clearSource();
  if (revoke) objectURL = url;
  still.src = url;
  still.alt = `Still image source: ${label}`;
  show(still);
  store.set({source: {kind: 'image', label}, paused: false});
}

export async function loadSample(item) {
  setBoardMode(!!item.board);
  const url = `/static/samples/${item.file}`;
  if (item.kind === 'video') await loadVideo(url, item.label, {loop: true});
  else loadImage(url, item.label);
  events.emit('source', item);
}

export function togglePause() {
  store.set({paused: !store.get().paused});
}

function sendFrame() {
  const {paused} = store.get();
  if (paused || frameBusy || !media || !isOpen()) return;
  const width = media.videoWidth || media.naturalWidth;
  const height = media.videoHeight || media.naturalHeight;
  if (!width || !height) return;
  if (media === video && (video.paused || video.ended)) return;
  frameBusy = true;
  const id = ++frameId;
  const scale = Math.min(1, MAX_WIDTH / width);
  capture.width = Math.round(width * scale);
  capture.height = Math.round(height * scale);
  capture.getContext('2d').drawImage(media, 0, 0, capture.width, capture.height);
  capture.toBlob(async blob => {
    if (!blob || !isOpen()) { frameBusy = false; return; }
    const bytes = await blob.arrayBuffer();
    if (!send({type: 'frame', id}) || !sendBinary(bytes, 'frame_jpeg')) frameBusy = false;
  }, 'image/jpeg', 0.7);
}

export async function runImageApi(file, command) {
  const data = new FormData();
  data.append('file', file);
  data.append('command', command);
  data.append('board', String(store.get().boardMode));
  const response = await fetch('/api/image', {method: 'POST', body: data});
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || 'Image API failed');
  store.set({overlay: result.overlay});
  return result;
}

export function initVideo() {
  video = document.getElementById('video');
  still = document.getElementById('still');
  on('frame_ack', () => { frameBusy = false; });
  on('close', () => { frameBusy = false; });
  on('open', () => send({type: 'source_mode', mode: store.get().boardMode ? 'board' : 'bus'}));
  setInterval(sendFrame, FRAME_INTERVAL_MS);
}
