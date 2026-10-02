// Camera-stage overlay: bus boxes, route chips, OCR line boxes (box_frame) and vote meters.
import {fitCanvas, store, watch} from './state.js';
import {currentMedia} from './video.js';
import {on} from './ws.js';

let canvas;
let stage;
let colors = {};
let hitBoxes = [];
let acked = 0;

export function readColors() {
  const style = getComputedStyle(document.documentElement);
  const token = name => style.getPropertyValue(name).trim();
  colors = {amber: token('--ov-amber'), green: token('--ov-green'), grey: token('--ov-grey'),
            onChip: token('--ov-on-chip'), ocr: token('--ov-ocr'), focus: token('--focus'),
            font: token('--font-ui') || 'sans-serif', line: Number(token('--overlay-line')) || 3};
}

// Older servers send no overlay.status; derive it from the boxes they do send.
export function overlayStatus(overlay) {
  if (overlay.status) return overlay.status;
  const boxes = overlay.boxes || [];
  if (!boxes.length) return 'no_bus_in_view';
  if (boxes.some(box => box.confirmed_route)) return 'confirmed';
  return 'reading_board';
}

export function statusText(overlay, {source = null, framesAcked = 0} = {}) {
  if (!overlay) {
    if (!source) return {kind: 'idle', text: 'No frames yet. Pick a source.'};
    return {kind: 'read', text: framesAcked ? 'Processing frames…' : `Waiting for the first frame from ${source.label}…`};
  }
  const seen = overlay.status_text;
  switch (overlayStatus(overlay)) {
    case 'no_bus_in_view': return {kind: 'idle', text: 'No bus in view'};
    case 'reading_board': return {kind: 'read', text: 'Reading board…'};
    case 'route_unreadable': return {kind: 'warn', text: seen ? `Route unreadable: saw ‘${seen}’` : 'Route unreadable'};
    case 'route_not_in_gtfs': return {kind: 'warn', text: `${seen || 'Route'} — not in GTFS`};
    case 'confirmed': {
      const target = overlay.state?.target_bus?.route;
      const route = overlay.matched_route || overlay.boxes?.find(box => box.confirmed_route)?.confirmed_route;
      return {kind: route && route === target ? 'ok' : 'read',
              text: `${route} confirmed${route && route === target ? ' — your bus' : ''}`};
    }
    default: return {kind: 'idle', text: overlay.status};
  }
}

function contentRect(width, height) {
  const box = stage.getBoundingClientRect();
  const scale = Math.min(box.width / width, box.height / height);
  return {x: (box.width - width * scale) / 2, y: (box.height - height * scale) / 2,
          width: width * scale, height: height * scale, stageWidth: box.width, stageHeight: box.height};
}

function chipColor(box, target) {
  const route = box.confirmed_route || box.matched_route;
  if (box.confirmed_route && box.confirmed_route === target) return {fill: colors.green, label: `${route} ✓`, kind: 'target'};
  if (route && target && route !== target) return {fill: colors.grey, label: route, kind: 'other'};
  return {fill: colors.amber, label: route || '?', kind: 'read'};
}

export function voteCount(box, votes) {
  const route = box.confirmed_route || box.matched_route;
  const buffer = votes?.[String(box.track_id)] || [];
  return {agree: route ? buffer.filter(value => value === route).length : 0, total: 3};
}

function draw() {
  const overlay = store.get().overlay;
  const media = currentMedia();
  const box = stage.getBoundingClientRect();
  const ctx = fitCanvas(canvas, box.width, box.height);
  hitBoxes = [];
  if (!overlay?.boxes?.length || !media) return;
  const naturalWidth = media.videoWidth || media.naturalWidth;
  const naturalHeight = media.videoHeight || media.naturalHeight;
  if (!naturalWidth || !naturalHeight) return;
  const rect = contentRect(naturalWidth, naturalHeight);
  const scale = rect.width / (overlay.frame_width || naturalWidth);
  const target = overlay.state?.target_bus?.route || store.get().target?.route;
  const selected = store.get().selectedTrack;
  const fontSize = Math.max(14, Math.min(22, rect.width / 40));
  for (const item of overlay.boxes) {
    const [x1, y1, x2, y2] = item.bbox.map(value => value * scale);
    const left = rect.x + x1;
    const top = rect.y + y1;
    const chip = chipColor(item, target);
    ctx.lineWidth = colors.line + (item.track_id === selected ? 2 : 0);
    ctx.strokeStyle = chip.fill;
    ctx.setLineDash(item.track_id === selected ? [10, 4] : []);
    ctx.strokeRect(left, top, x2 - x1, y2 - y1);
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = colors.ocr;
    for (const line of item.ocr_lines || []) {
      if (!line.box_frame) continue;
      ctx.beginPath();
      line.box_frame.forEach(([x, y], index) => ctx[index ? 'lineTo' : 'moveTo'](rect.x + x * scale, rect.y + y * scale));
      ctx.closePath();
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.font = `700 ${fontSize}px ${colors.font}`;
    const votes = voteCount(item, overlay.state?.votes);
    const label = `${chip.label}  ·  votes ${votes.agree}/${votes.total}`;
    const chipWidth = ctx.measureText(label).width + 16;
    const chipHeight = fontSize + 12;
    const chipX = Math.min(Math.max(rect.x, left), rect.x + rect.width - chipWidth);
    const above = top - chipHeight - 2;
    const chipY = above >= rect.y ? above : top + 2;
    ctx.fillStyle = chip.fill;
    ctx.fillRect(chipX, chipY, chipWidth, chipHeight);
    ctx.fillStyle = colors.onChip;
    ctx.textBaseline = 'middle';
    ctx.fillText(label, chipX + 8, chipY + chipHeight / 2 + 1);
    const pip = 8;
    for (let index = 0; index < votes.total; index += 1) {
      ctx.fillStyle = index < votes.agree ? chip.fill : 'rgba(0,0,0,.55)';
      ctx.fillRect(chipX + 8 + index * (pip + 4), chipY + chipHeight + 4, pip, pip);
      ctx.strokeStyle = chip.fill;
      ctx.lineWidth = 1;
      ctx.strokeRect(chipX + 8 + index * (pip + 4), chipY + chipHeight + 4, pip, pip);
    }
    hitBoxes.push({track: item.track_id, left, top, right: rect.x + x2, bottom: rect.y + y2});
  }
}

export function redraw() { if (canvas) draw(); }

export function initOverlay() {
  canvas = document.getElementById('stageCanvas');
  stage = document.getElementById('stage');
  readColors();
  canvas.addEventListener('click', event => {
    const bounds = canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const hit = hitBoxes.find(box => x >= box.left && x <= box.right && y >= box.top && y <= box.bottom);
    if (hit) store.set({selectedTrack: hit.track});
  });
  new ResizeObserver(draw).observe(stage);
  document.getElementById('video').addEventListener('loadedmetadata', draw);
  document.getElementById('still').addEventListener('load', draw);
  watch(state => state.overlay, draw);
  watch(state => state.selectedTrack, draw);
  watch(state => state.theme, () => { readColors(); draw(); });
  const strip = document.getElementById('statusStrip');
  const renderStrip = () => {
    const {overlay, source} = store.get();
    const {kind, text} = statusText(overlay, {source, framesAcked: acked});
    if (strip.dataset.kind !== kind) strip.dataset.kind = kind;
    const label = strip.querySelector('.status-text');
    if (label.textContent !== text) label.textContent = text;
  };
  watch(state => state.overlay, renderStrip);
  watch(state => state.source, () => { acked = 0; renderStrip(); });
  on('frame_ack', message => { if (!message.dropped) acked += 1; renderStrip(); });
}
