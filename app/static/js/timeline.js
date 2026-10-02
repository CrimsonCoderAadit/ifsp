// Speech timeline: last 30 s of scheduler decisions as one swimlane per priority.
import {fitCanvas, store} from './state.js';
import {on} from './ws.js';

const WINDOW_S = 30;
const LANES = ['CRITICAL', 'HIGH', 'BUS', 'INFO', 'CHAT'];
const ENDINGS = new Set(['completed', 'pre-empted', 'stopped']);
const rows = new Map();
const speakReceived = new Map();
let skew = 0;
let canvas;
let summary;
let colors = {};

function key(row) { return `${row.decision}|${row.text}|${row.first_timestamp ?? row.timestamp}`; }

function ingest(message) {
  for (const row of message.scheduler?.log || []) {
    rows.set(key(row), row);
    if (row.decision === 'played' && speakReceived.has(row.item_id)) {
      skew = speakReceived.get(row.item_id) - row.timestamp;
      speakReceived.delete(row.item_id);
    }
  }
  const cutoff = Date.now() / 1000 - skew - WINDOW_S * 2;
  for (const [id, row] of rows) if ((row.last_timestamp ?? row.timestamp) < cutoff) rows.delete(id);
}

export function readColors() {
  const style = getComputedStyle(document.documentElement);
  const token = name => style.getPropertyValue(name).trim();
  colors = {lanes: [token('--risk-critical'), token('--red'), token('--amber'), token('--grey'), token('--text-muted')],
            text: token('--text'), muted: token('--text-muted'), grid: token('--border'),
            cut: token('--red'), font: token('--font-ui') || 'sans-serif', mono: token('--font-mono') || 'monospace',
            surface: token('--surface')};
}

export function buildSegments(all, now) {
  const ordered = [...all].sort((a, b) => a.timestamp - b.timestamp);
  const segments = [];
  const open = new Map();
  const suppressed = new Map();
  const requeues = [];
  for (const row of ordered) {
    const lane = Number(row.priority);
    if (row.decision === 'played') {
      const previous = segments.filter(s => s.id === row.item_id && s.end != null && s.endKind === 'pre-empted').pop();
      const segment = {id: row.item_id, lane, start: row.timestamp, end: null, endKind: null, text: row.text};
      if (previous) requeues.push({from: previous, to: segment});
      segments.push(segment);
      open.set(row.item_id, segment);
    } else if (ENDINGS.has(row.decision) && open.has(row.item_id)) {
      const segment = open.get(row.item_id);
      segment.end = row.timestamp;
      segment.endKind = row.decision;
      open.delete(row.item_id);
    } else if (row.decision === 'suppressed_cooldown' && Number.isFinite(lane)) {
      const entry = suppressed.get(lane) || {lane, count: 0, ticks: []};
      entry.count += row.count || 1;
      entry.ticks.push(row.first_timestamp ?? row.timestamp, row.last_timestamp ?? row.timestamp);
      suppressed.set(lane, entry);
    }
  }
  for (const segment of segments) if (segment.end == null) segment.end = now;
  return {segments, requeues, suppressed: [...suppressed.values()]};
}

function hatch(ctx, x, y, width, height, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (let offset = -height; offset < width; offset += 5) {
    ctx.beginPath();
    ctx.moveTo(x + offset, y + height);
    ctx.lineTo(x + offset + height, y);
    ctx.stroke();
  }
  ctx.restore();
}

function draw() {
  if (!canvas || store.get().mode !== 'engineer') return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ctx = fitCanvas(canvas, width, height);
  const now = Date.now() / 1000 - skew;
  const start = now - WINDOW_S;
  const labelWidth = 74;
  const plotWidth = width - labelWidth - 8;
  const laneHeight = (height - 20) / LANES.length;
  const x = t => labelWidth + ((t - start) / WINDOW_S) * plotWidth;
  ctx.font = `600 13px ${colors.font}`;
  ctx.textBaseline = 'middle';
  LANES.forEach((lane, index) => {
    const y = index * laneHeight;
    ctx.fillStyle = colors.muted;
    ctx.fillText(lane, 4, y + laneHeight / 2);
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(labelWidth, y + laneHeight);
    ctx.lineTo(width, y + laneHeight);
    ctx.stroke();
  });
  ctx.fillStyle = colors.muted;
  ctx.font = `12px ${colors.mono}`;
  for (let seconds = 0; seconds <= WINDOW_S; seconds += 10) {
    const position = x(start + seconds);
    ctx.fillText(seconds === WINDOW_S ? 'now' : `-${WINDOW_S - seconds}s`, Math.min(position, width - 28), height - 8);
  }
  const {segments, requeues, suppressed} = buildSegments(rows.values(), now);
  let preempted = 0;
  let spoken = 0;
  for (const segment of segments) {
    if (segment.end < start) continue;
    spoken += 1;
    const y = segment.lane * laneHeight + laneHeight * 0.2;
    const barHeight = laneHeight * 0.6;
    const left = Math.max(labelWidth, x(segment.start));
    const right = Math.max(left + 3, x(segment.end));
    ctx.fillStyle = colors.lanes[segment.lane] || colors.text;
    ctx.fillRect(left, y, right - left, barHeight);
    if (segment.endKind === 'pre-empted' || segment.endKind === 'stopped') {
      preempted += segment.endKind === 'pre-empted' ? 1 : 0;
      hatch(ctx, right - 10, y, 10, barHeight, colors.surface);
      ctx.strokeStyle = colors.cut;
      ctx.lineWidth = 3;
      const cx = right + 6;
      const cy = y + barHeight / 2;
      ctx.beginPath();
      ctx.moveTo(cx - 5, cy - 5); ctx.lineTo(cx + 5, cy + 5);
      ctx.moveTo(cx + 5, cy - 5); ctx.lineTo(cx - 5, cy + 5);
      ctx.stroke();
    }
  }
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 2;
  for (const {from, to} of requeues) {
    if (to.start < start) continue;
    ctx.strokeStyle = colors.lanes[to.lane];
    ctx.beginPath();
    ctx.moveTo(x(from.end) + 6, from.lane * laneHeight + laneHeight * 0.85);
    ctx.lineTo(x(to.start), to.lane * laneHeight + laneHeight * 0.85);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  let suppressedTotal = 0;
  const badges = [];
  for (const entry of suppressed) {
    const ticks = entry.ticks.filter(t => t >= start);
    if (!ticks.length) continue;
    suppressedTotal += entry.count;
    const y = entry.lane * laneHeight;
    ctx.strokeStyle = colors.lanes[entry.lane];
    ctx.lineWidth = 1.5;
    const first = Math.min(...ticks);
    const last = Math.max(...ticks);
    const stride = Math.max(4, (x(last) - x(first)) / Math.min(entry.count, 40));
    for (let px = x(first); px <= x(last) + 0.5; px += stride) {
      ctx.beginPath();
      ctx.moveTo(px, y + laneHeight * 0.82);
      ctx.lineTo(px, y + laneHeight * 0.98);
      ctx.stroke();
    }
    const badge = `×${entry.count} suppressed`;
    badges.push({lane: LANES[entry.lane], count: entry.count, text: badge});
    ctx.font = `700 12px ${colors.font}`;
    const badgeWidth = ctx.measureText(badge).width + 10;
    const bx = Math.min(width - badgeWidth - 2, x(last) + 4);
    ctx.fillStyle = colors.text;
    ctx.fillRect(bx, y + 2, badgeWidth, 16);
    ctx.fillStyle = colors.surface;
    ctx.fillText(badge, bx + 5, y + 10);
  }
  Object.assign(canvas.dataset, {spoken: spoken, preempted: preempted, requeued: requeues.filter(r => r.to.start >= start).length,
                                 suppressed: suppressedTotal, badges: JSON.stringify(badges)});
  const text = `Last ${WINDOW_S} seconds: ${spoken} spoken, ${preempted} pre-empted, ${requeues.length} requeued, ${suppressedTotal} suppressed by cooldown.`;
  if (summary.textContent !== text) summary.textContent = text;
}

export function initTimeline() {
  canvas = document.getElementById('timeline');
  summary = document.getElementById('timelineSummary');
  readColors();
  on('state', ingest);
  on('speak', message => speakReceived.set(message.id, Date.now() / 1000));
  store.subscribe(state => { if (state.theme !== colors.theme) { readColors(); colors.theme = state.theme; } });
  setInterval(draw, 250);
}
