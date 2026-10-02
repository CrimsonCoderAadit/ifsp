// Latency cards with budgets and sparklines, plus the collapsible System card (/api/health).
import {fitCanvas, storage, store, watch} from './state.js';
import {on} from './ws.js';

const CARDS = [
  {id: 'stt', title: 'STT', note: 'audio end → transcript'},
  {id: 'ocr', title: 'OCR', note: 'per crop read'},
  {id: 'tts', title: 'TTS', note: 'event → audio start'},
  {id: 'detector', title: 'Detector', note: 'per frame'},
];
const DEFAULT_BUDGETS = {stt: 500, ocr: 250, tts: 300, detector: 100};
const SPARK_POINTS = 40;
const series = Object.fromEntries(CARDS.map(card => [card.id, []]));
const counts = {};
const seenPlayed = new Set();
const ttsAge = [];
let budgets = {...DEFAULT_BUDGETS};
let latest = {};
const elements = {};

try { budgets = {...DEFAULT_BUDGETS, ...JSON.parse(storage.get('budgets', '{}'))}; } catch { /* keep defaults */ }

const fmt = value => (value == null ? '—' : value < 10 ? value.toFixed(1) : String(Math.round(value)));

function percentile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const low = Math.floor(index);
  return sorted[low] + (sorted[Math.ceil(index)] - sorted[low]) * (index - low);
}

export function verdict(value, budget) {
  if (value == null) return {level: 'none', text: `budget ${budget} ms`};
  if (value > budget) return {level: 'over', text: `over budget (${budget} ms)`};
  if (value > budget * 0.8) return {level: 'warn', text: `near budget (${budget} ms)`};
  return {level: 'ok', text: `within budget (${budget} ms)`};
}

function sparkline(canvas, values, budget) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ctx = fitCanvas(canvas, width, height);
  const style = getComputedStyle(canvas);
  const max = Math.max(budget * 1.2, ...values);
  const y = value => height - 2 - (value / max) * (height - 4);
  ctx.strokeStyle = style.getPropertyValue('--spark-budget').trim();
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, y(budget));
  ctx.lineTo(width, y(budget));
  ctx.stroke();
  ctx.setLineDash([]);
  if (values.length < 2) return;
  ctx.strokeStyle = style.color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = (index / (SPARK_POINTS - 1)) * width;
    ctx[index ? 'lineTo' : 'moveTo'](x, y(value));
  });
  ctx.stroke();
}

function render() {
  for (const card of CARDS) {
    const element = elements[card.id];
    const stats = latest[card.id] || {};
    const budget = budgets[card.id];
    const {level, text} = verdict(stats.last, budget);
    element.dataset.level = level;
    element.querySelector('.big-value').textContent = fmt(stats.last);
    element.querySelector('.sub').textContent = `p50 ${fmt(stats.p50)} · p95 ${fmt(stats.p95)} · n ${stats.count ?? 0}`;
    element.querySelector('.verdict').textContent = text;
    element.querySelector('.extra').textContent = card.id === 'tts' && stats.synth != null ? `synthesis p50 ${fmt(stats.synth)} ms` : '';
    sparkline(element.querySelector('canvas'), series[card.id], budget);
  }
}

function onMetrics(message) {
  for (const id of ['stt', 'ocr', 'detector']) {
    const stats = message.latency_ms?.[id];
    if (!stats) continue;
    if (stats.count && stats.count !== counts[id]) {
      counts[id] = stats.count;
      series[id].push(stats.last);
      if (series[id].length > SPARK_POINTS) series[id].shift();
    }
    latest[id] = stats;
  }
  latest.tts = {...(latest.tts || {}), synth: message.latency_ms?.tts?.p50};
  render();
}

function onState(message) {
  let changed = false;
  for (const row of message.scheduler?.log || []) {
    if (row.decision !== 'played' || row.age_ms === '' || seenPlayed.has(row.item_id + row.timestamp)) continue;
    seenPlayed.add(row.item_id + row.timestamp);
    ttsAge.push(Number(row.age_ms));
    series.tts.push(Number(row.age_ms));
    if (series.tts.length > SPARK_POINTS) series.tts.shift();
    changed = true;
  }
  if (!changed) return;
  if (ttsAge.length > 1000) ttsAge.splice(0, ttsAge.length - 1000);
  latest.tts = {...(latest.tts || {}), last: ttsAge[ttsAge.length - 1], p50: percentile(ttsAge, .5),
                p95: percentile(ttsAge, .95), count: ttsAge.length};
  render();
}

function renderHealth(health) {
  const list = document.getElementById('healthList');
  list.replaceChildren();
  if (!health) return;
  const add = (term, value) => {
    const dt = document.createElement('dt');
    dt.textContent = term;
    const dd = document.createElement('dd');
    dd.textContent = value;
    list.append(dt, dd);
  };
  for (const [name, model] of Object.entries(health.models || {})) {
    const load = health.load_time_ms?.[name];
    const fallback = model.fallback_reason ? ` — FALLBACK: ${model.fallback_reason}` : '';
    const file = Array.isArray(model.model) ? model.model.join(', ') : model.model || 'n/a';
    add(name.toUpperCase(), `${model.backend} · ${file}${load != null ? ` · load ${fmt(load)} ms` : ''}${fallback}`);
  }
  add('Routes', `${health.routes} GTFS routes${health.load_time_ms?.route_metadata != null ? ` · metadata ${fmt(health.load_time_ms.route_metadata)} ms` : ''}`);
  add('Build', health.git?.commit ? `${health.git.commit.slice(0, 10)}${health.git.dirty ? ' (uncommitted changes)' : ''}` : 'unknown');
}

export async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    store.set({health: await response.json()});
  } catch { /* System card stays empty; the connection toast covers outages */ }
}

export function initLatency() {
  const grid = document.getElementById('latencyGrid');
  for (const card of CARDS) {
    const element = document.createElement('article');
    element.className = 'latency-card';
    element.dataset.metric = card.id;
    element.dataset.level = 'none';
    element.innerHTML = `<h3 class="latency-title"></h3><p class="latency-note"></p>
      <p class="latency-big"><span class="big-value">—</span> <span class="unit">ms</span></p>
      <p class="sub"></p><p class="extra"></p><canvas class="spark" aria-hidden="true"></canvas><p class="verdict"></p>`;
    element.querySelector('.latency-title').textContent = card.title;
    element.querySelector('.latency-note').textContent = card.note;
    grid.append(element);
    elements[card.id] = element;
    const input = document.getElementById(`budget-${card.id}`);
    input.value = budgets[card.id];
    input.addEventListener('change', () => {
      const value = Number(input.value);
      if (!(value > 0)) { input.value = budgets[card.id]; return; }
      budgets[card.id] = value;
      storage.set('budgets', JSON.stringify(budgets));
      render();
    });
  }
  on('metrics', onMetrics);
  on('state', onState);
  watch(state => state.health, renderHealth);
  render();
}
