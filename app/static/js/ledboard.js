// Dot-matrix target-bus board: route in large LED digits, GTFS destinations, and the found state.
import {store, watch} from './state.js';

const MAX_DESTINATIONS = 3;
const LOST_AFTER_MS = 5000;
const destinationCache = new Map();
let found = null;            // {route, direction, lastSeen}

export function destinationLine(names) {
  return (names || []).slice(0, MAX_DESTINATIONS)
    .map(name => name.split(/\s+TO\s+/i).map(part => part.trim()).join(' → ')).join('   ·   ');
}

// target_bus.long_names is authoritative; older servers omit it, so ask /api/route.
async function destinations(target) {
  if (Array.isArray(target.long_names)) return {names: target.long_names};
  if (!destinationCache.has(target.route)) {
    destinationCache.set(target.route, (async () => {
      try {
        const response = await fetch(`/api/route/${encodeURIComponent(target.route)}`);
        const body = await response.json().catch(() => ({}));
        if (response.ok) return {names: body.long_names || []};
        if (response.status === 404 && /^Unknown route/.test(body.detail || '')) return {missing: true};
        return {unavailable: true};
      } catch {
        return {unavailable: true};
      }
    })());
  }
  return destinationCache.get(target.route);
}

function setMarquee(text) {
  const marquee = document.getElementById('ledMarquee');
  for (const copy of marquee.querySelectorAll('.marquee-copy')) copy.textContent = text;
  marquee.dataset.scroll = String(text.length > 28);
  document.querySelector('.led-dest').hidden = !text;
}

async function renderDestinations(target) {
  const more = document.getElementById('ledMore');
  more.textContent = '';
  if (!target) { setMarquee('Say “find bus 21G”'); return; }
  setMarquee('');
  const result = await destinations(target);
  if (store.get().target?.route !== target.route) return;
  if (result.missing) setMarquee('Not in GTFS feed');
  else if (result.unavailable) setMarquee('Destination unavailable');
  else {
    setMarquee(destinationLine(result.names));
    const extra = Math.max(0, result.names.length - MAX_DESTINATIONS);
    more.textContent = extra ? `+${extra} more GTFS variants` : '';
  }
}

function renderStatus() {
  const target = store.get().target;
  const board = document.getElementById('ledBoard');
  const status = document.getElementById('ledStatus');
  const isFound = Boolean(target && found && found.route === target.route);
  board.dataset.found = String(isFound);
  status.dataset.found = String(isFound);
  status.textContent = !target ? 'No bus selected.'
    : isFound ? `${target.route} is here, ${found.direction}` : `Looking for bus ${target.route}…`;
}

function announce(text) {
  const info = document.getElementById('busInfo');
  info.textContent = text;
}

function pulse() {
  const board = document.getElementById('ledBoard');
  board.classList.remove('found-pulse');
  void board.offsetWidth;
  board.classList.add('found-pulse');
}

export function directionWord(text) {
  if (/left/.test(text || '')) return 'left';
  if (/right/.test(text || '')) return 'right';
  return 'ahead';
}

function onOverlay(overlay) {
  const target = store.get().target?.route;
  const box = target && overlay?.boxes?.find(item => item.confirmed_route === target);
  if (!box) return;
  const direction = directionWord(box.direction);
  const wasFound = found?.route === target;
  found = {route: target, direction, lastSeen: Date.now()};
  if (!wasFound) {
    pulse();
    announce(`Bus ${target} is here, ${direction}.`);
  }
  renderStatus();
}

function checkLost() {
  if (!found || Date.now() - found.lastSeen <= LOST_AFTER_MS) return;
  const route = found.route;
  found = null;
  renderStatus();
  if (store.get().target?.route === route) announce(`Lost sight of bus ${route}. Looking again.`);
}

function renderTarget(target) {
  const board = document.getElementById('ledBoard');
  board.dataset.empty = String(!target);
  document.getElementById('ledRoute').textContent = target?.route || '- - -';
  if (found && found.route !== target?.route) found = null;
  renderDestinations(target);
  renderStatus();
}

export function initLedBoard() {
  const targetKey = ({target}) => !target ? ''
    : `${target.route}|${Array.isArray(target.long_names) ? target.long_names.join('\n') : 'absent'}`;
  watch(targetKey, () => renderTarget(store.get().target));
  watch(state => state.overlay, onOverlay);
  setInterval(checkLost, 500);
  const pause = document.getElementById('marqueePause');
  pause.addEventListener('click', () => {
    const marquee = document.getElementById('ledMarquee');
    const paused = marquee.dataset.paused !== 'true';
    marquee.dataset.paused = String(paused);
    pause.setAttribute('aria-pressed', String(paused));
    pause.textContent = paused ? 'Resume scrolling' : 'Pause scrolling';
  });
}
