// Single-key shortcuts. Ignored while typing or while a modal is open; "?" shows the cheat sheet.
export const SHORTCUTS = [
  ['Space', 'Hold to talk'],
  ['R', 'Repeat last message'],
  ['S', 'Stop speech'],
  ['A', 'What’s ahead?'],
  ['C', 'Cancel bus search'],
  ['M', 'Switch Rider / Engineer mode'],
  ['H', 'Toggle high contrast'],
  ['1', 'Mock MEDIUM risk'],
  ['2', 'Mock HIGH risk'],
  ['3', 'Mock CRITICAL risk'],
  ['P', 'Pause / resume frames'],
  ['?', 'Show this list'],
];

function typing(target) {
  return target instanceof HTMLElement
    && (target.isContentEditable || target.matches('input:not([type=radio]):not([type=checkbox]), textarea, select'));
}

export function initShortcuts(actions, {blocked}) {
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || typing(event.target) || blocked()) return;
    if (event.code === 'Space') {
      const native = event.target instanceof HTMLElement && event.target.id !== 'ptt'
        && event.target.matches('button, input, summary, a[href], select, [tabindex]:not([tabindex="-1"])');
      if (native) return;
      event.preventDefault();
      if (!event.repeat) actions.talkStart();
      return;
    }
    if (event.repeat) return;
    const action = {r: 'repeat', s: 'stop', a: 'ahead', c: 'cancel', m: 'mode', h: 'contrast',
                    1: 'riskMedium', 2: 'riskHigh', 3: 'riskCritical', p: 'pause', '?': 'help'}[event.key.toLowerCase()];
    if (!action) return;
    event.preventDefault();
    actions[action]();
  });
  document.addEventListener('keyup', event => {
    if (event.code !== 'Space' || typing(event.target)) return;
    if (event.target instanceof HTMLElement && event.target.id === 'ptt') event.preventDefault();
    actions.talkEnd();
  });
  window.addEventListener('blur', () => actions.talkEnd());
}

export function fillCheatSheet(table) {
  for (const [key, label] of SHORTCUTS) {
    const row = table.insertRow();
    const keyCell = row.insertCell();
    const kbd = document.createElement('kbd');
    kbd.textContent = key;
    keyCell.append(kbd);
    row.insertCell().textContent = label;
  }
}
