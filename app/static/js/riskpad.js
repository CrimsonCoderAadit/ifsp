// Mock risk events for Person 1's contract: direction pad, level, object, Fire and 30 fps spam.
import {enableAudio} from './audio.js';
import {events} from './state.js';
import {send} from './ws.js';

const SPAM_FPS = 30;
const SPAM_SECONDS = 5;
let spamTimer = null;

function selected(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value;
}

export function fireRisk(level = selected('riskLevel')) {
  const object = document.getElementById('riskObject').value;
  const direction = selected('riskDirection');
  const sent = send({type: 'mock_risk', risk: level, object, direction});
  if (sent) events.emit('risk-sent', {level, object, direction});
  return sent;
}

function spam(button, status) {
  if (spamTimer) return;
  const started = performance.now();
  let sent = 0;
  button.disabled = true;
  status.textContent = `Sending HIGH ${document.getElementById('riskObject').value} warnings at ${SPAM_FPS} per second…`;
  spamTimer = setInterval(() => {
    fireRisk('HIGH');
    sent += 1;
    if (performance.now() - started >= SPAM_SECONDS * 1000) {
      clearInterval(spamTimer);
      spamTimer = null;
      button.disabled = false;
      status.textContent = `Sent ${sent} warnings in ${SPAM_SECONDS} s. See the timeline for how many were spoken.`;
    }
  }, 1000 / SPAM_FPS);
}

export function initRiskPad() {
  const status = document.getElementById('riskStatus');
  document.getElementById('riskFire').addEventListener('click', async () => {
    await enableAudio();
    if (fireRisk()) status.textContent = `Sent ${selected('riskLevel')} ${document.getElementById('riskObject').value} ${selected('riskDirection')}.`;
  });
  const spamButton = document.getElementById('riskSpam');
  spamButton.addEventListener('click', async () => {
    await enableAudio();
    spam(spamButton, status);
  });
}
