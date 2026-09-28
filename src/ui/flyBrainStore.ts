/**
 * Where the Fly's learned brain lives between games on this device.
 *
 * The fly is meant to remember across games (see engine/ai/fly.ts), so its
 * brain — a few dozen numbers of plain JSON — is kept in localStorage. This is
 * a deliberate exception to the "no local storage" posture in DEPLOYMENT.md,
 * made for this experimental fork only. Every access is guarded: if storage is
 * unavailable the fly simply starts fresh, and forgets at the end of the tab.
 */

import type { FlyBrain } from '../engine';
import { createFlyBrain, parseFlyBrain } from '../engine';

const KEY = 'whimsy.flyBrain.v1';

let brain: FlyBrain | null = null;

/** The device's fly brain, loaded once and shared by every game in the tab. */
export function loadFlyBrain(): FlyBrain {
  if (brain) return brain;
  let stored: FlyBrain | null = null;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (raw) stored = parseFlyBrain(raw);
  } catch {
    // Storage blocked (private mode, sandboxed frame): start fresh.
  }
  brain = stored ?? createFlyBrain();
  return brain;
}

export function saveFlyBrain(): void {
  if (!brain) return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(brain));
  } catch {
    // Best effort; the in-tab brain still carries over to the next game.
  }
}
