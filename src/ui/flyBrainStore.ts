/**
 * The Fly's brain for this tab.
 *
 * Every fly starts from the pre-trained brain shipped with the game
 * (engine/ai/trainedFlyBrain.ts) and keeps learning across the games played in
 * this tab. Nothing is written to the device, in keeping with DEPLOYMENT.md's
 * no-local-storage posture: a reload starts again from the shipped brain.
 */

import type { FlyBrain } from '../engine';
import { trainedFlyBrain } from '../engine';

let brain: FlyBrain | null = null;

/** This tab's fly brain, shared by every game played in it. */
export function tabFlyBrain(): FlyBrain {
  brain ??= trainedFlyBrain();
  return brain;
}
