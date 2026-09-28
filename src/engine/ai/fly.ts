/**
 * The Fly: a CPU seat (`difficulty: 'fly'`) steered by drives and a learned
 * reward memory, standing in for a simulated fruit fly brain.
 *
 * It does not replace the objective CPU — it rides on top of it. Every legal
 * action still gets its tactical + objective score; the fly then adds three
 * things, each in one place:
 *
 *   drives        hunger / aggression / fear, read off the board each call
 *        ↓        (`flyDrives` — the function a connectome simulation replaces)
 *   bias          per-tag pull: territory > planting > harvest > cards, scaled
 *        ↓        by the drives, plus whatever the fly has LEARNED about that
 *                 tag in this kind of situation (`flyBias`)
 *   brake         the risk rules, applied after the tactical vetoes: one fight
 *                 a turn is fine, a second needs good odds, a third near-
 *                 certain ones, and every bar rises as reinforcements run out
 *                 (`flyBrake`)
 *
 * LEARNING. A crude stand-in for the mushroom body: the fly keeps one value per
 * (situation, tag). When it acts, the tags of the chosen action get an
 * eligibility trace; when rewards arrive (read from the engine's event log on
 * the fly's next call), every traced value moves toward the reward. Traces
 * decay per decision, so credit goes mostly to recent choices.
 *
 * The learned values live in a `FlyBrain` — plain JSON, meant to be kept by the
 * host across games (the UI saves it to localStorage). Per-game bookkeeping
 * (traces, fights this turn, the reward log) lives in a `FlyEpisode` inside the
 * caller's `AiMemory` and is discarded with the game.
 *
 * NOT DETERMINISTIC ACROSS GAMES by design: the same seed played by a fly with
 * a different brain plays differently. Within one game, given the same brain
 * and memory, it is still a pure function of the state (no Math.random).
 */

import type { Action, GameEvent, GameState, PlayerId } from '../types';
import { enemyUnitsAt, gardenAt, playerUnitsAt } from '../helpers';
import { END_TURN_SCORE, ownedEconomyGardens } from './scoring';
import { desperation, enemyGnomes, ownGnomes } from './util';

// ---------------------------------------------------------------------------
// Tuning — every knob that shapes the fly's personality
// ---------------------------------------------------------------------------

/**
 * Rewards the fly learns from. The priority order is territory > planting >
 * harvest > card interaction; risk (fights, lost gnomes) is a brake on all of
 * them rather than a rival goal.
 */
export const FLY_REWARDS = {
  /** Per garden net-gained (or lost, negated) between the fly's turn starts. */
  territoryPerGarden: 6,
  gardenPlanted: 5,
  gardenUpgraded: 4,
  /** Per gnome from a Dandelion harvest or Mushroom clone. */
  harvestPerGnome: 2,
  maizeHarvest: 2,
  homeHarvest: 1,
  cardDrawn: 0.5,
  cardResolved: 1.5,
  /** Fizzled, cancelled, or discarded unplayed. */
  cardWasted: -1,
  enemyGnomeDestroyed: 2,
  /** Multiplied by `scarcity` — a gnome hurts more when few remain. */
  ownGnomeLost: -3,
  /** Per fight started beyond the first in one of the fly's own turns. */
  extraFight: -2,
  opponentEliminated: 4,
  eliminated: -10,
  won: 15,
} as const;

/** Standing pull toward each tag before drives and learning. */
export const FLY_PRIORITY: Record<FlyTag, number> = {
  territory: 3,
  plant: 2.5,
  harvest: 2,
  card: 1.2,
  draw: 0.6,
  fight: 1.5,
  advance: 0,
  pass: 0,
};

/** Minimum win probability to take the Nth fight of a turn (index 0 = first). */
export const FLY_FIGHT_ODDS = [0.4, 0.62, 0.85] as const;

const LEARNING_RATE = 0.05;
const TRACE_DECAY = 0.85;
const VALUE_LIMIT = 3;
const REWARD_LOG_LIMIT = 100;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** What kind of thing an action is, as far as the fly's incentives care. */
export type FlyTag = 'territory' | 'plant' | 'harvest' | 'card' | 'draw' | 'fight' | 'advance' | 'pass';

/** The persistent, learned half of the fly. Plain JSON; keep it across games. */
export interface FlyBrain {
  version: 1;
  gamesPlayed: number;
  /** Learned value per `${situation}:${tag}`; absent = 0. */
  values: Record<string, number>;
}

/** Drives read off the board: 1 is neutral. */
export interface FlyDrives {
  /** 1–2: economy is thin (few Wishes, no held economy garden). */
  hunger: number;
  /** 0.5–2: our board force against the strongest enemy's. */
  aggression: number;
  /** 1–4: how scarce our remaining reinforcements are. */
  fear: number;
}

export interface FlyRewardEntry {
  turn: number;
  reason: string;
  amount: number;
}

/** Per-game, per-seat bookkeeping. Discarded with the game. */
interface FlyEpisode {
  seed: number;
  lastEventCount: number;
  /** Turn number of the fly's current (or last) own turn. */
  turn: number;
  fightsThisTurn: number;
  /** Gardens held at the start of the fly's last turn (null before the first). */
  territory: number | null;
  traces: Map<string, number>;
  log: FlyRewardEntry[];
  finished: boolean;
}

/** The fly's slice of `AiMemory`. */
export interface FlyMemory {
  brain: FlyBrain;
  episodes: Map<PlayerId, FlyEpisode>;
}

/** Everything one decision needs, gathered once by `flyObserve`. */
export interface FlyContext {
  brain: FlyBrain;
  episode: FlyEpisode;
  drives: FlyDrives;
  situation: string;
}

// ---------------------------------------------------------------------------
// Brain lifecycle
// ---------------------------------------------------------------------------

export function createFlyBrain(): FlyBrain {
  return { version: 1, gamesPlayed: 0, values: {} };
}

export function createFlyMemory(brain: FlyBrain = createFlyBrain()): FlyMemory {
  return { brain, episodes: new Map() };
}

/** Read a brain back from JSON, or null if it is not one we understand. */
export function parseFlyBrain(json: string): FlyBrain | null {
  try {
    const raw: unknown = JSON.parse(json);
    if (typeof raw !== 'object' || raw === null) return null;
    const r = raw as Partial<FlyBrain>;
    if (r.version !== 1 || typeof r.gamesPlayed !== 'number' || typeof r.values !== 'object' || r.values === null) {
      return null;
    }
    const values: Record<string, number> = {};
    for (const [k, v] of Object.entries(r.values)) {
      if (typeof v === 'number' && Number.isFinite(v)) values[k] = clamp(v, -VALUE_LIMIT, VALUE_LIMIT);
    }
    return { version: 1, gamesPlayed: r.gamesPlayed, values };
  } catch {
    return null;
  }
}

export function isFly(state: GameState, player: PlayerId): boolean {
  return state.players[player]?.difficulty === 'fly';
}

// ---------------------------------------------------------------------------
// Drives and situation
// ---------------------------------------------------------------------------

/** 1 with a full budget, rising to 4 as the last gnomes run out. */
export function scarcity(state: GameState, player: PlayerId): number {
  const total = state.config.totalReinforcements;
  const remaining = Math.max(0, total - state.players[player].gnomesLost);
  const spent = total > 0 ? 1 - remaining / total : 1;
  return 1 + 3 * spent * spent;
}

/**
 * What the fly wants right now. This is the seam for the connectome: a
 * simulated brain replaces this function, reading the same board facts as
 * sensory input and returning the same three drives from its motor output.
 */
export function flyDrives(state: GameState, player: PlayerId): FlyDrives {
  const p = state.players[player];
  let hunger = 1;
  if (p.wishes <= 1) hunger += 0.5;
  if (ownedEconomyGardens(state, player) === 0) hunger += 0.5;

  const ours = ownGnomes(state, player).length;
  const counts = new Map<PlayerId, number>();
  for (const u of enemyGnomes(state, player)) counts.set(u.owner, (counts.get(u.owner) ?? 0) + 1);
  const strongest = Math.max(1, ...counts.values());
  const aggression = clamp(ours / strongest, 0.5, 2);

  return { hunger, aggression, fear: scarcity(state, player) };
}

/** A coarse situation key, so learning is conditional: "hungry+scarce" etc. */
function situationOf(drives: FlyDrives): string {
  const bits: string[] = [];
  if (drives.hunger > 1) bits.push('hungry');
  if (drives.aggression < 1) bits.push('outnumbered');
  if (drives.fear >= 2) bits.push('scarce');
  return bits.length > 0 ? bits.join('+') : 'calm';
}

// ---------------------------------------------------------------------------
// Tagging and fight odds
// ---------------------------------------------------------------------------

/** The incentive tags an action touches. A move can be several at once. */
export function flyTags(state: GameState, player: PlayerId, action: Action): FlyTag[] {
  switch (action.type) {
    case 'move': {
      const tags: FlyTag[] = [];
      if (enemyUnitsAt(state, action.to, player).length > 0) tags.push('fight');
      const g = gardenAt(state, action.to);
      const held = playerUnitsAt(state, action.to, player).some((u) => u.kind === 'gnome');
      if (g && !held && g.type !== 'flytrap' && !(g.type === 'home' && g.owner === player)) {
        tags.push('territory');
        if (g.type === 'dandelion' || g.type === 'mushroom' || g.type === 'maize') tags.push('harvest');
      }
      return tags.length > 0 ? tags : ['advance'];
    }
    case 'plant':
    case 'upgrade':
      return ['plant'];
    case 'drawCard':
      return ['draw'];
    case 'playCard':
      return ['card'];
    case 'endTurn':
      return ['pass'];
    default:
      return [];
  }
}

/** Win probability of a move that starts a fight (gambler's ruin, 1 vs N). */
function fightOdds(state: GameState, player: PlayerId, action: Action): number | null {
  if (action.type !== 'move') return null;
  const defenders = enemyUnitsAt(state, action.to, player).length;
  if (defenders === 0) return null;
  const attackers = 1 + desperation(state) * 0.15;
  return attackers / (attackers + defenders);
}

// ---------------------------------------------------------------------------
// The three hooks index.ts calls
// ---------------------------------------------------------------------------

/**
 * Start of every decision for a fly seat: pick up this game's episode, learn
 * from everything that happened since the last call, and read the drives.
 */
export function flyObserve(state: GameState, player: PlayerId, memory: FlyMemory): FlyContext {
  const episode = episodeFor(memory, state, player);
  learnFromEvents(state, player, memory.brain, episode);

  // New own turn: territory reward for what the last turn gained or lost, and
  // a fresh fight budget.
  const turn = state.turn;
  if (turn && turn.activePlayer === player && turn.number !== episode.turn) {
    episode.turn = turn.number;
    episode.fightsThisTurn = 0;
    const held = territoryHeld(state, player);
    if (episode.territory !== null && held !== episode.territory) {
      const delta = held - episode.territory;
      reward(memory.brain, episode, turn.number, delta * FLY_REWARDS.territoryPerGarden,
        `${delta > 0 ? 'gained' : 'lost'} ${Math.abs(delta)} garden${Math.abs(delta) === 1 ? '' : 's'}`);
    }
    episode.territory = held;
  }

  const drives = flyDrives(state, player);
  return { brain: memory.brain, episode, drives, situation: situationOf(drives) };
}

/**
 * Pull toward an action, added to its score BEFORE the tactical vetoes are
 * enforced — so it can make a sound action more attractive but never approve
 * one the tactics rejected.
 */
export function flyBias(ctx: FlyContext, state: GameState, player: PlayerId, action: Action): number {
  const { drives } = ctx;
  let bias = 0;
  for (const tag of flyTags(state, player, action)) {
    let drive = 1;
    if (tag === 'territory') drive = 0.5 + 0.5 * drives.aggression;
    else if (tag === 'plant' || tag === 'harvest') drive = drives.hunger;
    else if (tag === 'fight') drive = drives.aggression / drives.fear;
    bias += FLY_PRIORITY[tag] * drive + learned(ctx, tag);
  }
  return bias;
}

/**
 * The risk rules, applied AFTER the tactical vetoes — a hard brake, so a fight
 * that fails its bar drops below passing no matter how the bias pulled it.
 */
export function flyBrake(ctx: FlyContext, state: GameState, player: PlayerId, action: Action, score: number): number {
  const odds = fightOdds(state, player, action);
  if (odds === null) return score;
  const nth = Math.min(ctx.episode.fightsThisTurn, FLY_FIGHT_ODDS.length - 1);
  const g = action.type === 'move' ? gardenAt(state, action.to) : null;
  const storming = !!g && g.type === 'home' && g.owner !== player;
  const bar =
    FLY_FIGHT_ODDS[nth] +
    0.15 * (ctx.drives.fear - 1) - // scarce reinforcements raise every bar
    0.1 * (ctx.drives.aggression - 1) - // a stronger force lowers it a little
    (storming ? 0.1 : 0);
  if (odds < bar) return Math.min(score, END_TURN_SCORE - 0.05);
  // Accepted — but still priced: the expected loss, heavier when gnomes are scarce.
  return score - (1 - odds) * 2 * ctx.drives.fear;
}

/** Mark the chosen action's tags as eligible for the rewards that follow. */
export function flyRecordChoice(ctx: FlyContext, state: GameState, player: PlayerId, action: Action): void {
  for (const tag of flyTags(state, player, action)) ctx.episode.traces.set(`${ctx.situation}:${tag}`, 1);
}

/**
 * Settle a finished game for every fly seat: learn from the final events (the
 * win or elimination) and count the game. Idempotent per game. Hosts call it
 * once the game ends, then persist the brain.
 */
export function finishFlyGames(state: GameState, memory: FlyMemory): void {
  if (state.status !== 'finished') return;
  for (const p of state.players) {
    if (p.difficulty !== 'fly') continue;
    const episode = memory.episodes.get(p.id);
    if (!episode || episode.finished || episode.seed !== state.seed) continue;
    learnFromEvents(state, p.id, memory.brain, episode);
    episode.finished = true;
    memory.brain.gamesPlayed += 1;
  }
}

/** The fly's recent rewards, newest last — for a debug panel or a test. */
export function flyRewardLog(memory: FlyMemory, player: PlayerId): readonly FlyRewardEntry[] {
  return memory.episodes.get(player)?.log ?? [];
}

// ---------------------------------------------------------------------------
// Learning internals
// ---------------------------------------------------------------------------

function episodeFor(memory: FlyMemory, state: GameState, player: PlayerId): FlyEpisode {
  const existing = memory.episodes.get(player);
  if (existing && existing.seed === state.seed && state.eventCount >= existing.lastEventCount) return existing;
  const fresh: FlyEpisode = {
    seed: state.seed,
    lastEventCount: state.eventCount,
    turn: -1,
    fightsThisTurn: 0,
    territory: null,
    traces: new Map(),
    log: [],
    finished: false,
  };
  memory.episodes.set(player, fresh);
  return fresh;
}

function learned(ctx: FlyContext, tag: FlyTag): number {
  return ctx.brain.values[`${ctx.situation}:${tag}`] ?? 0;
}

/** Non-home gardens where we have a gnome and the enemy has nothing. */
function territoryHeld(state: GameState, player: PlayerId): number {
  let n = 0;
  for (const [key, g] of Object.entries(state.gardens)) {
    if (g.type === 'home' && g.owner === player) continue;
    const [x, y] = key.split(',').map(Number);
    const pos = { x, y };
    if (
      playerUnitsAt(state, pos, player).some((u) => u.kind === 'gnome') &&
      enemyUnitsAt(state, pos, player).length === 0
    ) {
      n += 1;
    }
  }
  return n;
}

/** Events emitted since the episode last looked, oldest first. */
function newEvents(state: GameState, episode: FlyEpisode): readonly GameEvent[] {
  const fresh = state.eventCount - episode.lastEventCount;
  episode.lastEventCount = state.eventCount;
  if (fresh <= 0) return [];
  return state.events.slice(Math.max(0, state.events.length - fresh));
}

function learnFromEvents(state: GameState, player: PlayerId, brain: FlyBrain, episode: FlyEpisode): void {
  const turn = state.turn?.number ?? 0;
  for (const ev of newEvents(state, episode)) {
    const r = rewardFor(state, player, episode, ev);
    if (r) reward(brain, episode, turn, r.amount, r.reason);
  }
  // Older choices earn less credit for whatever comes next.
  for (const [k, t] of episode.traces) {
    const next = t * TRACE_DECAY;
    if (next < 0.05) episode.traces.delete(k);
    else episode.traces.set(k, next);
  }
}

function reward(brain: FlyBrain, episode: FlyEpisode, turn: number, amount: number, reason: string): void {
  if (amount === 0) return;
  for (const [k, t] of episode.traces) {
    const v = (brain.values[k] ?? 0) + LEARNING_RATE * amount * t;
    brain.values[k] = clamp(v, -VALUE_LIMIT, VALUE_LIMIT);
  }
  episode.log.push({ turn, reason, amount: Math.round(amount * 100) / 100 });
  if (episode.log.length > REWARD_LOG_LIMIT) episode.log.shift();
}

function rewardFor(
  state: GameState,
  player: PlayerId,
  episode: FlyEpisode,
  ev: GameEvent,
): { amount: number; reason: string } | null {
  const R = FLY_REWARDS;
  switch (ev.type) {
    case 'gardenPlanted':
      return ev.player === player ? { amount: R.gardenPlanted, reason: `planted ${ev.gardenType}` } : null;
    case 'gardenUpgraded':
      return ev.player === player ? { amount: R.gardenUpgraded, reason: `upgraded ${ev.gardenType}` } : null;
    case 'dandelionHarvested':
      return ev.player === player && ev.gnomes > 0
        ? { amount: R.harvestPerGnome * ev.gnomes, reason: `dandelion +${ev.gnomes}` }
        : null;
    case 'mushroomHarvested':
      return ev.player === player && ev.cloned > 0
        ? { amount: R.harvestPerGnome * ev.cloned, reason: `mushroom cloned ${ev.cloned}` }
        : null;
    case 'maizeHarvested':
      return ev.player === player ? { amount: R.maizeHarvest * (ev.doubled ? 2 : 1), reason: 'maize harvest' } : null;
    case 'homeHarvested':
      return ev.player === player && ev.took !== 'nothing'
        ? { amount: R.homeHarvest, reason: `home harvest (${ev.took})` }
        : null;
    case 'cardDrawn':
      return ev.player === player ? { amount: R.cardDrawn, reason: 'drew a card' } : null;
    case 'cardResolved':
      return ev.player === player ? { amount: R.cardResolved, reason: `played ${ev.cardId}` } : null;
    case 'cardFizzled':
    case 'cardCancelled':
    case 'cardDiscarded':
      return ev.player === player ? { amount: R.cardWasted, reason: `wasted ${ev.cardId}` } : null;
    case 'unitDestroyed':
      if (ev.unitKind !== 'gnome') return null;
      return ev.player === player
        ? { amount: R.ownGnomeLost * scarcity(state, player), reason: 'lost a gnome' }
        : { amount: R.enemyGnomeDestroyed, reason: 'enemy gnome destroyed' };
    case 'fightStarted': {
      const attacker = ev.sides[1];
      if (attacker.kind !== 'player' || attacker.player !== player) return null;
      episode.fightsThisTurn += 1;
      return episode.fightsThisTurn > 1
        ? { amount: R.extraFight * (episode.fightsThisTurn - 1), reason: `fight #${episode.fightsThisTurn} this turn` }
        : null;
    }
    case 'playerEliminated':
      return ev.player === player
        ? { amount: R.eliminated, reason: 'eliminated' }
        : { amount: R.opponentEliminated, reason: 'an opponent fell' };
    case 'gameFinished':
      return ev.winner === player ? { amount: R.won, reason: 'won the game' } : null;
    default:
      return null;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
