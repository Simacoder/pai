// Small pure helpers for the site. No imports, so they are easy to test.

export type Bin = {predicted: number; actual: number; lo: number; hi: number; n: number};
export type TableRow = {gap: number; elo400: number; fitted: number; fitted_lo: number; fitted_hi: number};

export const LN10 = Math.LN10;

/** White's expected score (a draw counts as half) when White is rated `gap` points higher. */
export function winChance(gap: number, scale: number, whiteAdvantage: number): number {
  return 1 / (1 + Math.pow(10, -(gap + whiteAdvantage) / scale));
}

/**
 * Average gap: how far, on average, the predicted chance is from the actual win rate.
 * Games are sorted into bins of similar predicted chance; each bin's |actual - predicted|
 * is averaged, with bigger bins counting more. 0 means perfectly calibrated.
 */
export function averageGap(bins: Bin[]): number {
  const games = bins.reduce((sum, b) => sum + b.n, 0);
  return bins.reduce((sum, b) => sum + b.n * Math.abs(b.actual - b.predicted), 0) / games;
}

/** Score per game = average log10 likelihood (higher is better). Input is the natural-log loss. */
export function scorePerGame(logLoss: number): number {
  return -logLoss / LN10;
}

/** Straight-line interpolation of a column of the win-chance table at any gap. */
export function interpolate(table: TableRow[], gap: number, key: 'fitted_lo' | 'fitted_hi'): number {
  const first = table[0];
  const last = table[table.length - 1];
  if (gap <= first.gap) return first[key];
  if (gap >= last.gap) return last[key];
  const i = table.findIndex(row => row.gap > gap);
  const a = table[i - 1];
  const b = table[i];
  const t = (gap - a.gap) / (b.gap - a.gap);
  return a[key] + t * (b[key] - a[key]);
}

export function pct(x: number, digits = 1): string {
  return `${(100 * x).toFixed(digits)}%`;
}

export function whole(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}
