// Are chess ratings calibrated? A Probability for AI final project.
// All numbers come from chess-site-data.json (made by export_site_data.py from
// my real Lichess run) and test-improvement.json (printed by the one-time test run).
import {
  Area,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {useState} from 'react';
import './App.css';
import data from './chess-site-data.json';
import improvement from './test-improvement.json';
import guess from './my-guess.json';
import {averageGap, interpolate, LN10, pct, scorePerGame, whole, winChance} from './stats';
import type {Bin, TableRow} from './stats';

const RED = '#b3261e';
const BLUE = '#1a4fb4';
const INK = '#1b1f24';
const TICK = {fontSize: 16, fill: INK};
const AXIS_LABEL = {fontSize: 16, fill: INK};

// ---- numbers from the data files --------------------------------------------------
const eloBins = data.calibration_test.elo400 as Bin[];
const fitBins = data.calibration_test.fitted as Bin[];
const table = data.win_prob_table_white as TableRow[];
const scale = data.fitted.scale;
const whiteAdv = data.fitted.white_advantage;
const scaleCI = data.fitted.ci95.scale;
const advCI = data.fitted.ci95.white_advantage;
const rows = data.data.rows;
const counts = data.data.filter_counts;
const modelRow = (name: string) => data.models_test.find(m => m.model === name)!;
const elo = modelRow('elo400');
const eloWhite = modelRow('elo400+white');
const fit = modelRow('fitted');

const eloGap = averageGap(eloBins);
const fitGap = averageGap(fitBins);

// Score per game: log10 likelihood, a draw counted as half a win. Higher is better.
const scoreGain = improvement.logloss.mean / LN10;
const scoreGainLo = improvement.logloss.ci95[0] / LN10;
const scoreGainHi = improvement.logloss.ci95[1] / LN10;

const sampled =
  counts.kept + counts.dropped_other_time_control + counts.dropped_no_time_control + counts.dropped_termination + counts.dropped_no_result;
const row400 = table.find(r => r.gap === 400)!;
const nearGames = data.gap_histogram_test.filter(b => b.from >= -100 && b.to <= 100).reduce((s, b) => s + b.n, 0);

const withErr = (bins: Bin[], model: string) =>
  bins.map(b => ({...b, model, err: [b.actual - b.lo, b.hi - b.actual] as [number, number]}));
const eloPoints = withErr(eloBins, 'Textbook formula');
const fitPoints = withErr(fitBins, 'Fitted model');

const curve = Array.from({length: 81}, (_, i) => {
  const g = -400 + 10 * i;
  return {
    gap: g,
    elo400: winChance(g, 400, 0),
    fitted: winChance(g, scale, whiteAdv),
    band: [interpolate(table, g, 'fitted_lo'), interpolate(table, g, 'fitted_hi')],
  };
});

// ---- small pieces --------------------------------------------------------------------
function BinTip({active, payload}: {active?: boolean; payload?: {payload: Bin & {model: string}}[]}) {
  if (!active || !payload || payload.length === 0) return null;
  const b = payload[0].payload;
  return (
    <div className="tip">
      <strong>{b.model}</strong>
      <br />
      predicted {pct(b.predicted)}, actual {pct(b.actual)}
      <br />
      95% range {pct(b.lo)} to {pct(b.hi)}
      <br />
      {whole(b.n)} games
    </div>
  );
}

function Stat({big, label, tone}: {big: string; label: string; tone?: 'red' | 'blue'}) {
  return (
    <div className="stat">
      <span className={`big ${tone ?? ''}`}>{big}</span>
      <span className="label">{label}</span>
    </div>
  );
}

export default function App() {
  const [gap, setGap] = useState(200);
  const [showElo, setShowElo] = useState(true);
  const [showFit, setShowFit] = useState(true);

  const eloP = winChance(gap, 400, 0);
  const fitP = winChance(gap, scale, whiteAdv);
  const lo = interpolate(table, gap, 'fitted_lo');
  const hi = interpolate(table, gap, 'fitted_hi');
  const diff = 100 * (fitP - eloP);
  const who =
    gap === 0
      ? 'Both players have the same rating.'
      : gap > 0
        ? `White is rated ${gap} points higher than Black.`
        : `Black is rated ${-gap} points higher than White.`;
  const hasGuess = guess.text.trim() !== '';

  return (
    <main className="page">
      <h1>Are chess ratings calibrated?</h1>
      <p className="byline">A Probability for AI final project · {whole(counts.kept)} Lichess blitz games from February 2017</p>
      <p className="lede">
        When a rating gap says a player should win 70% of the time, do they win 70% of the time? I tested the
        textbook Elo formula against real games to find out.
      </p>

      <section aria-labelledby="short-answer">
        <h2 id="short-answer">The short answer</h2>
        <div className="answer">
          <p>
            <strong>Not quite. The textbook formula is overconfident for these games.</strong> Favourites win less
            often than it says, and underdogs win more often.
          </p>
          <p>
            When White is rated 400 points higher, the formula gives White <strong className="red">{pct(row400.elo400)}</strong>. In
            real games the fitted model, which matches what happened, says <strong className="blue">{pct(row400.fitted)}</strong>{' '}
            (95% range {pct(row400.fitted_lo)} to {pct(row400.fitted_hi)}).
          </p>
          <p>
            Stretching the rating scale from 400 to about {Math.round(scale)} and giving White a small
            first-move bonus fixes most of the gap. It beat the textbook formula on a locked test set that was opened
            once.
          </p>
        </div>

        <div className="stats">
          <Stat big={scale.toFixed(1)} label={`fitted scale s (textbook: 400). 95% range ${scaleCI[0].toFixed(1)} to ${scaleCI[1].toFixed(1)}`} tone="blue" />
          <Stat big={`${whiteAdv.toFixed(1)} pts`} label={`White's first-move bonus h. 95% range ${advCI[0].toFixed(1)} to ${advCI[1].toFixed(1)}`} />
          <Stat big={eloGap.toFixed(3)} label={`average gap of the textbook formula on the test set (0 is perfect)`} tone="red" />
          <Stat big={fitGap.toFixed(3)} label={`average gap of the fitted model on the test set`} tone="blue" />
        </div>
      </section>

      <section aria-labelledby="try-it">
        <h2 id="try-it">Try it: how often does the higher-rated player win?</h2>
        <p>
          Move the slider to change the rating gap. The gap is White's rating minus Black's. Most games are close:{' '}
          {pct(nearGames / rows.test, 0)} of test games are between players within 100 points of each other.
        </p>
        <div className="slider-box">
          <label htmlFor="gap">
            Rating gap: <strong>{gap > 0 ? `+${gap}` : gap}</strong>. {who}
          </label>
          <input
            id="gap"
            type="range"
            min={-400}
            max={400}
            step={10}
            value={gap}
            onChange={e => setGap(Number(e.target.value))}
          />
          <div className="readout" aria-live="polite">
            <div>
              <span className="big red">{pct(eloP)}</span>
              <span className="label">White's chance by the textbook formula</span>
            </div>
            <div>
              <span className="big blue">{pct(fitP)}</span>
              <span className="label">
                White's chance by the fitted model (95% range {pct(lo)} to {pct(hi)})
              </span>
            </div>
            <div>
              <span className="big">{diff > 0 ? '+' : ''}{diff.toFixed(1)} points</span>
              <span className="label">fitted minus textbook</span>
            </div>
          </div>
        </div>

        <div
          className="chart"
          role="img"
          aria-label="Line chart of White's chance of scoring against the rating gap, for the textbook formula and for the fitted model with its 95% range"
          style={{height: 400}}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={curve} margin={{top: 16, right: 24, bottom: 48, left: 24}}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="gap"
                domain={[-400, 400]}
                ticks={[-400, -200, 0, 200, 400]}
                tick={TICK}
                padding={{left: 20, right: 20}}
                label={{value: "Rating gap (White's rating minus Black's)", position: 'insideBottom', offset: -32, ...AXIS_LABEL}}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={v => pct(v, 0)}
                tick={TICK}
                width={64}
                label={{value: "White's chance", angle: -90, position: 'insideLeft', offset: 0, ...AXIS_LABEL}}
              />
              <Tooltip
                contentStyle={{fontSize: 16}}
                itemStyle={{fontSize: 16}}
                labelStyle={{fontSize: 16}}
                labelFormatter={g => `Gap ${g}`}
                formatter={(v: unknown) => (Array.isArray(v) ? `${pct(v[0])} to ${pct(v[1])}` : pct(Number(v)))}
              />
              <Legend verticalAlign="top" wrapperStyle={{fontSize: 16}} />
              <Area dataKey="band" name="Fitted model, 95% range" stroke="none" fill={BLUE} fillOpacity={0.2} />
              <Line dataKey="elo400" name="Textbook formula" stroke={RED} strokeWidth={3} dot={false} />
              <Line dataKey="fitted" name="Fitted model" stroke={BLUE} strokeWidth={3} dot={false} />
              <ReferenceLine x={gap} stroke={INK} strokeDasharray="5 4" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="caption">
          The textbook curve is steeper: it is more sure about favourites than real games allow. At a gap of 0 the fitted
          curve sits slightly above 50% because White moves first. The fitted model's 95% range is the pale blue band; it
          is only about half a percentage point wide, so it is hard to see.
        </p>
      </section>

      <section aria-labelledby="calibration">
        <h2 id="calibration">The calibration chart</h2>
        <p>
          A model is <strong>calibrated</strong> when, among games where it says "60%", White really scores about 60%.
          I sorted the {whole(rows.test)} locked test games into 20 bins by predicted chance, then plotted the actual score
          in each bin. Dots on the dashed grey line are perfectly calibrated. An <strong>overconfident</strong> model has
          dots below the line on the right (it predicted too high) and above the line on the left (it predicted too low).
        </p>
        <div className="toggles">
          <label>
            <input type="checkbox" checked={showElo} onChange={e => setShowElo(e.target.checked)} />
            Textbook formula (red triangles)
          </label>
          <label>
            <input type="checkbox" checked={showFit} onChange={e => setShowFit(e.target.checked)} />
            Fitted model (blue circles)
          </label>
        </div>
        <div
          className="chart"
          role="img"
          aria-label="Calibration chart on the locked test set. Predicted chance on the horizontal axis, actual score on the vertical axis, with 95% error bars. The textbook formula drifts away from the diagonal; the fitted model stays on it."
          style={{height: 500}}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{top: 16, right: 24, bottom: 48, left: 24}}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="predicted"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={v => pct(v, 0)}
                tick={TICK}
                padding={{left: 20, right: 20}}
                label={{value: "Predicted chance that White scores", position: 'insideBottom', offset: -32, ...AXIS_LABEL}}
              />
              <YAxis
                type="number"
                dataKey="actual"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={v => pct(v, 0)}
                tick={TICK}
                width={64}
                label={{value: "Actual score of White", angle: -90, position: 'insideLeft', offset: 0, ...AXIS_LABEL}}
              />
              <Tooltip content={<BinTip />} />
              <Legend verticalAlign="top" wrapperStyle={{fontSize: 16}} />
              <ReferenceLine segment={[{x: 0, y: 0}, {x: 1, y: 1}]} stroke="#5b6472" strokeDasharray="6 4" strokeWidth={2} />
              {showElo && (
                <Scatter name="Textbook formula" data={eloPoints} fill={RED} shape="triangle" isAnimationActive={false}>
                  <ErrorBar dataKey="err" direction="y" stroke={RED} strokeWidth={2} width={6} />
                </Scatter>
              )}
              {showFit && (
                <Scatter name="Fitted model" data={fitPoints} fill={BLUE} shape="circle" isAnimationActive={false}>
                  <ErrorBar dataKey="err" direction="y" stroke={BLUE} strokeWidth={2} width={6} />
                </Scatter>
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="caption">
          Bars show 95% ranges. A draw counts as half a win. The red triangles lie above the line on the left and below it on
          the right: the textbook formula puts its favourites too high and its underdogs too low.
        </p>
        <details>
          <summary>Show the numbers behind the chart</summary>
          <div className="table-wrap">
            <table>
              <caption>Calibration bins on the locked test set</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Predicted</th>
                  <th scope="col">Actual</th>
                  <th scope="col">95% range</th>
                  <th scope="col">Games</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Textbook', eloBins],
                  ['Fitted', fitBins],
                ].flatMap(([name, bins]) =>
                  (bins as Bin[]).map(b => (
                    <tr key={`${name}-${b.predicted}`}>
                      <td>{name as string}</td>
                      <td>{pct(b.predicted)}</td>
                      <td>{pct(b.actual)}</td>
                      <td>
                        {pct(b.lo)} to {pct(b.hi)}
                      </td>
                      <td>{whole(b.n)}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      <section aria-labelledby="model">
        <h2 id="model">The model: a story with two parameters</h2>
        <p>
          <strong>The story.</strong> Each game is a weighted coin flip. The weight depends only on the rating gap. White
          scores (a win is 1, a draw is half) with a chance that rises smoothly from 0% to 100% as White's rating gets higher
          than Black's.
        </p>
        <div className="formula" role="img" aria-label="p equals 1 divided by 1 plus 10 to the power of minus g plus h, all over s">
          <var>p</var> = 1 / (1 + 10<sup>−(<var>g</var> + <var>h</var>) / <var>s</var></sup>)
        </div>
        <ul className="tight">
          <li>
            <var>g</var> is the rating gap, White's rating minus Black's. It comes from the data.
          </li>
          <li>
            <var>s</var> is the <strong>scale</strong>, a parameter. The textbook Elo formula fixes it at 400. A bigger scale
            means a rating gap counts for less.
          </li>
          <li>
            <var>h</var> is White's first-move bonus in rating points, a parameter. The textbook formula fixes it at 0.
          </li>
        </ul>
        <p>
          <strong>Finding the parameters.</strong> I chose the values of <var>s</var> and <var>h</var> that gave the highest
          score on the training games. The score is the log₁₀ likelihood per game, with a draw counted as half a win, and
          higher is better.
        </p>
        <details>
          <summary>The same model in the course's words</summary>
          <p>
            This is logistic regression with one input. White's chance is σ(<var>w</var> · <var>g</var> + <var>b</var>), where
            the weight is <var>w</var> = ln(10) / <var>s</var> = {(LN10 / scale).toFixed(5)} per rating point, and the bias is{' '}
            <var>b</var> = ln(10) · <var>h</var> / <var>s</var> = {((LN10 * whiteAdv) / scale).toFixed(4)}. The textbook
            formula is the same model with the parameters fixed at <var>s</var> = 400 and <var>h</var> = 0.
          </p>
        </details>
      </section>

      <section aria-labelledby="data">
        <h2 id="data">The data</h2>
        <ul className="tight">
          <li>
            Source: the Lichess open database, the file for February 2017 (standard rated games). It has{' '}
            {whole(counts.games_seen)} games.
          </li>
          <li>
            I took a random 10% sample ({whole(sampled)} games), then kept only finished <strong>blitz</strong> games where
            both ratings were plain numbers and there was a result. That left {whole(counts.kept)} games.
          </li>
          <li>
            Dropped from the sample: {whole(counts.dropped_other_time_control)} games in other time controls,{' '}
            {whole(counts.dropped_no_time_control)} with no time control, {whole(counts.dropped_termination)} that ended
            without a normal finish (for example abandoned), and {whole(counts.dropped_no_result)} with no result.
          </li>
          <li>
            Each game gives two ratings from before the game, and the result. The rating gap is White's rating minus Black's.
          </li>
        </ul>
      </section>

      <section aria-labelledby="testing">
        <h2 id="testing">How it was tested</h2>
        <p>
          I split the games by date, so the model is always judged on games played after the ones it learned from:
        </p>
        <ul className="tight">
          <li>
            <strong>Training</strong>, up to 14 February: {whole(rows.train)} games. The parameters were fitted here.
          </li>
          <li>
            <strong>Validation</strong>, 15 to 21 February: {whole(rows.val)} games. Used to check the fit while I worked.
          </li>
          <li>
            <strong>Locked test set</strong>, 22 to 28 February: {whole(rows.test)} games. Locked away, then opened{' '}
            <strong>once</strong>, at the end, with the parameters already frozen.
          </li>
        </ul>
        <details>
          <summary>Proof the test set was not changed</summary>
          <p>
            When the test file was made, its SHA-256 fingerprint was saved. The one-time test run checks the file against
            it first, and refuses to run if it differs or if the test set was already opened. Fingerprint:{' '}
            <code style={{wordBreak: 'break-all'}}>{data.data.test_sha256}</code>
          </p>
        </details>

        <div className="table-wrap">
          <table>
            <caption>Results on the locked test set ({whole(rows.test)} games)</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Scale s</th>
                <th scope="col">White bonus h</th>
                <th scope="col">Score per game (higher is better)</th>
                <th scope="col">Average gap (lower is better)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Textbook formula</td>
                <td>{elo.scale.toFixed(1)}</td>
                <td>{elo.white_advantage.toFixed(1)}</td>
                <td>{scorePerGame(elo.log_loss).toFixed(5)}</td>
                <td>{eloGap.toFixed(4)}</td>
              </tr>
              <tr>
                <td>Textbook scale, fitted White bonus</td>
                <td>{eloWhite.scale.toFixed(1)}</td>
                <td>{eloWhite.white_advantage.toFixed(1)}</td>
                <td>{scorePerGame(eloWhite.log_loss).toFixed(5)}</td>
                <td>not computed</td>
              </tr>
              <tr>
                <td>Fitted model</td>
                <td>{fit.scale.toFixed(1)}</td>
                <td>{fit.white_advantage.toFixed(1)}</td>
                <td>{scorePerGame(fit.log_loss).toFixed(5)}</td>
                <td>{fitGap.toFixed(4)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          The average gap is how far, on average, the predicted chance is from the actual win rate, with bigger bins counting
          more. 0 would be perfectly calibrated.
        </p>
      </section>

      <section aria-labelledby="results">
        <h2 id="results">Results, and how sure I am</h2>
        <div className="answer">
          <p>
            On the locked test set the fitted model's score per game was higher than the textbook formula's by{' '}
            <strong>{scoreGain.toFixed(5)}</strong> (95% range {scoreGainLo.toFixed(5)} to {scoreGainHi.toFixed(5)}). The
            range does not include 0.
          </p>
          <p>
            In plain loss units that is {improvement.logloss.mean.toFixed(5)} ({improvement.logloss.ci95[0].toFixed(5)} to{' '}
            {improvement.logloss.ci95[1].toFixed(5)}). On the validation games it was{' '}
            {data.val_improvement_over_elo400.logloss.mean.toFixed(5)}, so the result held up on new games.
          </p>
        </div>
        <p>
          The ranges come from re-fitting the model {data.fitted.bootstrap_reps} times, each time on a random re-draw of the{' '}
          players, so that one very active player does not count as many independent games. The improvement is real, but small:
          at rating gaps of 100 to 400 points the two models differ by about 1 to 6 percentage points, and for the many games
          between players of similar rating they hardly differ at all.
        </p>
        <div className="table-wrap">
          <table>
            <caption>Average White score by rating level on the test set</caption>
            <thead>
              <tr>
                <th scope="col">Average rating of the two players</th>
                <th scope="col">Games</th>
                <th scope="col">Actual</th>
                <th scope="col">Textbook predicts</th>
                <th scope="col">Fitted predicts</th>
              </tr>
            </thead>
            <tbody>
              {data.rating_bands_test.map(b => (
                <tr key={b.band}>
                  <td>{b.band}</td>
                  <td>{whole(b.n)}</td>
                  <td>{pct(b.actual, 2)}</td>
                  <td>{pct(b.elo400, 2)}</td>
                  <td>{pct(b.fitted, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          The fitted model's average prediction is within about 0.2 percentage points of the truth at every rating level.
          That is consistent with White's first-move bonus being about the same for weak and strong players, but it does not
          prove it.
        </p>

        <h3>What this does not show, and what went wrong</h3>
        <div className="warn">
          <ul className="tight">
            <li>
              <strong>The scale of {scale.toFixed(0)} is not a pure measure of skill.</strong> A draw counts as half a win,
              which pulls results toward 50% and makes the scale look bigger. Noisy ratings do the same. I did not fit a
              win/draw/loss model, so I cannot say how much of the stretch comes from each cause.
            </li>
            <li>
              <strong>Lichess uses Glicko-2, not plain Elo.</strong> So the formula with a scale of 400 is a textbook baseline,
              not Lichess's own formula.
            </li>
            <li>
              <strong>The ranges are a little too narrow.</strong> I grouped games by the White player only, so a player's games
              as Black still count as independent.
            </li>
            <li>
              <strong>One month, blitz only, a 10% sample.</strong> I have not checked other months or other time controls.
            </li>
            <li>
              <strong>New accounts are in the data.</strong> The database does not mark uncertain (provisional) ratings, so I
              could not remove them.
            </li>
            <li>
              <strong>My first split was wrong.</strong> It put every game in the test file, because the dump contains a few
              games from 31 January and my script guessed the wrong month. It showed up in the counts, and I fixed it with
              explicit dates and re-ran before looking at any results.
            </li>
          </ul>
        </div>
        <p>
          <strong>What I would try next:</strong> a win/draw/loss model, to separate the effect of draws from the effect of
          noisy ratings, and the same test on other months and time controls.
        </p>
      </section>

      {hasGuess && (
        <section aria-labelledby="guess">
          <h2 id="guess">My guess before looking at the data</h2>
          <p>{guess.text}</p>
          {guess.confidence !== null && <p>My confidence: {guess.confidence}%.</p>}
          {guess.verdict.trim() !== '' && <p>{guess.verdict}</p>}
        </section>
      )}

      <section aria-labelledby="ai">
        <h2 id="ai">How I used AI</h2>
        <p>
          I used Claude (made by Anthropic). Claude wrote the Python scripts that filter the games, fit the model and compute the
          ranges, and wrote the code for this page. I chose the project idea, chose the Lichess file and the blitz games, decided
          to leave draws as a stated limitation, and ran every script on my own computer on the real Lichess file. Every number
          on this page is printed by those runs.
        </p>
      </section>

      <p className="source-line">
        Data: Lichess open database, February 2017 standard rated games,{' '}
        <a href="https://database.lichess.org">database.lichess.org</a> (CC0). No simulated data is used.
      </p>
    </main>
  );
}
