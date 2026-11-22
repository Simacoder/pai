# AI notes

A running list of what the AI (Claude, made by Anthropic) did and what I did or decided.

## What Claude did

- Suggested a plan for the chess question: the model's story (a logistic formula with a scale and a White bonus), the
  data filters, a split by date into training, validation and a locked test set, and how to measure calibration.
- Wrote the Python scripts: `lichess_split.py` (stream the Lichess file, filter, split, lock the test set with a
  SHA-256 hash), `fit_calibration.py` (fit the parameters, calibration plots, one-time test run), `bootstrap_ci.py`
  (95% ranges by re-drawing players) and `export_site_data.py` (the numbers for this website).
- Tested those scripts on made-up games with a known answer before they were used on real data. This found and fixed a
  bug where the White-bonus range had zero width.
- Made the thumbnail picture.
- Wrote `app/App.tsx`, edited `app/App.css`, and checked the site with `node tools/check.mjs` and in a browser
  (text size, keyboard use of the slider, how the charts look).
- Looked up the licence of the Lichess data.
- Helped read the results. The reading of "why is the scale bigger than 400" is left as open possibilities, not a
  conclusion.

## What I did or decided

- Chose the project idea (are chess ratings calibrated?) from the course's list, and the Lichess open database.
- Downloaded the February 2017 file, used blitz games, and ran every script on my own computer. The numbers on the
  page are the output of those runs.
- Found that my first split put every game in the test file (the file has a few games from 31 January), and re-ran it
  with explicit dates before looking at any results.
- Decided to skip the win/draw/loss model and state draws as a limitation instead.
- Decided to open the locked test set once, at the end, and did.

## Still to do by me

- Fill in `app/my-guess.json` with the guess I wrote down BEFORE looking at the data (the page shows that section only
  when the text is not empty).
- Read the page and change any sentence that does not say what I mean.
