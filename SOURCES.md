# Sources

List every dataset and outside source here: its name, where you got it (a
link), and its license. Your agent should add to this as you go.

| Data | Where it came from | License |
| --- | --- | --- |
| Lichess standard rated games, February 2017 (`lichess_db_standard_rated_2017-02.pgn.zst`) | Lichess open database, https://database.lichess.org. A 10% random sample of blitz games was used. The raw file is not uploaded. | CC0 (the database page says exports are released under Creative Commons CC0) |
| app/chess-site-data.json | Made from the Lichess file above by my own scripts (`lichess_split.py`, `fit_calibration.py`, `bootstrap_ci.py`, `export_site_data.py`). Real results, not simulated. | derived from CC0 data |
| app/test-improvement.json | The improvement over Elo-400 printed by `fit_calibration.py --final` on 22 November 2026 (saved in `test_results.txt`). Real results, not simulated. | derived from CC0 data |
