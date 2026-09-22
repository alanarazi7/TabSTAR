# TabArena leakage analysis: results

Outputs of `tabarena_leakage_analysis.py` on all 51 TabArena v0.1 datasets, split r0f0,
87 methods (TabArena's 84 plus the three arms below). Boards were scored with TabArena's
own scorer (`bencheval.BenchmarkEvaluator`, Elo calibrated to `RF (default)` = 1000,
200 bootstrap rounds).

Arms (raw method keys in the CSVs, and the names used in the page):

| Raw key | Name | Checkpoint |
|---|---|---|
| `[Leakage] TabSTAR_base (default)` | TabSTAR-Default-NoBug-WithLeakage-400 | released `alana89/TabSTAR`, pretrained on all 400 datasets |
| `[Leakage] TabSTAR_correct (default)` | TabSTAR-Default-NoBug-NoLeakage | the 320-dataset fold checkpoint that excluded the dataset (fold k0 off corpus) |
| `[Leakage] TabSTAR_wrong (default)` | TabSTAR-Default-NoBug-WithLeakage-320 | a 320-dataset fold checkpoint that still included the dataset (in corpus only) |

Files:

- `results_per_split.csv`: one row per method and dataset with the metric error, times and the
  `imputed` flag (TabArena fills the 320 arm's 19 off-corpus rows from `RF (default)` here).
- `tabarena_leaderboard.csv`: TabArena's leaderboard of the run, 87 methods, RF fill.
- `per_type_boards_impute_400.csv`: boards for all tasks, classification, binary, multiclass and
  regression, with the 320 arm's 19 off-corpus rows filled from the 400 arm instead of RF.
- `per_type_boards_drop320.csv`: the same boards without the 320 arm, 86 methods, no filled rows.
- `per_dataset_leakage.csv`: per dataset, the rank of each arm among the 87 methods on the split
  and the gain of each leaky arm over NoLeakage in places, in relative error and in units of the
  board's error spread. The 19 off-corpus datasets serve as the leak-free control.
- `tabstar_leakage_boards.html`: the sortable boards page built from the files above.
