import pandas as pd
from pandas import DataFrame

from tabstar_paper.leaderboard.data.keys import DATASET, TEST_SCORE, MODEL
from tabstar_paper.utils.io_handlers import load_json_lines

lines = load_json_lines("results.jsonl")
df = DataFrame(lines)
pivot_df = df.pivot_table(index=[DATASET], columns=[MODEL, "augment"], values=[TEST_SCORE])
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)
print(pivot_df)