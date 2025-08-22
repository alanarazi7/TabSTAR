from dataclasses import dataclass, asdict

from pandas import DataFrame

from tabstar_paper.leaderboard.data.keys import DATASET_SIZE, IS_TUNED, MODEL, BASE_MODEL


@dataclass
class CI:
    avg: float
    low: float
    high: float

    @property
    def half(self) -> float:
        return self.high - self.avg

def get_var_ci(df: DataFrame, score_key: str) -> CI:
    mean_score = df[score_key].mean()
    std_score = df[score_key].std()
    ci = 1.96 * (std_score / len(df) ** 0.5)
    low = mean_score - ci
    high = mean_score + ci
    ci = CI(avg=mean_score, low=low, high=high)
    return ci


def df_ci_for_model(df: DataFrame, score_key: str) -> DataFrame:
    group_cols = [MODEL, DATASET_SIZE, BASE_MODEL, IS_TUNED]
    final_scores = []
    for keys, model_df in df.groupby(group_cols):
        ci = get_var_ci(model_df, score_key=score_key)
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = dict(zip(group_cols, keys))
        d.update(asdict(ci))
        final_scores.append(d)
    final_df = DataFrame(final_scores)
    final_df = final_df.sort_values(by='avg', ascending=False)
    return final_df


