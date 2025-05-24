from pandas import DataFrame

from tabular.tabstar.preprocessing.verbalization import verbalize_feature
from tabular.utils.utils import verbose_print


def verbalize_x_txt(x_txt: DataFrame):
    verbose_print(f"🔤 Verbalizing textual features")
    for col in x_txt.columns:
        x_txt[col] = x_txt[col].apply(lambda v: verbalize_feature(col, v))
    verbose_print(f"Done verbalizing!")
