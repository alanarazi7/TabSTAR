import importlib.resources

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARClassifier

print(f"🌟 Planning to finetune TabSTAR on a toy dataset")
with importlib.resources.files("tabstar.resources").joinpath("imdb.csv").open("r") as f:
    # This dataset was used for pretraining, the results are not representative.
    x = pd.read_csv(f)
y = x.pop('Genre_is_Drama')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
tabstar = TabSTARClassifier()
tabstar.fit(x_train, y_train)
y_pred = tabstar.predict(x_test)
print(classification_report(y_test, y_pred))