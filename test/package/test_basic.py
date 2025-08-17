import pandas as pd
from tabstar.tabstar_model import TabSTARClassifier

def test_tabstar_classifier_fit_predict():
    # Create dummy data
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])

    # Instantiate and fit
    clf = TabSTARClassifier()
    clf.fit(X, y)

    # Predict
    preds = clf.predict(X)
    assert len(preds) == len(y)