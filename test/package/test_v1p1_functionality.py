import pandas as pd
from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier


def test_tabstar_v1p1_classifier_fit_predict():
    features = pd.DataFrame({'feature_one': [1, 2, 3, 4]*10,
                             'feature_two': [0, 1, 0, 1]*10})
    targets = pd.Series(['no', 'yes', 'no', 'yes']*10)

    classifier = TabSTARv1p1Classifier(max_epochs=1)
    assert classifier.fit(features, targets) is classifier

    predictions = classifier.predict(features)
    assert len(predictions) == len(targets)
    assert set(predictions) <= {'no', 'yes'}
    assert classifier.predict_proba(features).shape == (len(targets), 2)
    assert isinstance(classifier.score(features, targets), float)
