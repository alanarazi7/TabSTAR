import pandas as pd
from tabstar.tabstar_model import TabSTARClassifier

def test_tabstar_classifier_fit_predict():
    # Create simple input data
    features = pd.DataFrame({'feature_one': [1, 2, 3, 4]*10, 
                             'feature_two': [0, 1, 0, 1]*10})
    targets = pd.Series([0, 1, 0, 1]*10)

    # Instantiate and fit the classifier
    classifier = TabSTARClassifier()
    classifier.fit(features, targets)

    # Predict using the same features
    predictions = classifier.predict(features)
    assert len(predictions) == len(targets)