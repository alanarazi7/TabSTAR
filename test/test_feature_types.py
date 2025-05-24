from openml import OpenMLDataFeature

from tabular.preprocessing.objects import FeatureType


def test_feature_types_from_openml():
    for f in OpenMLDataFeature.LEGAL_DATA_TYPES:
        assert f in FeatureType.__members__.values(), f"FeatureType missing {f}"
