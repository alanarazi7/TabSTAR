from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.datasets.downloading import get_dataset_from_arg


def test_enum_identity():
    # value = OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT.value
    reconstructed = get_dataset_from_arg(46651)

    assert isinstance(reconstructed, OpenMLDatasetID), "Reconstructed is not an instance of OpenMLDatasetID"
    assert reconstructed == OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT, "Enum value mismatch"

    print("✅ Enum identity and isinstance checks passed.")