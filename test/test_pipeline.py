import pytest
from mlops_pipeline import preprocess_data

def test_preprocess_data():
    raw_data = ["This is a [test] sample!", "Another example..."]
    processed = preprocess_data(raw_data)
    assert len(processed) == len(raw_data)
    assert processed[0] == "This is a test sample"
    assert processed[1] == "Another example"
