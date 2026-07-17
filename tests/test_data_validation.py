import pytest

from driveintent.data.validate import ValidationError, validate_all, validate_cars, validate_feature_list


def test_valid_data_passes(tables):
    assert validate_all(tables, raise_on_error=False) == []


def test_malformed_cars_fail(tables):
    bad = tables["cars"].copy()
    bad.loc[bad.index[0], "listed_price"] = -5
    bad.loc[bad.index[1], "registration_year"] = 1990
    errs = validate_cars(bad)
    assert any("listed_price" in e for e in errs)
    assert any("registration_year" in e for e in errs)


def test_unsold_with_price_fails(tables):
    bad = tables["cars"].copy()
    idx = bad[~bad["sold_flag"]].index
    if len(idx):
        bad.loc[idx[0], "transaction_price"] = 100000
        assert validate_cars(bad)


def test_validate_all_raises(tables):
    bad = dict(tables)
    cars = bad["cars"].copy()
    cars.loc[cars.index[0], "inspection_score"] = 250
    bad["cars"] = cars
    with pytest.raises(ValidationError):
        validate_all(bad)


def test_leakage_feature_detection():
    assert validate_feature_list(["make", "transaction_price"])
    assert validate_feature_list(["make", "kilometres_driven"]) == []
