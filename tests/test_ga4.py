import pandas as pd
import pytest

from driveintent.data.ga4 import GA4ContractError, transform_flattened_ga4


def _sample():
    return pd.DataFrame({
        "event_timestamp": ["2025-01-01T10:00:00Z", "2025-01-01T10:01:00Z",
                            "2025-01-02T12:00:00Z"],
        "event_name": ["session_start", "view_item", "custom_noise"],
        "user_pseudo_id": ["anon-1"] * 3, "ga_session_id": [10, 10, 11],
        "item_id": [None, "CAR_00001", None], "traffic_source": ["google"] * 3,
        "traffic_medium": ["organic"] * 3, "engagement_time_msec": [0, 2500, 10],
    })


def test_ga4_adapter_maps_identity_session_and_units():
    events, sessions = transform_flattened_ga4(_sample())
    assert list(events["event_name"]) == ["session_start", "view_item"]
    assert events["event_id"].is_unique and events["engagement_time_seconds"].iloc[1] == 2.5
    assert len(sessions) == 1 and sessions["session_sequence_number"].iloc[0] == 1


def test_ga4_adapter_can_fail_on_unknown_taxonomy():
    with pytest.raises(GA4ContractError, match="unmapped"):
        transform_flattened_ga4(_sample(), unknown_events="error")


def test_ga4_adapter_rejects_missing_contract_fields():
    with pytest.raises(GA4ContractError, match="required"):
        transform_flattened_ga4(pd.DataFrame({"event_name": ["view_item"]}))
