import pandas as pd

from driveintent.data.generate import build_all


def test_reproducible_under_fixed_seed(cfg):
    a = build_all(cfg, small=True)
    b = build_all(cfg, small=True)
    for table in a:
        pd.testing.assert_frame_equal(a[table], b[table])


def test_keys_unique(tables):
    assert tables["cars"]["car_id"].is_unique
    assert tables["users"]["user_id"].is_unique
    assert tables["events"]["event_id"].is_unique
    assert tables["sessions"]["session_id"].is_unique


def test_price_relationships(tables):
    cars = tables["cars"].copy()
    cars["age"] = 2026 - cars["manufacturing_year"]
    young = cars[cars["age"] <= 3]["listed_price"].median()
    old = cars[cars["age"] >= 8]["listed_price"].median()
    assert young > old, "newer cars should be more expensive on average"
    sold = cars[cars["sold_flag"]]
    assert (sold["transaction_price"] <= sold["listed_price"] * 1.001).mean() > 0.9


def test_position_bias_in_clicks(tables):
    imp = tables["impressions"]
    ctr = imp.groupby("list_position")["clicked"].mean()
    assert ctr.loc[1] > ctr.loc[10], "top positions must attract more clicks"


def test_funnel_consistency(tables):
    ev = tables["events"]
    counts = ev["event_name"].value_counts()
    assert counts["view_item"] >= counts.get("booking_complete", 0)
    assert counts.get("booking_complete", 0) >= counts.get("purchase", 0)


def test_sessions_follow_signup_and_chronological_sequence(tables):
    sessions = tables["sessions"].merge(
        tables["users"][["user_id", "signup_date"]], on="user_id", validate="many_to_one"
    )
    assert (sessions["session_start"] >= pd.to_datetime(sessions["signup_date"])).all()
    ordered = sessions.sort_values(["user_id", "session_start"])
    expected = ordered.groupby("user_id").cumcount() + 1
    assert ordered["session_sequence_number"].eq(expected).all()


def test_purchase_events_match_inventory(tables):
    purchases = tables["events"].query("event_name == 'purchase'").merge(
        tables["cars"][["car_id", "inventory_exit_date"]], on="car_id", validate="one_to_one"
    )
    assert pd.to_datetime(purchases["event_timestamp"]).dt.normalize().eq(
        pd.to_datetime(purchases["inventory_exit_date"])
    ).all()


def test_target_rates_reasonable(tables):
    imp = tables["impressions"]
    assert 0.002 < imp["booked"].mean() < 0.20
    assert 0.05 < imp["clicked"].mean() < 0.6
