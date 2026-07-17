import pytest

from driveintent.analytics.analytics import sample_size_two_proportions, simulate_experiment


def test_sample_size_decreases_for_larger_effect():
    small_effect = sample_size_two_proportions(0.05, 0.10)
    large_effect = sample_size_two_proportions(0.05, 0.25)
    assert small_effect > large_effect > 0


@pytest.mark.parametrize("baseline,mde", [(0, 0.1), (1, 0.1), (0.5, 0), (0.9, 0.2)])
def test_sample_size_rejects_invalid_rates(baseline, mde):
    with pytest.raises(ValueError):
        sample_size_two_proportions(baseline, mde)


def test_experiment_simulation_is_reproducible_and_reports_real_power(cfg):
    first = simulate_experiment(cfg, n_per_arm=2000, baseline=0.05, true_lift=0.15, seed=7)
    second = simulate_experiment(cfg, n_per_arm=2000, baseline=0.05, true_lift=0.15, seed=7)
    assert first == second
    assert 0 <= first["approx_power"] <= 1
    assert first["randomization_unit"].startswith("user_id")


def test_experiment_power_increases_with_sample_size(cfg):
    low = simulate_experiment(cfg, n_per_arm=500, baseline=0.05, true_lift=0.15, seed=1)
    high = simulate_experiment(cfg, n_per_arm=5000, baseline=0.05, true_lift=0.15, seed=1)
    assert high["approx_power"] > low["approx_power"]
