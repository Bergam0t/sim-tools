"""
basic smoke tests
Create objects to check all okay
"""

import numpy as np
import pytest

import sim_tools.distributions as dists


SEED_1 = 42


@pytest.mark.parametrize("dist_class, args, expected_type", [
    (dists.Exponential, (10,), float),
    (dists.Lognormal, (10, 1), float),
    (dists.Normal, (10, 1), float),
    (dists.Uniform, (1, 10), float),
    (dists.Triangular, (1.0, 10.0, 25.0), float),
    (dists.Bernoulli, (0.3,), int),
    (dists.Erlang, (10.0, 2.8), float),
    (dists.ErlangK, (1, 2.8), float),
    (dists.Gamma, (1.2, 2.8), float),
    (dists.Weibull, (1.2, 2.8), float),
    (dists.Beta, (1.2, 2.8), float),
    (dists.PearsonV, (1.2, 2.8), float),
    (dists.PearsonVI, (1.2, 1.2, 2.8), float),
    (dists.Poisson, (5.5,), int),
    (dists.DiscreteEmpirical, ([1, 2, 3], [95, 3, 2]), int),
])
def test_distribution_sample_type(dist_class, args, expected_type):
    """
    Check that the distribution `sample` methods return the expected data type.
    """
    d = dist_class(*args, random_seed=SEED_1)
    assert isinstance(d.sample(), expected_type)


def test_grouped_continuous_empirical():
    """
    Check that the GroupedContinuousEmpirical `sample` method returns a float.
    """
    dist = dists.GroupedContinuousEmpirical(
        lower_bounds=[0, 5, 10, 15, 30, 45, 60, 120, 180, 240, 480],
        upper_bounds=[5, 10, 15, 30, 45, 60, 120, 180, 240, 480, 2880],
        freq=[34, 4, 8, 13, 15, 13, 19, 13, 9, 12, 73],
        random_seed=SEED_1,
    )
    assert isinstance(dist.sample(), float)


def test_combination():
    """
    Check that the CombinationDistribution `sample` method (combining
    Exponential and Normal) returns a float.
    """
    d_exp = dists.Exponential(10, random_seed=SEED_1)
    d_nor = dists.Normal(10, 1, random_seed=SEED_1)
    d = dists.CombinationDistribution(d_exp, d_nor)
    assert isinstance(d.sample(), float)


def test_truncated_type():
    """
    Check that the TruncatedDistribution `sample` method (applied to Normal)
    returns a float.
    """
    d1 = dists.Normal(10, 1, random_seed=SEED_1)
    d2 = dists.TruncatedDistribution(d1, lower_bound=10.0)
    assert isinstance(d2.sample(), float)


def test_fixed_type():
    """
    Check that the FixedDistribution `sample` method returns a float.
    """
    d = dists.FixedDistribution(5.0)
    assert isinstance(d.sample(), float)


def test_fixed_value():
    """
    Check that the FixedDistribution `sample` method returns the same value as
    input.
    """
    d = dists.FixedDistribution(5.0)
    assert d.sample() == 5.0


def test_continous_empirical_length():
    """
    Check that ContinuousEmpirical `sample` method returns the expected number
    of samples.
    """
    dist = dists.GroupedContinuousEmpirical(
        lower_bounds=[0, 5, 10, 15, 30, 45, 60, 120, 180, 240, 480],
        upper_bounds=[5, 10, 15, 30, 45, 60, 120, 180, 240, 480, 2880],
        freq=[34, 4, 8, 13, 15, 13, 19, 13, 9, 12, 73],
        random_seed=SEED_1,
    )
    expected_size = 10
    assert len(dist.sample(expected_size)) == expected_size


def test_discrete_multiple():
    """
    Check that Discrete `sample` method returns the expected number of samples.
    """
    d = dists.DiscreteEmpirical(values=[1, 2, 3], freq=[95, 3, 2], random_seed=SEED_1)
    assert len(d.sample(size=100)) == 100


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, 10.0),
        (10, 10.0),
        (100, 10.0),
        (10_000_000, 10.0),
        (10_000_000, 0.0),
        (10_000_000, 0.01),
    ],
)
def test_truncated_min(n, expected):
    """
    Check that samples from the TruncatedDistribution do not fall below the
    specified lower bound.
    """
    d1 = dists.Normal(10, 1, random_seed=SEED_1)
    d2 = dists.TruncatedDistribution(d1, lower_bound=expected)
    assert min(d2.sample(size=n)) >= expected


def test_grouped_continuous_empirical_mean_and_var():
    """Test that theorectical mean is approx = to sampled.
    """
    # Define test data
    lower_bounds = [0, 1, 2]
    upper_bounds = [1, 2, 3]
    freq = [10, 20, 30]
    
    # Create distribution with fixed random seed for reproducibility
    dist = dists.GroupedContinuousEmpirical(
        lower_bounds, upper_bounds, freq, random_seed=42)
    
    # Get theoretical mean
    theoretical_mean = dist.mean
    # Get theoretical variance
    theoretical_variance = dist.variance
    
    # Generate a large sample
    samples = dist.sample(500_000)
    
    # Calculate sample mean and var
    sample_mean = np.mean(samples)
    sample_variance = np.var(samples)
   
    # Assert that sample mean approximates theoretical mean
    assert sample_mean == pytest.approx(theoretical_mean, rel=1e-2)
    assert sample_variance == pytest.approx(theoretical_variance, rel=1e-2)


