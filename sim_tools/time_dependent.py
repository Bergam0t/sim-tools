"""
Classes and functions to support time dependent samplingm in DES models.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.random import SeedSequence
import pandas as pd
import matplotlib.pyplot as plt


# pylint: disable=too-few-public-methods
class NSPPThinning:
    """
    Non Stationary Poisson Process via Thinning.

    Thinning is an acceptance-rejection approach to sampling
    inter-arrival times (IAT) from a time-dependent distribution
    where each time period follows its own exponential distribution.

    This implementation takes mean inter-arrival times as inputs, making it
    consistent with NumPy's exponential distribution parameterization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        interval_width: Optional[float] = None,
        random_seed1: Optional[int | SeedSequence] = None,
        random_seed2: Optional[int | SeedSequence] = None,
    ):
        """
        Non Stationary Poisson Process via Thinning.

        Time dependency is andled for a single table
        consisting of equally spaced intervals.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with time points and mean inter-arrival times.
            Columns should be "t" and "mean_iat" respectively.

        interval_width: float, optional (default=None)
            The width of each time interval. If None, it will be calculated
            from consecutive time points in the data. Required if data has only
            one row.

        random_seed1: int | SeedSequence, optional (default=None)
            Random seed for the exponential distribution

        random_seed2: int | SeedSequence, optional (default=None)
            Random seed for the uniform distribution used
            for acceptance/rejection sampling.
        """
        self.data = data
        self.arr_rng = np.random.default_rng(random_seed1)
        self.thinning_rng = np.random.default_rng(random_seed2)

        # Find the minimum mean IAT (corresponds to the maximum arrival rate)
        self.min_iat = data["mean_iat"].min()

        if self.min_iat <= 0:
            raise ValueError("Mean inter-arrival times must be positive")

        # Use provided interval width or calculate from data
        if interval_width is not None:
            self.interval = interval_width
        elif len(data) > 1:
            # Calculate from data (assumes all intervals are equal in length)
            self.interval = data.iloc[1]["t"] - data.iloc[0]["t"]
        else:
            raise ValueError(
                "With only one data point, interval_width must be provided"
            )

        self.rejects_last_sample = None

    def __repr__(self):
        """Return a string representation of the NSPPThinning instance."""
        # Truncate the data representation if too long
        max_len = 100
        data_str = repr(self.data)
        if len(data_str) > max_len:
            data_str = data_str[:max_len] + "..."

        # Return class name with both data and interval information
        return (
            f"{self.__class__.__name__}(data={data_str}, " +
            f"interval={self.interval})"
        )

    def sample(self, simulation_time: float) -> float:
        """
        Run a single iteration of acceptance-rejection
        thinning alg to sample the next inter-arrival time

        Parameters
        ----------
        simulation_time: float
            The current simulation time. This is used to look up
            the mean IAT for the time period.

        Returns
        -------
        float
            The inter-arrival time
        """

        # this gives us the index of dataframe to use
        t = int(simulation_time // self.interval) % len(self.data)
        mean_iat_t = self.data["mean_iat"].iloc[t]

        # set to a large number so that at least 1 sample taken!
        u = np.inf

        # included for audit and tracking purposes.
        self.rejects_last_sample = 0

        interarrival_time = 0.0

        # We accept the sample if u < (min_iat / mean_iat_t)
        # This is equivalent to the original u < (lambda_t / lambda_max)
        # since lambda = 1/mean_iat
        while u >= (self.min_iat / mean_iat_t):
            self.rejects_last_sample += 1
            interarrival_time += self.arr_rng.exponential(self.min_iat)
            u = self.thinning_rng.uniform(0.0, 1.0)

        return interarrival_time


def nspp_simulation(
    arrival_profile: pd.DataFrame,
    run_length: Optional[float] = None,
    n_reps: Optional[int] = 1000,
) -> pd.DataFrame:
    """
    Generate a pandas dataframe that contains multiple replications of
    a non-stationary poisson process for the set arrival profile.

    This uses the sim-tools NSPPThinning class.

    Useful for validating the the NSPP has been set up correctly and is
    producing the desired profile for the simulation model.

    On each replication the function counts the number of arrivals during the
    intervals from the arrival profile.  Returns a data frame with reps (rows)
    and interval arrivals (columns).

    Parameters
    ----------
    arrival_profile: pandas.DataFrame
        The arrival profile is a pandas data frame containing 't',
        'arrival_rate' and 'mean_iat' columns.

    run_length: float, optional (default=None)
        How long should the simulation be run. If none then uses the last
        value in 't' + the interval (assumes equal width intervals)

    n_reps: int, optional (default=1000)
        The number of replications to run.

    Returns
    -------
    pd.DataFrame.


    """
    # replication results
    replication_results = []

    # multiple replications
    for rep in range(n_reps):

        # method for producing n non-overlapping streams
        seed_sequence = np.random.SeedSequence(rep)

        # Generate n high quality child seeds
        seeds = seed_sequence.spawn(2)

        # create nspp
        nspp_rng = NSPPThinning(
            data=arrival_profile, random_seed1=seeds[0], random_seed2=seeds[1]
        )

        # if no run length has been set....
        if run_length is None:
            run_length = (
                arrival_profile["t"].iloc[len(arrival_profile) - 1] +
                nspp_rng.interval
            )

        # list - each item is an interval in the arrival profile
        interval_samples = [0] * arrival_profile.shape[0]
        simulation_time = 0.0
        while simulation_time < run_length:
            iat = nspp_rng.sample(simulation_time)
            simulation_time += iat

            if simulation_time < run_length:
                # data collection: add one to count for hour of the day
                # note list NSPPThinning this assume equal intervals
                interval_of_day = (
                    int(simulation_time // nspp_rng.interval) %
                    len(arrival_profile)
                )
                interval_samples[interval_of_day] += 1

        replication_results.append(interval_samples)

    # produce summary chart of arrivals per interval
    # format in a dataframe
    df_replications = pd.DataFrame(replication_results)
    df_replications.index = np.arange(1, len(df_replications) + 1)
    df_replications.index.name = "rep"

    return df_replications


def nspp_plot(
    arrival_profile: pd.DataFrame,
    run_length: Optional[float] = None,
    n_reps: Optional[int] = 1000,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a matplotlib chart to visualise a non-stationary poisson process
    for the set arrival profile.

    This uses the sim-tools NSPPThinning class.

    Useful for validating the the NSPP has been set up correctly and is
    producing the desired profile for the simulation model.

    Parameters
    ----------
    arrival_profile: pandas.DataFrame
        The arrival profile is a pandas data frame containing 't',
        'arrival_rate' and 'mean_iat' columns.

    run_length: float, optional (default=None)
        How long should the simulation be run. If none then uses the last value
        in 't' + the interval (assumes equal width intervals)

    n_reps: int, optional (default=1000)
        The number of replications to run.
    """

    # verification of arrival_profile

    # is it a dataframe
    if not isinstance(arrival_profile, pd.DataFrame):
        raise ValueError(
            "arrival_profile expected pd.DataFrame " +
            f"got {type(arrival_profile)}"
        )

    # all columns are present
    required_columns = ["t", "arrival_rate", "mean_iat"]
    for col in required_columns:
        if col not in arrival_profile.columns:
            raise ValueError(
                f"arrival_profile must contain "
                f"the following columns: {required_columns}. "
            )

    # generate the sample data
    df_interval_results = nspp_simulation(arrival_profile, run_length, n_reps)

    interval_means = df_interval_results.mean(axis=0)
    interval_sd = df_interval_results.std(axis=0)

    upper = interval_means + interval_sd
    lower = interval_means - interval_sd
    lower[lower < 0] = 0

    # visualise
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot()

    # plot in this case returns a 2D line plot object
    _ = ax.plot(arrival_profile["t"], interval_means, label="Mean")
    _ = ax.fill_between(
        arrival_profile["t"], lower, upper, alpha=0.2, label="+-1SD"
    )

    # chart appearance
    _ = ax.legend(loc="best", ncol=3)
    _ = ax.set_ylim(
        0,
    )
    _ = ax.set_xlim(0, arrival_profile.shape[0] - 1)
    _ = ax.set_ylabel("arrivals")
    _ = ax.set_xlabel("interval (from profile)")
    _ = plt.xticks(arrival_profile["t"])

    return fig, ax
