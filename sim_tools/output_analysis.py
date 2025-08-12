"""
module: output_analysis

Provides tools for selecting the number selecting the number of
replications to run with a Discrete-Event Simulation.

The Confidence Interval Method (tables and visualisation)

The Replications Algorithm (Hoad et al. 2010).
"""

import warnings
from typing import Protocol, runtime_checkable, Optional

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import t


OBSERVER_INTERFACE_ERROR = (
    "Observers of OnlineStatistics must implement "
    + "ReplicationObserver interface. i.e. "
    + "update(results: OnlineStatistics) -> None"
)

ALG_INTERFACE_ERROR = (
    "Parameter 'model' must implement "
    + "ReplicationsAlgorithmModelAdapter interface. i.e. "
    + "single_run(replication_no: int) -> float"
)


# pylint: disable=too-few-public-methods
@runtime_checkable
class ReplicationObserver(Protocol):
    """
    Interface for an observer of an instance of the ReplicationsAnalyser.
    """
    def update(self, results) -> None:
        """
        Add an observation of a replication

        Parameters
        -----------
        results: OnlineStatistic
            The current replication to observe.
        """


class OnlineStatistics:
    """
    Computes running sample mean and variance using Welford's algorithm.

    This is a robust and numerically stable approach first described in the
    1960s and popularised in Donald Knuth's *The Art of Computer Programming*
    (Vol. 2).

    The term *"online"* means each new data point is processed immediately
    to update statistics, without storing or reprocessing the entire dataset.

    This implementation additionally supports computation of:
      - Confidence intervals (CIs).
      - Percentage deviation of CI half-widths from the mean.

    Attributes
    ----------
    n : int
        Number of data points processed so far.
    x_i : float
        Most recent data point.
    mean : float
        Current running mean.
    _sq : float
        Sum of squared differences from the current mean (used for variance).
    alpha : float
        Significance level for confidence interval calculations
    observer : list
        Registered observers notified upon updates.
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        alpha: Optional[float] = 0.1,
        observer: Optional[ReplicationObserver] = None,
    ) -> None:
        """
        Initialise a new OnlineStatistics object.

        Parameters
        ----------
        data: np.ndarray, optional (default = None)
            Initial dataset to process.

        alpha: float, optional (default = 0.1)
            Significance level for confidence interval calculations
            (CI level = 100 * (1 - alpha) %).

        observer: ReplicationObserver, optional (default=None)
            A user may optionally track the updates to the statistics using a
            `ReplicationObserver` (e.g. `ReplicationTabuliser`). This allows
            further tabular or visual analysis or saving results to file if
            required.

        Raises
        ------
        ValueError
            If `data` is provided but is not a NumPy array.
        """

        self.n = 0
        self.x_i = None
        self.mean = None
        self._sq = None
        self.alpha = alpha
        self._observers = []
        if observer is not None:
            self.register_observer(observer)

        if data is not None:
            if isinstance(data, np.ndarray):
                for x in data:
                    self.update(x)
            # Raise an error if in different format - else will invisibly
            # proceed and won't notice it hasn't done this
            else:
                raise ValueError(
                    f"data must be np.ndarray but is type {type(data)}")

    def register_observer(self, observer: ReplicationObserver) -> None:
        """
        Register an observer to be notified on each statistics update.

        Parameters
        ----------
        observer : ReplicationObserver
            Object implementing the observer interface.

        Raises
        ------
        ValueError
            If `observer` is not an instance of ReplicationObserver.
        """
        if not isinstance(observer, ReplicationObserver):
            raise ValueError(OBSERVER_INTERFACE_ERROR)

        self._observers.append(observer)

    @property
    def variance(self) -> float:
        """
        Sample variance of the data.

        Returns
        -------
        float
            Sample variance, calculated as the sum of squared differences 
            from the mean divided by (n - 1).
        """
        return self._sq / (self.n - 1)

    @property
    def std(self) -> float:
        """
        Standard deviation of data.

        Returns
        -------
        float
            Standard deviation, or NaN if fewer than 3 points are available.
        """
        if self.n > 2:
            return np.sqrt(self.variance)
        return np.nan

    @property
    def std_error(self) -> float:
        """
        Standard error of the mean.

        Returns
        -------
        float
            Standard error, equal to `std / sqrt(n)`.
        """
        return self.std / np.sqrt(self.n)

    @property
    def half_width(self) -> float:
        """
        Half-width of the confidence interval.

        Returns
        -------
        float
            The margin of error for the confidence interval.
        """
        dof = self.n - 1
        t_value = t.ppf(1 - (self.alpha / 2), dof)
        return t_value * self.std_error

    @property
    def lci(self) -> float:
        """
        Lower bound of the confidence interval.

        Returns
        -------
        float
            Lower confidence limit, or NaN if fewer than 3 values are
            available.
        """
        if self.n > 2:
            return self.mean - self.half_width
        return np.nan

    @property
    def uci(self) -> float:
        """
        Upper bound of the confidence interval.

        Returns
        -------
        float
            Upper confidence limit, or NaN if fewer than 3 values are
            available.
        """
        if self.n > 2:
            return self.mean + self.half_width
        return np.nan

    @property
    def deviation(self) -> float:
        """
        Precision of the confidence interval expressed as the percentage
        deviation of the half width from the mean.

        Returns
        -------
        float
            CI half-width divided by the mean, or NaN if fewer than 3 values.
        """
        if self.n > 2:
            return self.half_width / self.mean
        return np.nan

    def update(self, x: float) -> None:
        """
        Update statistics with a new observation using Welford's algorithm.

        See Knuth. D `The Art of Computer Programming` Vol 2. 2nd ed. Page 216.

        Parameters
        ----------
        x : float
            New observation.
        """
        self.n += 1
        self.x_i = x
        # Initial statistics
        if self.n == 1:
            self.mean = x
            self._sq = 0
        else:
            # Updated statistics
            updated_mean = self.mean + ((x - self.mean) / self.n)
            self._sq += (x - self.mean) * (x - updated_mean)
            self.mean = updated_mean
        self.notify()

    def notify(self) -> None:
        """
        Notify all registered observers that an update has occurred.
        """
        for observer in self._observers:
            observer.update(self)


class ReplicationTabulizer:
    """
    Observer class for recording replication results from an 
    `OnlineStatistics` instance during simulation runs or repeated experiments.

    Implements the observer pattern to collect statistics after each update
    from the observed object, enabling later tabulation and analysis. After
    data collection, results can be exported as a summary dataframe (equivalent
    Implement as the part of observer pattern. Provides a summary frame
    to the output of `confidence_interval_method`).

    Attributes
    ----------
    stdev : list[float]
        Sequence of recorded standard deviations.
    lower : list[float]
        Sequence of recorded lower confidence interval bounds.
    upper : list[float]
        Sequence of recorded upper confidence interval bounds.
    dev : list[float]
        Sequence of recorded percentage deviations of CI half-width from the
        mean.
    cumulative_mean : list[float]
        Sequence of running mean values.
    x_i : list[float]
        Sequence of last observed raw data points.
    n : int
        Total number of updates recorded.
    """

    def __init__(self):
        """
        Initialise an empty `ReplicationTabulizer`.

        All recorded metrics are stored in parallel lists, which grow as
        `update()` is called.
        """
        self.stdev = []
        self.lower = []
        self.upper = []
        self.dev = []
        self.cumulative_mean = []
        self.x_i = []
        self.n = 0

    def update(self, results: OnlineStatistics) -> None:
        """
        Record the latest statistics from an observed `OnlineStatistics`
        instance.

        This method should be called by the observed object when its state
        changes (i.e., when a new data point has been processed).

        Parameters
        ----------
        results : OnlineStatistics
            The current statistics object containing the latest values.
        """
        self.x_i.append(results.x_i)
        self.cumulative_mean.append(results.mean)
        self.stdev.append(results.std)
        self.lower.append(results.lci)
        self.upper.append(results.uci)
        self.dev.append(results.deviation)
        self.n += 1

    def summary_table(self) -> pd.DataFrame:
        """
        Compile all recorded replications into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A table with one row per replication (update), containing:
            - `Mean` (latest observed value)
            - `Cumulative Mean`
            - `Standard Deviation`
            - `Lower Interval`
            - `Upper Interval`
            - `% deviation` (CI half-width as a fraction of cumulative mean)
        """
        # combine results into a single dataframe
        results = pd.DataFrame(
            [
                self.x_i,
                self.cumulative_mean,
                self.stdev,
                self.lower,
                self.upper,
                self.dev,
            ]
        ).T
        results.columns = [
            "Mean",
            "Cumulative Mean",
            "Standard Deviation",
            "Lower Interval",
            "Upper Interval",
            "% deviation",
        ]
        results.index = np.arange(1, self.n + 1)
        results.index.name = "replications"

        return results


def confidence_interval_method(
    replications,
    alpha: Optional[float] = 0.05,
    desired_precision: Optional[float] = 0.05,
    min_rep: Optional[int] = 5,
    decimal_places: Optional[int] = 2,
):
    """
    Determine the minimum number of simulation replications required to achieve
    a target precision in the confidence interval of a performance metric.

    This function applies the **confidence interval method**: it identifies the
    smallest replication count where the relative half-width of the confidence
    interval is less than the specified `desired_precision`.

    Parameters
    ----------
    replications: arraylike
        Array (e.g. np.ndarray or list) of replication results for a
        performance metric.
    alpha: float, optional (default=0.05)
        Significance level for confidence interval calculations
        (CI level = 100 * (1 - alpha) %).
    desired_precision: float, optional (default=0.05)
        Target CI half-width precision (i.e. percentage deviation of the
        confidence interval from the mean).
    min_rep: int, optional (default=5)
        Minimum number of replications to consider before evaluating precision.
        Helps avoid unstable early results.
    decimal_places: int, optional (default=2)
        Number of decimal places to round values in the returned results table.

    Returns
    -------
    tuple of (int, pandas.DataFrame)
        - **n_reps** : int
          The smallest number of replications achieving the desired precision.
          Returns -1 if target precision is never reached.
        - **results** : pandas.DataFrame
          Summary statistics at each replication:
          `"Mean"`, `"Cumulative Mean"`, `"Standard Deviation"`,
          `"Lower Interval"`, `"Upper Interval"`, `"% deviation"`.

    Warns
    -----
    UserWarning
        If the desired precision is not achieved for any replication.
    """
    # Set up method for calculating statistics
    observer = ReplicationTabulizer()
    stats = OnlineStatistics(
        alpha=alpha, data=np.array(replications[:2]), observer=observer)

    # Calculate statistics with each replication
    for i in range(2, len(replications)):
        stats.update(replications[i])

    results = observer.summary_table()

    # Find minimum number of replications where deviation is below target
    try:
        n_reps = (
            results.iloc[min_rep:]
            .loc[results["% deviation"] <= desired_precision]
            .iloc[0]
            .name
        )
    except IndexError:
        message = "WARNING: the replications do not reach desired precision"
        warnings.warn(message)
        n_reps = -1

    return n_reps, results.round(decimal_places)


def plotly_confidence_interval_method(
    n_reps, conf_ints, metric_name, figsize=(1200, 400), shaded=True
):
    """
    Create an interactive Plotly visualisation of the cumulative mean and
    confidence intervals for each replication.

    This plot displays:
      - The running (cumulative) mean of a performance metric.
      - Lower and upper bounds of the confidence interval at each replication.
      - Annotated deviation (as % of mean) on hover.
      - A vertical dashed line at the minimum number of replications (`n_reps`)
        required to achieve the target precision.

    Parameters
    ----------
    n_reps: int
        Minimum number of replications needed to achieve desired precision
        (typically the output of `confidence_interval_method`).
    conf_ints: pandas.DataFrame
        Results DataFrame from `confidence_interval_method`, containing
        columns: `"Cumulative Mean"`, `"Lower Interval"`, `"Upper Interval"`,
        etc.
    metric_name: str
        Name of the performance metric displayed in the y-axis label.
    figsize: tuple, optional (default=(1200,400))
        Figure size in pixels: (width, height).
    shaded: bool, optional
        If True, use shaded CI region. If False, use dashed lines (legacy).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Calculate relative deviations
    deviation_pct = (
        (conf_ints["Upper Interval"] - conf_ints["Cumulative Mean"])
        / conf_ints["Cumulative Mean"]
        * 100
    ).round(2)

    # Confidence interval
    if shaded:
        # Shaded style
        fig.add_trace(
            go.Scatter(
                x=conf_ints.index,
                y=conf_ints["Upper Interval"],
                mode="lines",
                line={"width": 0},
                name="Upper Interval",
                text=[f"Deviation: {d}%" for d in deviation_pct],
                hoverinfo="x+y+name+text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=conf_ints.index,
                y=conf_ints["Lower Interval"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(0,176,185,0.2)",
                name="Lower Interval",
                text=[f"Deviation: {d}%" for d in deviation_pct],
                hoverinfo="x+y+name+text",
            )
        )
    else:
        # Dashed lines style
        for col, color, dash in zip(
            ["Lower Interval", "Upper Interval"],
            ["lightblue", "lightblue"],
            ["dot", "dot"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=conf_ints.index,
                    y=conf_ints[col],
                    line={"color": color, "dash": dash},
                    name=col,
                    text=[f"Deviation: {d}%" for d in deviation_pct],
                    hoverinfo="x+y+name+text",
                )
            )

    # Cumulative mean line
    fig.add_trace(
        go.Scatter(
            x=conf_ints.index,
            y=conf_ints["Cumulative Mean"],
            line={"color": "blue", "width": 2},
            name="Cumulative Mean",
            hoverinfo="x+y+name"
        )
    )

    # Vertical threshold line
    fig.add_shape(
        type="line",
        x0=n_reps,
        x1=n_reps,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "red", "dash": "dash"},
    )

    # Configure layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        yaxis_title=f"Cumulative Mean: {metric_name}",
        hovermode="x unified",
        showlegend=True,
    )

    return fig


@runtime_checkable
class ReplicationsAlgorithmModelAdapter(Protocol):
    """
    Adapter pattern for the "Replications Algorithm".

    All models that use ReplicationsAlgorithm must provide a
    single_run(replication_number) interface.
    """

    def single_run(self, replication_number: int) -> float:
        """
        Perform a unique replication of the model. Return a performance measure
        """


# pylint: disable=too-many-instance-attributes
class ReplicationsAlgorithm:
    """
    Automatically determine the number of simulation replications needed
    to achieve and maintain a target confidence interval precision.

    Implements the *Replications Algorithm* from Hoad, Robinson & Davies
    (2010), which combines:
      - The **confidence interval method** to assess whether the
        target precision has been met.
      - A **sequential look-ahead procedure** to verify that
        precision remains stable in additional replications.

    Attributes
    ----------
    alpha : float
        Significance level for confidence interval calculations.
    half_width_precision : float
        Target CI half-width precision (i.e. percentage deviation of the
        confidence interval from the mean).
    initial_replications : int
        Number of replications to run before evaluating precision.
    look_ahead : int
        Number of additional replications to simulate for stability checks
        (adjusted proportionally when `n > 100`).
    replication_budget : int
        Maximum number of replications allowed.
    verbose : bool
        If True, prints the current replication count during execution.
    observer : ReplicationObserver or None
        Optional observer object to record statistics at each update.
    n : int
        Current replication count (updated during execution).
    _n_solution : int
        Solution replication count once convergence is met (or replication
        budget if not met).
    stats : OnlineStatistics or None
        Tracks running mean, variance, and confidence interval metrics.

    Notes
    -----
    Only works with a single performance measure.

    References
    ----------
    Hoad, K., Robinson, S., & Davies, R. (2010). Automated selection of the
    number of replications for a discrete-event simulation. *Journal of the
    Operational Research Society*, 61(11), 1632-1644.
    https://www.jstor.org/stable/40926090
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        alpha: Optional[float] = 0.05,
        half_width_precision: Optional[float] = 0.1,
        initial_replications: Optional[int] = 3,
        look_ahead: Optional[int] = 5,
        replication_budget: Optional[float] = 1000,
        verbose: Optional[bool] = False,
        observer: Optional[ReplicationObserver] = None,
    ):
        """
        Initialise the replications algorithm

        Parameters
        ----------
        alpha: float, optional (default = 0.05)
            Significance level for confidence interval calculations
            (CI level = 100 * (1 - alpha) %).
        half_width_precision: float, optional (default = 0.1)
            Target CI half-width precision (i.e. percentage deviation of the
            confidence interval from the mean).
        initial_replications : int
            Number of replications to run before evaluating precision.
        look_ahead: int, optional (default = 5)
            Number of additional replications to simulate for stability checks.
            When the number of replications n <= 100 the value of look ahead
            is used. When n > 100 then look_ahead / 100 * max(n, 100) is used.
        replication_budget: int, optional (default = 1000)
            Maximum number of replications allowed; algorithm stops if not
            converged by then. Useful for larger models where replication
            runtime is a constraint.
        verbose: bool, optional (default=False)
            If True, prints replication count progress.
        observer: ReplicationObserver, optional (default=None)
            Optional observer to record statistics after each replication. For
            example `ReplicationTabulizer` to return a table equivalent to
            `confidence_interval_method`.

        Raises
        ------
        ValueError
            If parameter types or values are invalid (checked in
            `valid_inputs()`).
        """
        self.alpha = alpha
        self.half_width_precision = half_width_precision
        self.initial_replications = initial_replications
        self.look_ahead = look_ahead
        self.replication_budget = replication_budget
        self.verbose = verbose

        # Initially set n to number of initial replications
        self.n = self.initial_replications

        self._n_solution = self.replication_budget
        self.observer = observer
        self.stats = None

        # Check validity of provided parameters
        self.valid_inputs()

    def valid_inputs(self):
        """
        Checks validity of provided parameters.

        Ensures:
          - `initial_replications` and `look_ahead` are non-negative integers.
          - `half_width_precision` is > 0.
          - `replication_budget` is not less than `initial_replications`.

        Raises
        ------
        ValueError
            If any conditions are not met.
        """
        for p in [self.initial_replications, self.look_ahead]:
            if not isinstance(p, int) or p < 0:
                raise ValueError(f'{p} must be a non-negative integer.')

        if self.half_width_precision <= 0:
            raise ValueError('half_width_precision must be greater than 0.')

        if self.replication_budget < self.initial_replications:
            raise ValueError(
                'replication_budget must be less than initial_replications.')

    def _klimit(self) -> int:
        """
        Determine the number of additional replications to check after the
        desired confidence interval precision is first reached.

        The look-ahead count scales with the total number of replications:
        - If n ≤ 100, returns the fixed `look_ahead` value.
        - If n > 100, returns `look_ahead / 100 * max(n, 100)`, rounded down.

        Returns
        -------
        int
            Number of additional replications to check precision stability.
            Returned value is always rounded down to the nearest integer.
        """
        return int((self.look_ahead / 100) * max(self.n, 100))

    def select(self, model: ReplicationsAlgorithmModelAdapter) -> int:
        """
        Executes the replication algorithm, determining the necessary number
        of replications to achieve and maintain the desired precision.

        The process:
          1. Runs `initial_replications` of the model.
          2. Updates running statistics and calculates CI precision.
          3. If precision met, tests stability via the look-ahead procedure.
          4. Stops when stable precision is achieved or budget is exhausted.

        Parameters
        ----------
        model : ReplicationsAlgorithmModelAdapter
            Simulation model implementing `single_run(replication_index)`.

        Returns
        -------
        int
            Number of replications required to achieve and maintain target
            precision. If convergence is not reached within the budget, returns
            the budget value.

        Raises
        ------
        ValueError
            If the provided `model` is not an instance of
            `ReplicationsAlgorithmModelAdapter`.

        Warns
        -----
        UserWarning
            If convergence is not reached within the allowed replication
            budget.
        """
        # Check validity of provided model
        if not isinstance(model, ReplicationsAlgorithmModelAdapter):
            raise ValueError(ALG_INTERFACE_ERROR)

        converged = False

        # Run initial replications of model
        x_i = [
            model.single_run(rep) for rep in range(self.initial_replications)]

        # Initialise running mean and std dev
        self.stats = OnlineStatistics(
            data=np.array(x_i), alpha=self.alpha, observer=self.observer
        )

        while not converged and self.n <= self.replication_budget:
            if self.n > self.initial_replications:
                # Update X_n and d_req
                self.stats.update(x_i)

            # Precision achieved?
            if self.stats.deviation <= self.half_width_precision:

                # Store current solution
                self._n_solution = self.n
                converged = True

                if self._klimit() > 0:
                    k = 1

                    # Look ahead loop
                    while converged and k <= self._klimit():
                        if self.verbose:
                            print(f"{self.n+k}", end=", ")

                        # Simulate replication n + k
                        x_i = model.single_run(self.n + k - 1)

                        # Update X_n and d_req
                        self.stats.update(x_i)

                        # Check new precision
                        if self.stats.deviation > self.half_width_precision:
                            # Precision not maintained
                            converged = False
                            self.n += k
                        else:
                            k += 1

                # Terminate if precision maintained over lookahead
                if converged:
                    return self._n_solution

            # Precision not achieved/maintained so simulate another replication
            self.n += 1
            if self.verbose:
                print(f"{self.n}", end=", ")
            x_i = model.single_run(self.n - 1)

        # If code gets to here then no solution found within budget.
        warnings.warn(
            "Algorithm did not converge for metric'. "
            + "Returning replication budget as solution"
        )
        return self._n_solution
