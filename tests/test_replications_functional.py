"""Replication functional tests

Functional testing for code related to selecting the number of replications.

Credit: Some of these tests are adapted from-
    Heather, A. Monks, T. (2025). Python DES RAP Template. GitHub.
    https://github.com/pythonhealthdatascience/rap_template_python_des.
"""

import pandas as pd
import pytest

from tests.dummy_model import DummySimulationModel
from sim_tools.output_analysis import (confidence_interval_method,
                                       ReplicationsAlgorithm,
                                       ReplicationTabulizer)


def test_consistent_methods():
    """
    Check that ReplicationsAlgorithm and confidence_interval_method generate
    consistent results.
    """
    # Define parameters required by both methods
    reps = 20
    alpha = 0.05
    desired_precision = 0.05

    # Set up model
    model = DummySimulationModel(mean=70, std_dev=4)

    # Run the confidence interval method
    results = [model.single_run(rep) for rep in range(1, reps+1)]
    ci_method_n_reps, ci_method_summary_table = confidence_interval_method(
        replications=results,
        alpha=alpha,
        desired_precision=desired_precision,
        min_rep=0,
        decimal_places=2)

    # Run the algorithm
    observer = ReplicationTabulizer()
    analyser = ReplicationsAlgorithm(
        alpha=alpha,
        half_width_precision=desired_precision,
        initial_replications=reps,
        look_ahead=0,
        replication_budget=reps,
        verbose=False,
        observer=observer
    )
    algorithm_n_reps = analyser.select(
        DummySimulationModel())
    algorithm_summary_table = observer.summary_table()

    # Compare the n_reps
    assert ci_method_n_reps == algorithm_n_reps, (
        f"Expected same solution, but n_reps was {ci_method_n_reps} from " +
        f"confidence_interval_method and {algorithm_n_reps} from " +
        "ReplicationsAlgorithm."
    )

    # Compare the summary tables
    pd.testing.assert_frame_equal(
        ci_method_summary_table, algorithm_summary_table)


def test_ci_method_output():
    """
    Check that the output from confidence_interval_method meets our
    expectations.
    """
    # Create empty list to store errors (else if each were an assert
    # statement, it would stop after the first)
    errors = []

    # Choose a number of replications to run for
    reps = 20

    # Run the model
    model = DummySimulationModel(mean=10, std_dev=0.5)
    results = [model.single_run(rep) for rep in range(1, reps+1)]

    # Run the confidence interval method
    n_reps, summary_table = confidence_interval_method(
        replications=results,
        alpha=0.05,
        desired_precision=0.05,
        min_rep=3,
        decimal_places=2)

    # Check that the results dataframe contains the right number of rows
    if not len(summary_table) == reps:
        errors.append(
            f"Ran {reps} replications but summary_table only has " +
            f"{len(summary_table)} entries.")

    # Check that the replications are appropriately numbered
    if not min(summary_table.index) == 1:
        errors.append(
            "Minimum replication in summary_table should be 1 but it is " +
            f"{min(summary_table.index)}.")

    # Check that min_reps is no more than the number run
    if not n_reps <= reps:
        errors.append(
            "The minimum number of replications required as returned by the " +
            "confidence_interval_method should be less than the number we " +
            f"ran ({reps}) but it was {n_reps}.")

    # Check if there were any errors
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


def test_algorithm_initial():
    """
    Check that the solution from the ReplicationsAlgorithm is as expected when
    there is a high number of initial replications specified.
    """
    # Set up the model with mean and std_dev that would be solved < 200 reps
    model = DummySimulationModel(mean=100, std_dev=5)

    # Set up the algorithm with a high number of initial replications and no
    # look ahead period
    initial_replications = 200
    observer = ReplicationTabulizer()
    analyser = ReplicationsAlgorithm(
        alpha=0.05,
        half_width_precision=0.05,
        initial_replications=initial_replications,
        look_ahead=0,
        replication_budget=1000,
        verbose=False,
        observer=observer
    )

    # Run the algorithm and get results
    n_reps = analyser.select(model)
    summary_table = observer.summary_table()

    # Check that soution equals initial_replications
    assert n_reps == initial_replications

    # Check that number of rows in summary table equals initial_replications
    assert len(summary_table) == initial_replications


def test_algorithm_nosolution():
    """
    Check that running for less than 3 replications in total will result in no
    solution, and that a warning message is then created.
    """
    # Set up the model with mean and std_dev that would be solved > 3 reps
    model = DummySimulationModel(mean=1, std_dev=1)

    # Set up algorithm to run max of 2 replications
    reps = 2
    observer = ReplicationTabulizer()
    analyser = ReplicationsAlgorithm(
        alpha=0.05,
        half_width_precision=0.05,
        initial_replications=reps,
        look_ahead=0,
        replication_budget=reps,
        verbose=False,
        observer=observer
    )

    # Run algorithm, checking that it produces a warning
    with pytest.warns():
        n_reps = analyser.select(model)

    # Check that solution is equal to max replications
    assert n_reps == reps

    # Check that the summary tables has no more than 2 rows
    summary_table = observer.summary_table()
    assert len(summary_table) < reps + 1
