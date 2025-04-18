"""
Common validation methods for sim-tools functions and methods.

This module provides a composable validation framework for checking 
input parameters for sim-tools code. Each validator follows the
Single Responsibility Principle, checking exactly one constraint, and can be 
combined with others through the `validate()` function.

Examples
--------
>>> from validation import validate, is_numeric, is_positive
>>> 
>>> # Validate that a parameter is a positive number
>>> def calculate_square_root(x):
...     validate(x, "x", is_numeric, is_positive)
...     return math.sqrt(x)
>>> 
>>> # Validate a probability parameter
>>> def bernoulli_trial(p):
...     validate(p, "p", is_numeric, is_probability)
...     return random.random() < p

Available Validators
-------------------
is_numeric : Checks if value is a number (int or float)
is_positive : Checks if value is positive (greater than zero)
is_probability : Checks if value is between 0 and 1 inclusive

Notes
-----
Validation functions raise appropriate exceptions with descriptive messages:
- TypeError for type validation failures
- ValueError for value constraint failures
"""

from typing import Any, Callable

# Type for validator functions
ValidatorFunc = Callable[[Any, str], None]

# Individual validator functions - each with a single responsibility
def is_numeric(value: Any, name: str) -> None:
    """Validates that a value is a number (int or float)."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")

def is_positive(value: Any, name: str) -> None:
    """Validates that a value is positive (> 0)."""
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")

def is_non_negative(value: Any, name: str) -> None:
    """Validates that a value is greater than or equal to 0."""
    if not value >= 0:
        raise ValueError(f"{name} must be greater than or equal to 0, got {value}")


def is_probability(value: Any, name: str) -> None:
    """Validates that a value is a valid probability (between 0 and 1)."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")

def validate(value: Any, name: str, *validators: ValidatorFunc) -> None:
    """Generic validation function that applies multiple validators to a value."""
    for validator in validators:
        validator(value, name)

def is_ordered_pair(low: Any, high: Any, low_name: str = "low", high_name: str = "high") -> None:
    """Validates that two values are in ascending order (low < high)."""
    if not low < high:
        raise ValueError(f"{low_name} must be less than {high_name}, got {low} >= {high}")

def is_ordered_triplet(low: Any, middle: Any, high: Any, 
                      low_name: str = "low", middle_name: str = "middle", high_name: str = "high") -> None:
    """Validates that three values are in ascending order (low < middle < high)."""
    if not low < middle:
        raise ValueError(f"{low_name} must be less than {middle_name}, got {low} >= {middle}")
    if not middle < high:
        raise ValueError(f"{middle_name} must be less than {high_name}, got {middle} >= {high}")
 

